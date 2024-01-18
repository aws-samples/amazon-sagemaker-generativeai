import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import s3 = require('aws-cdk-lib/aws-s3');
import efs = require('aws-cdk-lib/aws-efs');
import * as datasync from 'aws-cdk-lib/aws-datasync';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as tcdk from 'amazon-textract-idp-cdk-constructs';
import lambda = require('aws-cdk-lib/aws-lambda');
import * as customResources from 'aws-cdk-lib/custom-resources';
import { aws_opensearchservice as opensearch } from 'aws-cdk-lib';
import * as kinesisfirehose from 'aws-cdk-lib/aws-kinesisfirehose';
import * as glue from '@aws-cdk/aws-glue-alpha';
import dynamodb = require('aws-cdk-lib/aws-dynamodb');
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';


export class BackendStack extends cdk.Stack {
  readonly backendStackProps: {
    vpc: ec2.IVpc;
    openSearchDomain: string;
    openSearchDomainArn: string;
    firehoseName: string;
    firehoseArn: string;
    bucket: s3.Bucket;
  }

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

     // Get context values
     const endpointEmbed = this.node.tryGetContext('embeddingsModelEndpointName');
     const endpointText = this.node.tryGetContext('textModelEndpointName');

    // S3 Bucket
    const contentBucket = new s3.Bucket(this, 'DocumentsBucket', { 
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      encryption: s3.BucketEncryption.S3_MANAGED,
      serverAccessLogsPrefix: 'accesslogs/',
      enforceSSL: true,
      eventBridgeEnabled: true,
      objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_PREFERRED,

    });

    // VPC
    const vpc = new ec2.Vpc(this, 'VPC', {
      natGateways: 1,
      gatewayEndpoints: {
        S3: {
          service: ec2.GatewayVpcEndpointAwsService.S3,
        },
      },
    });
    vpc.addFlowLog('FlowLogS3', {
      destination: ec2.FlowLogDestination.toS3(contentBucket, 'flowlogs/')
    });
    vpc.addInterfaceEndpoint('EcrDockerEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
    });
    vpc.addInterfaceEndpoint('KmsEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.KMS,
    }); 

    // EFS (used as a drop point for PDF documents)
    const fileSystem = new efs.FileSystem(this, 'PdfFileSystem', {
      vpc: vpc,
      encrypted: true,
      enableAutomaticBackups: true,
      performanceMode: efs.PerformanceMode.GENERAL_PURPOSE, // default
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      },
    });
    fileSystem.connections.allowDefaultPortFrom(ec2.Peer.ipv4(vpc.vpcCidrBlock));

    // Jump host
    const jumpHostSG = new ec2.SecurityGroup(this, 'JumpHostSecurityGroup', {
      vpc,
      description: 'Allow all VPC traffic',
      allowAllOutbound: true,
    });
    const jumpHost = new ec2.BastionHostLinux(this, 'JumpHost', {
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
      machineImage: ec2.MachineImage.latestAmazonLinux2023(),
      securityGroup: jumpHostSG
    });
    contentBucket.grantRead(jumpHost);
    jumpHost.instance.role.addManagedPolicy(iam.ManagedPolicy.fromManagedPolicyArn(this, 'ssmpolicy', 'arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore'));
    jumpHost.instance.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'elasticfilesystem:ClientRootAccess',
          'elasticfilesystem:ClientWrite',
          'elasticfilesystem:ClientMount',
          'elasticfilesystem:DescribeMountTargets'
        ],
        resources: [fileSystem.fileSystemArn]
      })
    );
    jumpHost.instance.userData.addCommands("yum check-update -y",    
      "yum upgrade -y",                                 
      "yum install -y amazon-efs-utils",                
      "yum install -y nfs-utils",                       
      "file_system_id_1=" + fileSystem.fileSystemId,
      "efs_mount_point_1=/mnt/efs/fs1",
      "mkdir -p \"${efs_mount_point_1}\"",
      "test -f \"/sbin/mount.efs\" && echo \"${file_system_id_1}:/ ${efs_mount_point_1} efs _netdev,tls\" >> /etc/fstab || " +
      "echo \"${file_system_id_1}.efs." + cdk.Stack.of(this).region + ".amazonaws.com:/ ${efs_mount_point_1} nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport,_netdev 0 0\" >> /etc/fstab",
      "test -f \"/sbin/mount.efs\" && echo \"[client-info]\" >> /etc/amazon/efs/efs-utils.conf && echo \"source=liw\" >> /etc/amazon/efs/efs-utils.conf",
      "retryCnt=15; waitTime=30; while true; do mount -a -t efs,nfs4 defaults; if [ $? = 0 ] || [ $retryCnt -lt 1 ]; then echo File system mounted successfully; break; fi; echo File system not available, retrying to mount.; ((retryCnt--)); sleep $waitTime; done;")

    // DataSync
    const s3AccessPolicy = new iam.PolicyDocument({
      statements: [new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["s3:Abort*",
          "s3:DeleteObject*",
          "s3:GetBucket*",
          "s3:GetObject*",
          "s3:List*",
          "s3:PutObject",
          "s3:PutObjectLegalHold",
          "s3:PutObjectRetention",
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging"
        ],
        resources: [contentBucket.bucketArn, contentBucket.bucketArn + "/*"]
      })]
    });
    const datasyncServiceRole = new iam.Role(this, 'TextractServiceRole', {
      assumedBy: new iam.ServicePrincipal('datasync.amazonaws.com'),
      inlinePolicies: {S3AccessPolicy: s3AccessPolicy}
    });
    datasyncServiceRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'elasticfilesystem:ClientRootAccess',
          'elasticfilesystem:ClientWrite',
          'elasticfilesystem:ClientMount',
          'elasticfilesystem:DescribeMountTargets'
        ],
        resources: [fileSystem.fileSystemArn]
      })
    );
    datasyncServiceRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['ec2:DescribeAvailabilityZones'],
        resources: ['*']
      })
    );
    // datasyncServiceRole.addToPolicy(
    //   new iam.PolicyStatement({
    //     effect: iam.Effect.ALLOW,
    //     actions: ['s3:List*'],
    //     resources: [contentBucket.bucketArn, contentBucket.bucketArn + "/*"]
    //   })
    // );
    // contentBucket.grantReadWrite(datasyncServiceRole);
    const datasyncLocationS3 = new datasync.CfnLocationS3(this, 'TargetLocationS3', {
      s3Config: {
        bucketAccessRoleArn: datasyncServiceRole.roleArn
      },
      s3BucketArn: contentBucket.bucketArn,
      subdirectory: 'datasync',
    });
    const logGroup = new logs.LogGroup(this, 'DatasyncLogGroup', {
      retention: logs.RetentionDays.ONE_WEEK,
    });
    logGroup.grantWrite(new iam.ServicePrincipal('datasync.amazonaws.com'));
    const efsSgs = fileSystem.connections.securityGroups;
    const sgArns: string[] = [];
    efsSgs.forEach(sg => {
      sgArns.push("arn:aws:ec2:" + this.region + ":" + this.account + ":security-group/" + sg.securityGroupId);
    });
    const datasyncLocationEFS = new datasync.CfnLocationEFS(this, 'SrcLocationEFS', {
      ec2Config: {
        securityGroupArns: sgArns,
        subnetArn: "arn:aws:ec2:" + this.region + ":" + this.account + ":subnet/" + vpc.privateSubnets[0].subnetId
      },
      efsFilesystemArn: fileSystem.fileSystemArn,
      fileSystemAccessRoleArn: datasyncServiceRole.roleArn,
      inTransitEncryption: 'TLS1_2',
    });
    datasyncLocationEFS.node.addDependency(fileSystem);
    const datasyncTask = new datasync.CfnTask(this, 'DatasyncTask', {
      destinationLocationArn: datasyncLocationS3.attrLocationArn,
      sourceLocationArn: datasyncLocationEFS.attrLocationArn,
    
      // the properties below are optional
      cloudWatchLogGroupArn: logGroup.logGroupArn,
      includes: [{
        filterType: 'SIMPLE_PATTERN',
        value: '/ingest/*',
      }],
      schedule: {
        scheduleExpression: 'rate(1 hour)',
      },
    });

    // OpenSearch
    const openSearchDomain = new opensearch.Domain(this, 'openSearch-Domain', {
      version: opensearch.EngineVersion.OPENSEARCH_2_5,
      enableVersionUpgrade: true,
      vpc: vpc,
      ebs: {
        enabled: true,
        volumeSize: 300
      },
      zoneAwareness: {
        enabled: true
      },
      capacity: {
        dataNodes: 4,
        dataNodeInstanceType: "r6g.2xlarge.search",
        masterNodes: 3,
        masterNodeInstanceType: "r6g.2xlarge.search"
      },
      accessPolicies: [
        new cdk.aws_iam.PolicyStatement({
          actions: ['es:ESHttp*',],
          resources: ['*'],
          effect: cdk.aws_iam.Effect.ALLOW,
          principals: [new cdk.aws_iam.AccountRootPrincipal]
        })],
        enforceHttps: true,
        nodeToNodeEncryption: true,
        encryptionAtRest: {
          enabled: true
        },
        removalPolicy: cdk.RemovalPolicy.DESTROY
    })
    openSearchDomain.connections.allowFrom(ec2.Peer.ipv4(vpc.vpcCidrBlock), ec2.Port.allTraffic(), 'All traffic from VPC to OpenSearch')
    const createOsIndexLambda = new lambda.Function( this, `osIndexCustomResourceLambda`, {
        runtime: lambda.Runtime.PYTHON_3_11,
        vpc: vpc,
        code: lambda.Code.fromAsset( "lambda/ossetup", {
          bundling: {
            image: lambda.Runtime.PYTHON_3_11.bundlingImage,
            command: [
              'bash', '-c',
              'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
            ],
          },
        }),
        handler: 'lambda_function.on_event',
        tracing: lambda.Tracing.ACTIVE,
        timeout: cdk.Duration.minutes(1),
        memorySize: 1024,
        environment: {
          DOMAINURL: openSearchDomain.domainEndpoint,
          INDEX: 'embeddings'
        }
      }
    );
    createOsIndexLambda.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'es:ESHttp*'
          ],
        resources: [openSearchDomain.domainArn + "/*"]
      })
    )
    const customResourceProvider = new customResources.Provider( this, `osIndexCustomResourceProvider`, {
        onEventHandler: createOsIndexLambda,
      }
    );
    new cdk.CustomResource(this, `customResourceConfigureIndex`, {
      serviceToken: customResourceProvider.serviceToken,
    });

    // Firehose
    const fhLogGroup = new logs.LogGroup(this, 'FhLogGroup', {
      retention: logs.RetentionDays.ONE_WEEK,
    });
    const fhLogStream = new logs.LogStream(this, 'FhLogStream', {
      logGroup: fhLogGroup,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const fhLogStreamPrompts = new logs.LogStream(this, 'FhLogStreamPrompts', {
      logGroup: fhLogGroup,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    const fhRole = new iam.Role(this, 'FhRole', {
      assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
    });
    contentBucket.grantReadWrite(fhRole);
    fhRole.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'logs:PutLogEvents',
        ],
        resources: ['*']
      })
    );
    const s3DestinationConfigurationProperty: kinesisfirehose.CfnDeliveryStream.S3DestinationConfigurationProperty = {
      bucketArn: contentBucket.bucketArn,
      roleArn: fhRole.roleArn,
      bufferingHints: {
        intervalInSeconds: 60,
        sizeInMBs: 5,
      },
      cloudWatchLoggingOptions: {
        enabled: true,
        logGroupName: fhLogGroup.logGroupName,
        logStreamName: fhLogStream.logStreamName
      },
      compressionFormat: 'GZIP',
      prefix: 'embeddingarchive/',
    };
    const s3DestinationConfigurationPropertyPrompts: kinesisfirehose.CfnDeliveryStream.S3DestinationConfigurationProperty = {
      bucketArn: contentBucket.bucketArn,
      roleArn: fhRole.roleArn,
      bufferingHints: {
        intervalInSeconds: 60,
        sizeInMBs: 5,
      },
      cloudWatchLoggingOptions: {
        enabled: true,
        logGroupName: fhLogGroup.logGroupName,
        logStreamName: fhLogStreamPrompts.logStreamName
      },
      compressionFormat: 'GZIP',
      prefix: 'promptarchive/',
    };
    const fh_embed = new kinesisfirehose.CfnDeliveryStream(this, "Firehose", {
      deliveryStreamType: "DirectPut",
      s3DestinationConfiguration: s3DestinationConfigurationProperty
    });
    const fh_prompts = new kinesisfirehose.CfnDeliveryStream(this, "FirehosePrompts", {
      deliveryStreamType: "DirectPut",
      s3DestinationConfiguration: s3DestinationConfigurationPropertyPrompts
    });

    // Step Functions
    const decider_task = new tcdk.TextractPOCDecider(this, "Decider", {});
    const textract_async_task = new tcdk.TextractGenericAsyncSfnTask( this, "TextractAsync", {
      s3OutputBucket: contentBucket.bucketName,
      s3TempOutputPrefix: 'textract/out',
      integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
      taskTimeout: sfn.Timeout.duration(cdk.Duration.hours(1)),
      input: sfn.TaskInput.fromObject({
        Token: sfn.JsonPath.taskToken,
        ExecutionId: sfn.JsonPath.stringAt('$$.Execution.Id'),
        Payload: sfn.JsonPath.entirePayload
      }),
      resultPath: '$.textract_result'
    });
    const lambda_textract_post_processing_function = new lambda.DockerImageFunction(this, 'LambdaTextractPostProcessing', {
      code: lambda.DockerImageCode.fromImageAsset('lambda/textractpostprocessor'),
      tracing: lambda.Tracing.ACTIVE,
      timeout: cdk.Duration.minutes(15),
      memorySize: 2048,
      environment: {
        SKIP_PAGES: "CONTENTS,TABLE OF CONTENTS,FOREWORDS, ANNEXES,Table of Contents,ACRONYMS, ABBREVIATIONS",
        NO_LINES_HEADER: "3",
        NO_LINES_FOOTER: "10",
        FILTER_PARA_WORDS:"10"
      }
    });
    contentBucket.grantReadWrite(lambda_textract_post_processing_function);
    const textractAsyncCallTask = new tasks.LambdaInvoke(this, "TextractPostProcessorTask", {
      lambdaFunction: lambda_textract_post_processing_function,
      resultPath: "$",
      outputPath: "$.Payload"
    });
    
    const csvToEmbeddingFn = new lambda.Function(this, 'csvToEmbeddingFn', {
      runtime: lambda.Runtime.PYTHON_3_11,
      vpc: vpc,
      code: lambda.Code.fromAsset('lambda/embeddingprocessor', {
          bundling: {
            image: lambda.Runtime.PYTHON_3_11.bundlingImage,
            command: [
              'bash', '-c',
              'pip install -r requirements.txt -t /asset-output && cp -au . /asset-output'
            ],
          },
      }),
      handler: 'lambda_function.lambda_handler',
      tracing: lambda.Tracing.ACTIVE,
      timeout: cdk.Duration.minutes(1),
      memorySize: 1024,
      environment: {
        DOMAINURL: openSearchDomain.domainEndpoint,
        INDEX: 'embeddings',
        FIREHOSE: fh_embed.ref,
        EMBEDDINGS_MODEL_ENDPOINT: endpointEmbed
      }
    });
    csvToEmbeddingFn.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:InvokeEndpoint"],
        resources: ["arn:aws:sagemaker:" + this.region + ":" + this.account + ":endpoint/" + endpointEmbed]
      })
    );
    csvToEmbeddingFn.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'firehose:PutRecordBatch',
        ],
        resources: [fh_embed.attrArn]
      })
    );
// Add policy to allow access to OpenSearch
    csvToEmbeddingFn.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'es:ESHttp*'
          ],
        resources: [openSearchDomain.domainArn + "/*"]
      })
    );

    const csvToEmbedding = new sfn.CustomState(this, "CsvToEmbeddingMap", {
      stateJson: {
        Type: "Map",
        ItemProcessor: {
          ProcessorConfig: {
            Mode: "DISTRIBUTED",
            ExecutionType: "EXPRESS"
          },
          StartAt: "LambdaBatchProcessor",
          States: {
            LambdaBatchProcessor: {
              Type: "Task",
              Resource: "arn:aws:states:::lambda:invoke",
              OutputPath: "$.Payload",
              Parameters: {
                "Payload.$": "$",
                "FunctionName": csvToEmbeddingFn.functionArn
              },
              Retry: [
                {
                  ErrorEquals: [
                    "Lambda.ServiceException",
                    "Lambda.AWSLambdaException",
                    "Lambda.SdkClientException",
                    "Lambda.TooManyRequestsException"
                  ],
                  IntervalSeconds: 2,
                  MaxAttempts: 6,
                  BackoffRate: 2
                }
              ],
              End: true
            }
          }
        },
        ItemReader: {
          Resource: "arn:aws:states:::s3:getObject",
          ReaderConfig: {
            InputType: "CSV",
            CSVHeaderLocation: "FIRST_ROW"
          },
          Parameters: {
            "Bucket.$": "$.csvBucket",
            "Key.$": "$.csvPath"
          }
        },
        MaxConcurrency: 5,
        Label: "CsvToEmbedding",
        ItemBatcher: {
          MaxItemsPerBatch: 5,
          MaxInputBytesPerBatch: 2048
        }
      }
    })
    const async_chain = sfn.Chain.start(textract_async_task).next(textractAsyncCallTask).next(csvToEmbedding);
    const workflow_chain = sfn.Chain.start(decider_task).next(async_chain);
    const sfnPdfToText = new sfn.StateMachine(this, 'StateMachinePdfToText', {
      definition: workflow_chain,
      timeout: cdk.Duration.minutes(30),
    });
    csvToEmbeddingFn.grantInvoke(sfnPdfToText);
    contentBucket.grantReadWrite(sfnPdfToText);
    sfnPdfToText.addToRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'states:StartExecution',
          'states:DescribeExecution',
          'states:StopExecution'
        ],
        resources: ['arn:aws:states:' + this.region + ':' + this.account + ':stateMachine:StateMachinePdfToText*']
      })
    );

    // Event Bridge
    const rulePdfToText = new events.Rule(this, 'rulePdfToText', {
      eventPattern: {
        source: ["aws.s3"],
        detailType: ["Object Created"],
        detail: {
          bucket: {
            name: [contentBucket.bucketName]
          },
          object: {
            key: [{
              prefix: "datasync/ingest"
            }]
          }
        }
      },
    });
    const dlq_sfn = new sqs.Queue(this, 'DeadLetterQueueSFN');
    const role_events = new iam.Role(this, 'RoleEventBridge', {
      assumedBy: new iam.ServicePrincipal('events.amazonaws.com'),
    });
    rulePdfToText.addTarget(new targets.SfnStateMachine(sfnPdfToText, {
      input: events.RuleTargetInput.fromObject({ s3Path: `s3://${events.EventField.fromPath('$.detail.bucket.name')}/${events.EventField.fromPath('$.detail.object.key')}` }),
      deadLetterQueue: dlq_sfn,
      role: role_events
    }));

    //DynamoDB table with drift detection baselines for both reference data and prompts
    // Fields:
    // jobtype (PK) [BASELINE|SNAPSHOT]
    // jobdate (RK) 
    // varpc [The number of principal components required to explain 95% of the variance]
    // clutersizes [Number of items in each cluster]
    // inertia [Squared distances]
    // score [Silhouette score]
    const driftTable = new dynamodb.Table(this, 'DriftTable', {
      partitionKey: { name: 'jobtype', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'jobdate', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });
    // Fields:
    // jobtypedate (jobtype-jobdate) (PK)
    // centroid (number) (RK)
    // center
    const driftTableCentroids = new dynamodb.Table(this, 'DriftTableCentroids', {
      partitionKey: { name: 'jobtypedate', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'centroid', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });
    const driftTablePrompts = new dynamodb.Table(this, 'DriftTablePrompts', {
      partitionKey: { name: 'jobtype', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'jobdate', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });
    const driftTableCentroidsPrompts = new dynamodb.Table(this, 'DriftTableCentroidsPrompts', {
      partitionKey: { name: 'jobtypedate', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'centroid', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });
    // Fields:
    // jobtype (PK) [DISTANCE]
    // jobdate (RK) 
    // mean 
    // median
    // stdev
    const distanceTable = new dynamodb.Table(this, 'DistanceTable', {
      partitionKey: { name: 'jobtype', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'jobdate', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED
    });

    // Glue job for embedding drift and prompt distance
    const driftJob = new glue.Job(this, 'EmbeddingDriftJob', {
      jobName: 'embedding-drift-analysis',
      executable: glue.JobExecutable.pythonEtl({
        glueVersion: glue.GlueVersion.V4_0,
        script: new glue.AssetCode('scripts/glue-drift-job.py'),
        pythonVersion: glue.PythonVersion.THREE,
      }),
      workerType: glue.WorkerType.G_2X,
      workerCount: 10,
      description: 'Embedding drift analysis',
      defaultArguments: {
        '--data_path': 's3://' + contentBucket.bucketName + '/embeddingarchive/',
        '--out_table': driftTable.tableName,
        '--centroid_table': driftTableCentroids.tableName,
        '--job_type': 'SNAPSHOT'
      }
    });
    contentBucket.grantRead(driftJob);
    driftTable.grantReadWriteData(driftJob);
    driftTableCentroids.grantReadWriteData(driftJob);
    driftTablePrompts.grantReadWriteData(driftJob);
    driftTableCentroidsPrompts.grantReadWriteData(driftJob);
    const distanceJob = new glue.Job(this, 'EmbeddingDistanceJob', {
      jobName: 'embedding-distance-analysis',
      executable: glue.JobExecutable.pythonEtl({
        glueVersion: glue.GlueVersion.V4_0,
        script: new glue.AssetCode('scripts/glue-ref-prompt-job.py'),
        pythonVersion: glue.PythonVersion.THREE,
      }),
      workerType: glue.WorkerType.G_2X,
      workerCount: 10,
      description: 'Embedding distance analysis',
      defaultArguments: {
        '--data_path': 's3://' + contentBucket.bucketName + '/embeddingarchive/',
        '--prompt_path': 's3://' + contentBucket.bucketName + '/promptarchive/',
        '--out_table': distanceTable.tableName,
        '--distance_path': 's3://' + contentBucket.bucketName + '/promptdistance/',
        '--job_type': 'DISTANCE'
      }
    });
    contentBucket.grantReadWrite(distanceJob);
    distanceTable.grantFullAccess(distanceJob);

    //SageMaker Notebook Instance
    const nb_role = new iam.Role(this, 'NotebookInstanceRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
    });
    nb_role.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
        actions: [
          'firehose:PutRecordBatch',
        ],
        resources: [fh_prompts.attrArn]
      })
    );
    nb_role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:InvokeEndpoint"],
        resources: [
          "arn:aws:sagemaker:" + this.region + ":" + this.account + ":endpoint/" + endpointText,
          "arn:aws:sagemaker:" + this.region + ":" + this.account + ":endpoint/" + endpointEmbed
        ]
      })
    );
    nb_role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'cloudwatch:PutMetricData',
        ],
        resources: ['*']
      })
    );
    nb_role.addToPolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'es:ESHttp*'
          ],
        resources: [openSearchDomain.domainArn + "/*"]
      })
    );
    
    const notebook = new sagemaker.CfnNotebookInstance(this, 'NotebookInstance', {
      instanceType: 'ml.t3.medium',
      roleArn: nb_role.roleArn,
      defaultCodeRepository: "https://github.com/aws-samples/amazon-sagemaker-generativeai.git",
      subnetId: vpc.privateSubnets[0].subnetId,
      securityGroupIds: [jumpHostSG.securityGroupId],
    });

    // Outputs
    new cdk.CfnOutput(this, 'DriftTableReference', {
      value: `${driftTable.tableName}`,
    });
    new cdk.CfnOutput(this, 'DriftTablePromptsName', {
      value: `${driftTablePrompts.tableName}`,
    });
    new cdk.CfnOutput(this, 'CentroidTableReference', {
      value: `${driftTableCentroids.tableName}`,
    });
    new cdk.CfnOutput(this, 'CentroidTablePrompts', {
      value: `${driftTableCentroidsPrompts.tableName}`,
    });
    new cdk.CfnOutput(this, 'BucketName', {
      value: `${contentBucket.bucketName}`,
    });
    new cdk.CfnOutput(this, 'JumpHostId', {
      value: `${jumpHost.instanceId}`,
    });
    new cdk.CfnOutput(this, 'OpensearchDomain', {
      value: `${openSearchDomain.domainName}`,
    });
    new cdk.CfnOutput(this, 'OpensearchEndpoint', {
      value: `${openSearchDomain.domainEndpoint}`,
    });
    new cdk.CfnOutput(this, 'FirehosePromptsName', {
      value: `${fh_prompts.ref}`,
    });
    new cdk.CfnOutput(this, 'NotebookInstanceName', {
      value: `${notebook.attrNotebookInstanceName}`,
    });
    new cdk.CfnOutput(this, 'DataSyncTaskID', {
      value: `${datasyncTask.ref}`,
    });

    this.backendStackProps = {
      vpc: vpc,
      openSearchDomain: openSearchDomain.domainEndpoint,
      openSearchDomainArn: openSearchDomain.domainArn,
      firehoseName: fh_prompts.ref,
      firehoseArn: fh_prompts.attrArn,
      bucket: contentBucket
    };
  }
}
