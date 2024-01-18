import ecs = require('aws-cdk-lib/aws-ecs');
import { DockerImageAsset, Platform } from 'aws-cdk-lib/aws-ecr-assets';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import dynamodb = require('aws-cdk-lib/aws-dynamodb');
import iam = require('aws-cdk-lib/aws-iam');
import ecs_patterns = require('aws-cdk-lib/aws-ecs-patterns');
import cdk = require('aws-cdk-lib');
import {  HttpOrigin } from 'aws-cdk-lib/aws-cloudfront-origins';
import { Distribution, ViewerProtocolPolicy, OriginProtocolPolicy, AllowedMethods, CachePolicy,
  OriginRequestPolicy, OriginRequestCookieBehavior, OriginRequestHeaderBehavior, OriginRequestQueryStringBehavior } from 'aws-cdk-lib/aws-cloudfront';
import { UserPool, UserPoolClientIdentityProvider, OAuthScope  } from 'aws-cdk-lib/aws-cognito';
import { AuthenticateCognitoAction } from 'aws-cdk-lib/aws-elasticloadbalancingv2-actions';
import { ListenerAction, ApplicationProtocol, ListenerCondition } from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as acm from 'aws-cdk-lib/aws-certificatemanager';
import * as cw from 'aws-cdk-lib/aws-cloudwatch';
import * as route53 from 'aws-cdk-lib/aws-route53';
import { CloudFrontTarget, UserPoolDomainTarget } from 'aws-cdk-lib/aws-route53-targets';
import path = require('path');
import s3 = require('aws-cdk-lib/aws-s3');


interface FrontendStackProps extends cdk.StackProps {
    vpc: ec2.IVpc
    openSearchDomain: string;
    openSearchDomainArn: string;
    firehoseName: string;
    firehoseArn: string;
    bucket: s3.Bucket;
}

export class FrontendStack extends cdk.Stack {
    constructor(scope: cdk.App, id: string, props: FrontendStackProps) {
    super(scope, id, props);

    // Get context values
    const appCustomDomainName = this.node.tryGetContext('appCustomDomainName');
    const loadBalancerOriginCustomDomainName = this.node.tryGetContext('loadBalancerOriginCustomDomainName');
    const customDomainRoute53HostedZoneID = this.node.tryGetContext('customDomainRoute53HostedZoneID');
    const customDomainRoute53HostedZoneName = this.node.tryGetContext('customDomainRoute53HostedZoneName');
    const customDomainCertificateArn = this.node.tryGetContext('customDomainCertificateArn');
    const endpointText = this.node.tryGetContext('textModelEndpointName');
    const endpointEmbed = this.node.tryGetContext('embeddingsModelEndpointName');

    // Generate random string
    function generateRandomString(length: number) {
      let result = '';
      const characters = 'abcdefghijklmnopqrstuvwxyz0123456789';
      for (let i = 0; i < length; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
      }
      return result;
    }

    // Docker image build and upload to ECR
    const dockerImage = new DockerImageAsset(this, 'MyBuildImage', {
        directory: path.join(__dirname, '../../frontend'),
        platform: Platform.LINUX_AMD64
      });
    
    // Load existing hosted zone
    const hosted_zone = route53.HostedZone.fromHostedZoneAttributes(this, 'HostedZone', {
      hostedZoneId: customDomainRoute53HostedZoneID,
      zoneName: customDomainRoute53HostedZoneName
    });

    const vpc = props.vpc
    const cluster = new ecs.Cluster(this, 'Cluster', { 
      vpc,
      enableFargateCapacityProviders: true,
      containerInsights: true
    });
    const taskDefinition = new ecs.FargateTaskDefinition(this, 'TaskDef', {cpu: 512, memoryLimitMiB: 2048});

    // DynamoDB Table
    const conversationMemoryTable = new dynamodb.Table(this, 'ConversationMemoryTable', {
      partitionKey: {
        name: 'id',
        type: dynamodb.AttributeType.STRING
      }})
    
    // ECS Task Definition
    const appContainer = taskDefinition.addContainer('StreamlitContainer', {
      image: ecs.ContainerImage.fromDockerImageAsset(dockerImage),
      cpu: 512,
      memoryLimitMiB: 2048,
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'streamlit-log-group', logRetention: 30 }),
      environment: {
        'DDB_TABLE_NAME': conversationMemoryTable.tableName,
        'OPENSEARCH_ENDPOINT': props.openSearchDomain,
        'OPENSEARCH_INDEX': 'embeddings',
        'FIREHOSE': props.firehoseName,
        'TEXT_MODEL_ENDPOINT': endpointText,
        'EMBEDDINGS_MODEL_ENDPOINT': endpointEmbed
      }
    });
    
    // Add policy to allow task access to invove Jumpstart model
    taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: ['sagemaker:InvokeEndpoint'],
        resources: [
            "arn:aws:sagemaker:" + this.region + ":" + this.account + ":endpoint/" + endpointText,
            "arn:aws:sagemaker:" + this.region + ":" + this.account + ":endpoint/" + endpointEmbed
        ]
      }))
    
    // Add policy to allow task to access DynamoDB
    taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          "dynamodb:PutItem",
          "dynamodb:GetItem"
          ],
          resources: [conversationMemoryTable.tableArn]
        }))

    // Cloudwatch access
    taskDefinition.addToTaskRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'cloudwatch:PutMetricData',
        ],
        resources: ['*']
      })
    );

    // Firehose access
    taskDefinition.addToTaskRolePolicy(
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'firehose:PutRecordBatch',
        ],
        resources: [props.firehoseArn]
      })
    );
    
    // Add policy to allow task to access OpenSearch
    taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          'es:ESHttp*'
          ],
        resources: [props.openSearchDomainArn + "/*"]
      }))
    appContainer.addPortMappings({ containerPort: 8501, protocol: ecs.Protocol.TCP});
    
    const certificate = acm.Certificate.fromCertificateArn(this, 'ACMCertificate', `${customDomainCertificateArn}`);

    // ECS Fargate Service
    const service = new ecs_patterns.ApplicationLoadBalancedFargateService(this, 'FargateService', {
      cluster: cluster,
      taskDefinition: taskDefinition,
      protocol: ApplicationProtocol.HTTPS,
      certificate: certificate,
      domainName: loadBalancerOriginCustomDomainName,
      domainZone: hosted_zone
    });

    // Set up access logging
    service.loadBalancer.logAccessLogs(props.bucket, 'alblog')

    // Get load balancer security group and add a https rule
    const alb_sg = service.loadBalancer.connections.securityGroups[0];
    alb_sg.addEgressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'Allow HTTPS');

    // CloudFront distribution for ecs service
    const customHeaderValue = generateRandomString(10)
    const origin = new HttpOrigin(`${loadBalancerOriginCustomDomainName}`, {
      protocolPolicy: OriginProtocolPolicy.HTTPS_ONLY,
      customHeaders: {
        "X-Custom-Header": customHeaderValue
      }
    });

    // Origin request policy
    const originRequestPolicy = new OriginRequestPolicy(this, 'OriginRequestPolicy', {
      originRequestPolicyName: 'ALBPolicy',
      cookieBehavior: OriginRequestCookieBehavior.all(),
      headerBehavior: OriginRequestHeaderBehavior.all(),
      queryStringBehavior: OriginRequestQueryStringBehavior.all(),
    
    });
    
    // CloudFront distribution
    const distribution = new Distribution(this, 'Distribution', {
      certificate: certificate,
      domainNames: [appCustomDomainName],
      defaultBehavior: {
        origin: origin,
        viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        originRequestPolicy: originRequestPolicy,
        allowedMethods: AllowedMethods.ALLOW_ALL,
        cachePolicy: CachePolicy.CACHING_DISABLED,
      }
    });

    // Add cloudfront domain name to route53
    const cloudFrontDNS = new route53.ARecord(this, 'CloudFrontARecord', {
      zone: hosted_zone,
      target: route53.RecordTarget.fromAlias(new CloudFrontTarget(distribution)),
      recordName: appCustomDomainName
    });

    // Setup cognito for user authentication
    const userPool = new UserPool(this, 'UserPool', {
      selfSignUpEnabled: false,
      signInAliases: { email: true },
    });
    const userPoolClient = userPool.addClient('UserPoolClient', {
      userPoolClientName: "alb-auth-client",
      generateSecret: true,
      oAuth:{
        flows: {
          authorizationCodeGrant: true,
        },
        scopes: [OAuthScope.OPENID],
        callbackUrls: [`https://${distribution.distributionDomainName}/oauth2/idpresponse`,
        `https://${distribution.distributionDomainName}`,
        `https://${appCustomDomainName}/oauth2/idpresponse`,
        `https://${appCustomDomainName}`
        ],
        logoutUrls: [`https://${distribution.distributionDomainName}`,
          `https://${appCustomDomainName}`
        ]
      },
      supportedIdentityProviders: [
        UserPoolClientIdentityProvider.COGNITO
      ]
    });

    // generate cognito domain prefix from app custom domain. Not more than 15 characters
    const domain_prefix = appCustomDomainName.replace(/\./g, '-')
    let result;
    if (domain_prefix.length > 20) {
      result = domain_prefix.slice(0, 20);
    } else {
      result = domain_prefix; 
    }
    if (result.endsWith('-')) {
      result = result.slice(0, -1);
    }

    // cognito user pool domain
    const userPoolDomain = userPool.addDomain('UserPoolDomain', {
      cognitoDomain: {
        domainPrefix: result
      }
    });

    service.listener.addAction(
      'cognito-auth', {
        priority: 1,
        conditions: [ListenerCondition.httpHeader("X-Custom-Header", [customHeaderValue])],
        action: new AuthenticateCognitoAction({
          userPool,
          userPoolClient,
          userPoolDomain,
          next: ListenerAction.forward([service.targetGroup])
        })
      }
    );
    service.listener.addAction(
      'Default', {
        action: ListenerAction.fixedResponse(403, {
          contentType: 'text/plain',
          messageBody: 'Forbidden'
        })
      }
    );

    // Cloudwatch dashboard
    const dashboard = new cw.Dashboard(this, 'RagScoreDashboard', {
      defaultInterval: cdk.Duration.days(7),
      dashboardName: "RAG_Scores"
    });
    dashboard.addWidgets(new cw.GraphWidget({
      title: "Average similarity score",
      left: [new cw.Metric({
        metricName: "similarity",
        namespace: "rag",
      })],
    }));
    dashboard.addWidgets(new cw.SingleValueWidget({
      title: "Number of similarity scores recorded",
      metrics: [new cw.Metric({
        metricName: "similarity",
        namespace: "rag",
        statistic: "SampleCount",
      })],
    }));

    new cdk.CfnOutput(this, 'CloudFrontDomain', {
      value: `https://${distribution.distributionDomainName}`
    });
    new cdk.CfnOutput(this, 'AppURL', {
      value: `https://${appCustomDomainName}`
    });
    new cdk.CfnOutput(this, 'CognitoUserPool', {
      value: userPool.userPoolId
    });
  }
}