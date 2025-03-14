import sys
import time
import boto3
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import split, col, regexp_replace, transform
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import array_to_vector
from pyspark.ml.feature import VectorAssembler, PCA
import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import json

if __name__ == "__main__":
    # Job setup
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'data_path', 'out_table', 'job_type', 'centroid_table'])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    # Read data
    path = args['data_path']
    df = spark.read.option("recursiveFileLookup", "true").text(path)
    # Remove brackets
    df = df.withColumn("value", regexp_replace("value", r'(\[)', '')).withColumn("value", regexp_replace("value", r'(])', ''))
    # Expand to array
    df = df.select(split(col("value"),",").alias("EmbedArray")).drop("value")
    # Cast to float
    df = df.withColumn("EmbedArray", transform(col("EmbedArray"), lambda x: x.cast("float")))
    df = df.withColumn("EmbedArray", col("EmbedArray").cast("array<float>"))
    # Convert to Vector
    df = df.select(array_to_vector('EmbedArray').alias('EmbedArray'))

    # Run PCA
    assembler = VectorAssembler( inputCols=["EmbedArray"], outputCol="features")
    dfTrain = assembler.transform(df).drop('EmbedArray')
    pca = PCA(k=100, inputCol="features")
    pca.setOutputCol("pca_features")
    pca_model = pca.fit(dfTrain)
    pca_model.setOutputCol("output")
    dfPca = pca_model.transform(dfTrain)

    # Calculate how many components give 95% explained variance
    expl_var = pca_model.explainedVariance.cumsum()
    expl_95 = np.argwhere(expl_var > 0.95)
    if expl_95.shape[0] > 0:
        expl_95 = expl_95[0][0]
    else:
        expl_95 = 100

    # Run k-means
    kmeans = KMeans(k=10)
    kmeans_model = kmeans.fit(dfPca)
    kmeans_model.setPredictionCol("label")
    dfKmeans = kmeans_model.transform(dfPca).select("features", "label")
    centers = kmeans_model.clusterCenters()
    evaluator = ClusteringEvaluator(predictionCol='label', featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')
    score=evaluator.evaluate(dfKmeans)

    # Record results
    ddbclient = boto3.client('dynamodb')
    tbl = args['out_table']
    c_tbl = args['centroid_table']
    jobtype = args['job_type']
    jobdate = str(int(time.time()))
    rowitem = {
        'jobtype': {
            'S': jobtype
        },
        'jobdate': {
            'N': jobdate
        },
        'score': {
            'N': str(score)
        },
        'inertia': {
            'N': str(kmeans_model.summary.trainingCost)
        },
        'varpc': {
            'N': str(expl_95)
        },
        'clustersizes': {
            'S': str(kmeans_model.summary.clusterSizes)
        }
    }
    print(f"About to put item {json.dumps(rowitem)} to table {tbl}")
    ddbclient.put_item(
        Item=rowitem,
        ReturnConsumedCapacity='TOTAL',
        TableName=tbl
    )
    for idx, c in enumerate(centers):
        c_rowitem = {
            'jobtypedate': {
                'S': f"{jobtype}-{jobdate}"
            },
            'centroid': {
                'N': str(idx)
            },
            'center': {
                'S': ','.join([str(x) for x in c])
            },
        }
        print(f"About to put item {json.dumps(c_rowitem)} to table {c_tbl}")
        ddbclient.put_item(
            Item=c_rowitem,
            ReturnConsumedCapacity='TOTAL',
            TableName=c_tbl
        )

    # Finish
    job.commit()
