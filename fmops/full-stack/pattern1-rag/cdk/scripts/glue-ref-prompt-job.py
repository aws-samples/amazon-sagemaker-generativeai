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
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.functions import mean as _mean, stddev as _stddev

if __name__ == "__main__":
    # Job setup
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'data_path', 'prompt_path', 'out_table', 'job_type', 'distance_path'])
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    # Read data
    path = args['data_path']
    promptPath = args['prompt_path']
    df = spark.read.option("recursiveFileLookup", "true").text(path)
    dfPrompt = spark.read.option("recursiveFileLookup", "true").text(promptPath)
    # Remove brackets
    df = df.withColumn("value", regexp_replace("value", r'(\[)', '')).withColumn("value", regexp_replace("value", r'(])', ''))
    dfPrompt = dfPrompt.withColumn("value", regexp_replace("value", r'(\[)', '')).withColumn("value", regexp_replace("value", r'(])', ''))
    # Expand to array
    df = df.select(split(col("value"),",").alias("EmbedArray")).drop("value")
    dfPrompt = dfPrompt.select(split(col("value"),",").alias("EmbedArray")).drop("value")
    # Cast to float
    df = df.withColumn("EmbedArray", transform(col("EmbedArray"), lambda x: x.cast("float")))
    df = df.withColumn("EmbedArray", col("EmbedArray").cast("array<float>"))
    dfPrompt = dfPrompt.withColumn("EmbedArray", transform(col("EmbedArray"), lambda x: x.cast("float")))
    dfPrompt = dfPrompt.withColumn("EmbedArray", col("EmbedArray").cast("array<float>"))
    # Convert to Vector
    df = df.select(array_to_vector('EmbedArray').alias('EmbedArray'))
    dfPrompt = dfPrompt.select(array_to_vector('EmbedArray').alias('EmbedArray'))

    # Run PCA
    assembler = VectorAssembler( inputCols=["EmbedArray"], outputCol="features")
    dfTrain = assembler.transform(df).drop('EmbedArray')
    dfTrainPrompt = assembler.transform(dfPrompt).drop('EmbedArray')
    pca = PCA(k=100, inputCol="features")
    pca.setOutputCol("pca_features")
    pca_model = pca.fit(dfTrain)
    pca_model.setOutputCol("output")
    dfPca = pca_model.transform(dfTrain)
    dfPcaPrompt = pca_model.transform(dfTrainPrompt)

    # Run k-means
    kmeans = KMeans(k=10)
    kmeans_model = kmeans.fit(dfPca)
    kmeans_model.setPredictionCol("label")
    dfKmeansPrompt = kmeans_model.transform(dfPcaPrompt).select("features", "label")

    # Build a data frame with cluster centers for each cluster
    l_clusters = kmeans_model.clusterCenters()
    d_clusters = {int(i):[float(l_clusters[i][j]) for j in range(len(l_clusters[i]))] for i in range(len(l_clusters))}
    df_centers = spark.sparkContext.parallelize([(k,)+(v,) for k,v in d_clusters.items()]).toDF(['prediction','center'])

    # Add centers to the predictions for the prompts
    dfKmeansPrompt = dfKmeansPrompt.withColumn('prediction',F.col('label').cast(IntegerType()))
    dfKmeansPrompt = dfKmeansPrompt.join(df_centers,on='prediction',how='left')

    # calculate distance between prompt and cluster
    get_dist = F.udf(lambda features, center : float(features.squared_distance(center)),FloatType())
    dfKmeansPrompt = dfKmeansPrompt.withColumn('dist',get_dist(F.col('features'),F.col('center')))

    # calculate mean, median, standard deviation of distance
    df_stats = dfKmeansPrompt.select(
        _mean(col('dist')).alias('mean'),
        _stddev(col('dist')).alias('std')
    ).collect()
    mean = df_stats[0]['mean']
    std = df_stats[0]['std']
    median = dfKmeansPrompt.approxQuantile("dist", [0.5], 0.25)[0]

    # Record results
    ddbclient = boto3.client('dynamodb')
    tbl = args['out_table']
    jobtype = args['job_type']
    jobdate = str(int(time.time()))
    rowitem = {
        'jobtype': {
            'S': jobtype
        },
        'jobdate': {
            'N': jobdate
        },
        'mean': {
            'N': str(mean)
        },
        'median': {
            'N': str(median)
        },
        'stdev': {
            'N': str(std)
        }
    }
    print(f"About to put item {json.dumps(rowitem)} to table {tbl}")
    ddbclient.put_item(
        Item=rowitem,
        ReturnConsumedCapacity='TOTAL',
        TableName=tbl
    )
    distance_path = args['distance_path']
    dfKmeansPrompt.write.mode('overwrite').parquet(distance_path)

    # Finish
    job.commit()
