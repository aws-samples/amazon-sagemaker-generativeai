###
 # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 # SPDX-License-Identifier: MIT-0
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy of this
 # software and associated documentation files (the "Software"), to deal in the Software
 # without restriction, including without limitation the rights to use, copy, modify,
 # merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 # permit persons to whom the Software is furnished to do so.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 # INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 # PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 # HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 # OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 # SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 # Copyright Amazon.com, Inc. and its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: MIT
######
import boto3
import json
import pandas as pd

_s3Client = None

def getS3Client():
    global _s3Client
    if( _s3Client is None):
        _s3Client = boto3.client('s3')
    return _s3Client

def readTextFileFromS3( bucketName, prefix ):
    s3Client = getS3Client()
    print( "reading from bucket ", bucketName, " key ", prefix)
    data = s3Client.get_object(Bucket=bucketName, Key=prefix)
    contents = data['Body'].read()
    return contents.decode("utf-8")

def split_s3_path(s3_path):
    path_parts = s3_path.replace("s3://","").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key

def readJSONFileFromS3( bucketName, prefix ):
    s3Client = getS3Client()
    print( "reading from bucket ", bucketName, " key ", prefix)
    data = s3Client.get_object(Bucket=bucketName, Key=prefix)
    contents = data['Body'].read()
    return contents.decode("utf-8")

def getExtractedDataFromS3( bucketName, prefix ):
    contents = readTextFileFromS3(bucketName,prefix)
    json_content = json.loads(contents)
    data = pd.DataFrame(json_content["Blocks"])
    dataFiltered = pd.DataFrame(data[data.BlockType.eq('LINE')])
    #if 'Confidence' in dataFiltered.columns:
    #    dataFiltered = dataFiltered[dataFiltered['Confidence']> 80]
    return dataFiltered
