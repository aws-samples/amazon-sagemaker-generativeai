import traceback
import json
import boto3
from opensearchpy.helpers import bulk
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
import os
import uuid

"""
Example input:

{
  "Items": [
    {
      "paragraph": "93",
      "content": "Customers are responsible for making their own independent assessment of the information in this document. This document: (a) is for informational purposes only, (b) represents current AWS product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided \"as is\" without warranties, representations, or conditions of any kind, whether express or implied. The responsibilities and liabilities of AWS to its customers are controlled by AWS agreements, and this document is not part of, nor does it modify, any agreement between AWS and its customers."
    },
    {
      "paragraph": "94",
      "content": "© 2023 Amazon Web Services, Inc. or its affiliates. All rights reserved."
    },
    {
      "paragraph": "95",
      "content": "For the latest AWS terminology, see the AWS glossary in the AWS Glossary Reference."
    }
  ]
}

"""

def get_embedding(text, endpoint_name):

  sm_client = boto3.client('runtime.sagemaker')
    
  encoded_text = text.encode("utf-8")
  response = sm_client.invoke_endpoint(EndpointName=endpoint_name,
    ContentType='application/x-text',
    Accept='application/json',
    Body=encoded_text)
  
  response_body = json.loads(response.get('Body').read())
  embedding = response_body.get('embedding')[0]
  
  return embedding


def lambda_handler(event, context):

    print("Received event: " + json.dumps(event, indent=2))
    os_url = os.environ['DOMAINURL']
    index_name = os.environ['INDEX']
    fh_name = os.environ['FIREHOSE']
    embedding_model = os.environ['EMBEDDINGS_MODEL_ENDPOINT']

    fhclient = boto3.client('firehose')
  
    try:
        credentials = boto3.Session().get_credentials()
        region = boto3.Session().region_name
        auth = AWSV4SignerAuth(credentials, region, "es")
        opensearch = OpenSearch( 
          hosts = [{'host': os_url, 'port': 443}],
          http_auth = auth,
          use_ssl = True,
          verify_certs = True,
          connection_class = RequestsHttpConnection
        )
        requests = []
        fh_stream_records = []
        for item in event['Items']:
            print(f"Working on item {item}")
            content = item['content']
            embedding = get_embedding(content, embedding_model)
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": index_name,
                "embedding": embedding,
                "passage": content,
                "doc_id": _id,
            }
            requests.append(request)
            fh_stream_records.append({'Data': (str(embedding)+ "\n").encode('utf-8')})
        bulk(opensearch, requests)
        opensearch.indices.refresh(index=index_name)
        fhclient.put_record_batch( DeliveryStreamName=fh_name, Records=fh_stream_records)
        return { "Message": "Success "}
    except Exception as e:
        trc = traceback.format_exc()
        print(trc)
        return { "Message": "Failure"}
