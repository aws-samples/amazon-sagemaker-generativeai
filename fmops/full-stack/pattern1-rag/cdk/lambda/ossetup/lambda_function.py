import json
import os
import requests
import boto3
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

def on_event(event, context):

    print("Received event: " + json.dumps(event, indent=2))
    request_type = event['RequestType']
    if request_type == 'Create': 
        print("Creating domain")
        os_url = os.environ['DOMAINURL']
        index_name = os.environ['INDEX']

        mapping = {
            'settings': {
                'index': {
                    'knn': True  # Enable k-NN search for this index
                }
            },
            'mappings': {
                'properties': {
                    'embedding': {  # k-NN vector field
                        'type': 'knn_vector',
                        'dimension': 4096 # Dimension of the vector
                    },
                    'passage': {
                        'type': 'text'
                    },
                    'doc_id': {
                        'type': 'keyword'
                    }
                }
            }
        }
        print(f"Checking domain {os_url}/{index_name}")

        # region_name = boto3.Session().region_name
        # auth = BotoAWSRequestsAuth(aws_host=os_url, aws_region=regio_name, aws_service='es')
        # response = requests.head(f"{os_url}/{index_name}", auth=auth)
        # # If the index does not exist (status code 404), create the index
        # if response.status_code == 404:
        #     response = requests.put(f"{os_url}/{index_name}", json=mapping, auth=auth)
        #     print(f'Index created: {response.text}')
        # else:
        #     print('Index already exists!')
        # return { 'PhysicalResourceId': index_name}
        
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

        if not opensearch.indices.exists(index_name):
            opensearch.indices.create(index_name, body=mapping)
            print(f'Index created: {opensearch.indices.get(index_name)}')
        else:
            print('Index already exists!')
        return { 'PhysicalResourceId': index_name}
