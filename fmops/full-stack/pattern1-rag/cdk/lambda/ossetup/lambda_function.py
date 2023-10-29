import json
import os
import requests

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
        response = requests.head(f"{os_url}/{index_name}")
        # If the index does not exist (status code 404), create the index
        if response.status_code == 404:
            response = requests.put(f"{os_url}/{index_name}", json=mapping)
            print(f'Index created: {response.text}')
        else:
            print('Index already exists!')
        return { 'PhysicalResourceId': index_name}
