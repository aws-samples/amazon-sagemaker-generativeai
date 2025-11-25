#!/usr/bin/env python3
"""
Test the polling logic by checking an existing endpoint
"""

import asyncio
import json
import sys
import os
import time

# Add the server module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sama_deployment_mcp_server'))

from server import get_sagemaker_client
import boto3
from botocore.exceptions import ClientError

async def test_polling_logic():
    """Test the polling logic on an existing endpoint"""
    print("Testing polling logic...")
    
    # Use the existing InService endpoint
    endpoint_name = "jumpstart-meta-textgeneration-llama-3-2-3b-20250701-234111"
    
    try:
        sagemaker_client = get_sagemaker_client()
        
        print(f"Testing polling for endpoint: {endpoint_name}")
        
        # Simulate the polling logic from the updated deploy function
        max_wait_time = 5 * 60  # 5 minutes for test
        poll_interval = 10  # Check every 10 seconds for test
        start_time = time.time()
        
        poll_count = 0
        while True:
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                poll_count += 1
                
                print(f"Poll #{poll_count}: Endpoint {endpoint_name} status: {status}")
                
                if status == 'InService':
                    print(f"✅ SUCCESS: Endpoint is InService after {poll_count} polls")
                    elapsed_time = time.time() - start_time
                    print(f"Total polling time: {elapsed_time:.1f} seconds")
                    break
                elif status == 'Failed':
                    failure_reason = response.get('FailureReason', 'Unknown failure reason')
                    print(f"❌ FAILED: Endpoint failed - {failure_reason}")
                    break
                elif status in ['Creating', 'Updating']:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_wait_time:
                        print(f"❌ TIMEOUT: Polling timeout after {max_wait_time/60:.1f} minutes")
                        break
                    
                    print(f"   Still {status.lower()}... waiting {poll_interval} seconds")
                    await asyncio.sleep(poll_interval)
                else:
                    print(f"⚠️  Unexpected status: {status}")
                    await asyncio.sleep(poll_interval)
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'ValidationException':
                    print(f"❌ Endpoint not found: {endpoint_name}")
                    break
                else:
                    print(f"❌ AWS Error: {e}")
                    break
            except Exception as e:
                print(f"❌ Error during polling: {e}")
                break
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    print("🔍 Testing polling logic...")
    asyncio.run(test_polling_logic())
