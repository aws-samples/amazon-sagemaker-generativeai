#!/usr/bin/env python3
"""
Test script for the updated deployment function with polling
"""

import asyncio
import json
import sys
import os

# Add the server module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sama_deployment_mcp_server'))

from server import deploy_jumpstart_model

async def test_deployment():
    """Test the deployment function with polling"""
    print("Testing deployment with polling...")
    
    try:
        result = await deploy_jumpstart_model(
            model_id="meta-textgeneration-llama-3-2-3b",
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
            accept_eula=True
        )
        
        print("Deployment result:")
        print(result)
        
        # Parse the result to check success
        result_dict = json.loads(result)
        if result_dict.get("success"):
            print(f"\n✅ SUCCESS: Endpoint {result_dict['endpoint_name']} is ready!")
            print(f"Deployment took: {result_dict.get('deployment_time_minutes', 'unknown')} minutes")
        else:
            print(f"\n❌ FAILED: {result_dict.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    print("🚀 Starting deployment test with polling...")
    print("This will take 10-20 minutes to complete...")
    asyncio.run(test_deployment())
