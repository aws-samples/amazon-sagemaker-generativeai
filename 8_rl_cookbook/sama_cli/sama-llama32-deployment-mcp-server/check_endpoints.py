#!/usr/bin/env python3
"""
Quick script to check current SageMaker endpoints
"""

import asyncio
import json
import sys
import os

# Add the server module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sama_deployment_mcp_server'))

from server import list_sagemaker_endpoints

async def check_endpoints():
    """Check current endpoints"""
    print("Checking current SageMaker endpoints...")
    
    try:
        result = await list_sagemaker_endpoints(max_results=20)
        result_dict = json.loads(result)
        
        if result_dict.get("success"):
            endpoints = result_dict.get("endpoints", [])
            print(f"\nFound {len(endpoints)} endpoints:")
            
            for endpoint in endpoints:
                print(f"  • {endpoint['endpoint_name']}")
                print(f"    Status: {endpoint['endpoint_status']}")
                print(f"    Created: {endpoint['creation_time']}")
                print()
        else:
            print(f"❌ Failed to list endpoints: {result_dict.get('error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_endpoints())
