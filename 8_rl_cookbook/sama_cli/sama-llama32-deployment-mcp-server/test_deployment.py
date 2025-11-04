#!/usr/bin/env python3
"""
Test script to directly call the deployment MCP server functions
"""

import asyncio
import sys
import os

# Add the server module to path
sys.path.insert(0, '/home/sagemaker-user/Agents/SAMA/SAMA-CLI/deployment-mcp-server')

from deployment_mcp_server.server import deploy_jumpstart_model

async def test_deployment():
    """Test the deployment function directly."""
    print("🧪 Testing deployment MCP server function directly...")
    
    try:
        # Test the deploy function
        result = await deploy_jumpstart_model(
            model_id="meta-textgeneration-llama-3-2-3b",
            instance_type="ml.g5.xlarge",
            model_version="1.*",
            initial_instance_count=1,
            endpoint_name=None,
            accept_eula=True
        )
        
        print("✅ Function call successful!")
        print("📋 Result:")
        print(result)
        
    except Exception as e:
        print(f"❌ Function call failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_deployment())
