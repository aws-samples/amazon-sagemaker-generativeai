#!/usr/bin/env python3
"""
Standalone Network MCP Server for SAMA Deployment
Runs as a network service that multiple agents can connect to
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Optional

# Import the existing SAMA server components
from sama_deployment_mcp_server.server import (
    mcp, validate_aws_credentials, get_sagemaker_client
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone server."""
    parser = argparse.ArgumentParser(
        description='SAMA Deployment MCP Server - Standalone Network Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Port to listen on (default: 8001)'
    )
    
    parser.add_argument(
        '--allow-write',
        action='store_true',
        default=True,
        help='Allow write operations (model deployment, endpoint deletion)'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point for standalone network server."""
    args = parse_args()
    
    # Set environment variables
    os.environ['ALLOW_WRITE'] = str(args.allow_write).lower()
    
    # Set default AWS region if not set
    if not os.getenv('AWS_REGION'):
        os.environ['AWS_REGION'] = 'us-east-2'
    
    # Validate AWS credentials
    if not validate_aws_credentials():
        logger.error("Failed to validate AWS credentials. Please check your AWS configuration.")
        return 1
    
    logger.info("Starting SAMA Deployment MCP Server in Network Mode...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"AWS Region: {os.getenv('AWS_REGION')}")
    logger.info(f"Allow Write: {args.allow_write}")
    
    try:
        # Run the FastMCP server in network mode
        await mcp.run_server(host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
