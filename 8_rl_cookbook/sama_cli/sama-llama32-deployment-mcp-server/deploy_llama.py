#!/usr/bin/env python3
"""
Deploy meta-textgeneration-llama-3-2-3b on ml.g5.xlarge
"""

import logging
from datetime import datetime
from sagemaker.jumpstart.model import JumpStartModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_llama_model():
    """Deploy the LLaMA 3.2 3B model."""
    try:
        logger.info("Starting deployment of meta-textgeneration-llama-3-2-3b")
        
        # Model configuration
        model_id = "meta-textgeneration-llama-3-2-3b"
        model_version = "1.*"
        instance_type = "ml.g5.xlarge"
        initial_instance_count = 1
        accept_eula = True
        
        # Generate endpoint name
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"jumpstart-llama-3-2-3b-{timestamp}"
        
        logger.info(f"Creating JumpStart model: {model_id}")
        model = JumpStartModel(
            model_id=model_id,
            model_version=model_version
        )
        
        logger.info(f"Deploying to endpoint: {endpoint_name}")
        logger.info(f"Instance type: {instance_type}")
        logger.info(f"Instance count: {initial_instance_count}")
        
        # Deploy the model
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            accept_eula=accept_eula
        )
        
        logger.info("✅ Deployment successful!")
        print(f"\n🎉 Model deployed successfully!")
        print(f"📍 Endpoint name: {endpoint_name}")
        print(f"🖥️  Instance type: {instance_type}")
        print(f"📊 Instance count: {initial_instance_count}")
        print(f"🔗 Predictor endpoint: {predictor.endpoint_name}")
        print(f"📝 Content type: {predictor.content_type}")
        print(f"✅ Accept type: {predictor.accept}")
        
        # Create a sample inference payload
        sample_payload = {
            "inputs": "The future of artificial intelligence is",
            "parameters": {
                "max_new_tokens": 64,
                "top_p": 0.9,
                "temperature": 0.6,
                "return_full_text": False
            }
        }
        
        print(f"\n📋 Sample inference payload:")
        print(f"   {sample_payload}")
        print(f"\n🚀 To test the endpoint, use:")
        print(f"   response = predictor.predict(payload, custom_attributes='accept_eula=true')")
        
        return {
            "status": "success",
            "endpoint_name": endpoint_name,
            "predictor": predictor,
            "sample_payload": sample_payload
        }
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        print(f"\n❌ Deployment failed: {str(e)}")
        
        # Provide troubleshooting tips
        print(f"\n🔧 Troubleshooting tips:")
        print(f"   • Check if ml.g5.xlarge is available in your region")
        print(f"   • Verify you have sufficient service limits")
        print(f"   • Ensure you have proper SageMaker permissions")
        print(f"   • Check your AWS credentials are configured")
        
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    result = deploy_llama_model()
