#!/bin/bash

# Function to detect if running in SageMaker Studio
is_sagemaker_studio() {
    # Check for SageMaker-specific environment variables
    if [[ -n "$SM_CURRENT_HOST" ]] || [[ -n "$SAGEMAKER_INTERNAL_IMAGE_URI" ]] || [[ -n "$SM_USER_ID" ]]; then
        return 0  # True - in SageMaker Studio
    fi
    
    # Check for SageMaker Studio specific paths
    if [[ -d "/opt/ml" ]] && [[ -f "/opt/ml/metadata/resource-metadata.json" ]]; then
        return 0  # True - in SageMaker Studio
    fi
    
    # Check if running in a container with SageMaker characteristics
    if [[ -f "/.dockerenv" ]] && [[ $(hostname) =~ ^sagemaker-* ]]; then
        return 0  # True - likely in SageMaker Studio
    fi
    
    # Check for SageMaker Studio user
    if [[ $(whoami) == "sagemaker-user" ]]; then
        return 0  # True - likely in SageMaker Studio
    fi
    
    return 1  # False - not in SageMaker Studio
}

# Check if required parameters are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <REPO_NAME> [TAG]"
    echo "Example: $0 my-xgboost-image"
    echo "Example: $0 my-xgboost-image v1.0.0"
    exit 1
fi

# Try to get region from STS ARN first, fallback to configured region
AWS_REGION=$(aws sts get-caller-identity --query 'Arn' --output text 2>/dev/null | cut -d':' -f4)
if [ -z "$AWS_REGION" ]; then
    AWS_REGION=$(aws configure get region)
fi

# If still empty, exit with error
if [ -z "$AWS_REGION" ]; then
    echo "Error: Could not determine AWS region. Please configure your AWS CLI or set AWS_DEFAULT_REGION environment variable."
    exit 1
fi

export AWS_REGION
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export REPO_NAME=$1
export TAG=${2:-latest}  # Use provided tag or default to 'latest'

echo "This process may take 10-15 minutes to complete..."

echo "Building image..."

# Detect environment and use appropriate Docker build command
if is_sagemaker_studio; then
    echo "Detected SageMaker Studio environment - using --network sagemaker"
    docker build --network sagemaker --platform linux/amd64 -t ${REGISTRY}${REPO_NAME}:${TAG} .
else
    echo "Detected local/standard environment - using default network"
    docker build --platform linux/amd64 -t ${REGISTRY}${REPO_NAME}:${TAG} .
fi

# Create registry if needed (using proper AWS CLI query)
echo "Checking if repository exists..."
REPO_EXISTS=$(aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${AWS_REGION} 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Creating repository ${REPO_NAME}..."
    aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION}
else
    echo "Repository ${REPO_NAME} already exists"
fi

# Login to registry
echo "Logging in to $REGISTRY ..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

echo "Pushing image to ${REGISTRY}${REPO_NAME}:${TAG} ..."

# Push image to registry
docker image push ${REGISTRY}${REPO_NAME}:${TAG}

echo "Image push completed successfully!"