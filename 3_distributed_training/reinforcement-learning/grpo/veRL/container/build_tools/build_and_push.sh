#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# set region
# region=
if [ "$#" -eq 3 ]; then
    region=$1
    image=$2
    tag=$3
else
    echo "usage: $0 <aws-region> $1 <image-name> $2 <image-tag>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${tag}"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${region} --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    aws ecr create-repository --region ${region} --repository-name "${image}" > /dev/null
fi


# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# Get the login command from ECR and execute it directly
# !!! region must be same as the region base image used in Dockerfile
# $(aws ecr get-login --no-include-email --region us-east-1  --registry-ids 763104351884)

docker build  -t ${image} $DIR/..
docker tag ${image} ${fullname}


docker build -t ${image}:${tag} -f container/Dockerfile .
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
docker tag ${image}:${tag} ${fullname}




# # Get the login command from ECR and execute it directly
# $(aws ecr get-login --region ${region} --no-include-email)
docker push ${fullname}
if [ $? -eq 0 ]; then
	echo "Amazon ECR URI: ${fullname}"
else
	echo "Error: Image build and push failed"
	exit 1
fi