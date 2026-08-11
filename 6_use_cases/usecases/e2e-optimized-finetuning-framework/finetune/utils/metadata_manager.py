"""Metadata management for training and deployment tracking"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


class MetadataManager:
    """Manage training and deployment metadata in S3"""

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: str = "metadata",
        profile: Optional[str] = None,
        auto_create: bool = True,
    ):
        """
        Initialize MetadataManager with S3 backend.

        Args:
            bucket: S3 bucket name. If None, uses config or creates dynamically
            prefix: S3 prefix for metadata files
            profile: AWS profile name (uses default if None)
            auto_create: Automatically create bucket if it doesn't exist

        Raises:
            ValueError: If profile is not provided and auto_create is False
        """
        if not profile and not auto_create:
            raise ValueError("profile must be provided when auto_create is False")

        session = boto3.Session(profile_name=profile)
        self.s3_client = session.client("s3")
        self.sts_client = session.client("sts")

        # Get bucket name
        if bucket:
            self.bucket = bucket
        else:
            self.bucket = self._get_or_create_bucket(auto_create)

        self.prefix = prefix
        self.metadata_key = f"{prefix}/metadata.json"
        self.data = self._load()

    def _get_or_create_bucket(self, auto_create: bool) -> str:
        """Get or create metadata bucket"""
        account_id = self.sts_client.get_caller_identity()["Account"]
        bucket_name = f"finetune-metadata-{account_id}"

        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return bucket_name
        except ClientError:
            if auto_create:
                self.s3_client.create_bucket(Bucket=bucket_name)
                # Enable encryption by default
                self.s3_client.put_bucket_encryption(
                    Bucket=bucket_name,
                    ServerSideEncryptionConfiguration={
                        "Rules": [
                            {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                        ]
                    },
                )
                # Block public access
                self.s3_client.put_public_access_block(
                    Bucket=bucket_name,
                    PublicAccessBlockConfiguration={
                        "BlockPublicAcls": True,
                        "IgnorePublicAcls": True,
                        "BlockPublicPolicy": True,
                        "RestrictPublicBuckets": True,
                    },
                )
                return bucket_name
            else:
                raise ValueError(f"Bucket {bucket_name} does not exist and auto_create is False")

    def _load(self) -> Dict[str, Any]:
        """Load metadata from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=self.metadata_key)
            return json.loads(response["Body"].read())
        except ClientError:
            return {"training": {}, "deployment": {}, "history": []}

    def _save(self) -> None:
        """Save metadata to S3 with security"""
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=self.metadata_key,
            Body=json.dumps(self.data, indent=2),
            ContentType="application/json",
            ServerSideEncryption="AES256",
            ACL="private",
        )

    def record_training(self, job_name: str, model_id: str, training_type: str) -> None:
        """Record training job completion"""
        self.data["training"] = {
            "last_job_name": job_name,
            "last_job_date": datetime.now().isoformat(),
            "model_id": model_id,
            "training_type": training_type,
        }
        self.data["history"].append(
            {
                "type": "training",
                "job_name": job_name,
                "date": datetime.now().isoformat(),
            }
        )
        self._save()

    def record_deployment(self, model_name: str, endpoint_name: str) -> None:
        """Record deployment completion"""
        self.data["deployment"] = {
            "last_model_name": model_name,
            "last_endpoint_name": endpoint_name,
            "last_endpoint_date": datetime.now().isoformat(),
        }
        self.data["history"].append(
            {
                "type": "deployment",
                "model_name": model_name,
                "endpoint_name": endpoint_name,
                "date": datetime.now().isoformat(),
            }
        )
        self._save()

    def get_last_training_job(self) -> Optional[str]:
        """Get last training job name"""
        return self.data.get("training", {}).get("last_job_name")

    def get_last_endpoint(self) -> Optional[str]:
        """Get last endpoint name"""
        return self.data.get("deployment", {}).get("last_endpoint_name")

    def get_deployment_command(self) -> Optional[str]:
        """Get deployment command for last training job"""
        training = self.data.get("training", {})

        if not training.get("last_job_name"):
            return None

        model_name = training.get("last_job_name")
        endpoint_name = f"{model_name}-endpoint"

        return f"python main.py --deploy --model_name {model_name} --endpoint_name {endpoint_name}"
