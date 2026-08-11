import threading
from typing import Any, Dict

import boto3
import sagemaker
from botocore.exceptions import BotoCoreError, ClientError
from sagemaker.session import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from finetune.exceptions import SageMakerException
from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class MultiRegionAWSManager:
    _instance = None

    def __new__(cls, configs: Dict[str, Any] = None, is_sagemaker: bool = False):
        """
        Initialize the MultiRegionAWSManager class.

        Args:
            configs (dict, optional): A dictionary containing AWS credential configurations.
            is_sagemaker (bool): Flag to indicate if running in a SageMaker notebook.
        """
        if not cls._instance:
            cls._instance = super(MultiRegionAWSManager, cls).__new__(cls)
            cls._instance.configs = configs
            cls._instance.is_sagemaker = is_sagemaker
            cls._instance.aws_managers = (
                {}
            )  # Dictionary to store AWSManager instances for each region

        return cls._instance

    def get_aws_manager(self, region):
        """
        Get the AWSManager instance for the specified region.

        Args:
            region (str): The AWS region.

        Returns:
            AWSManager instance for the specified region.
        """
        if region not in self.aws_managers:
            self.aws_managers[region] = AWSManager(self.configs, self.is_sagemaker, region)
        return self.aws_managers[region]


class AWSManager:
    _lock = threading.Lock()
    DEFAULT_REGION = "us-east-1"

    def __init__(
        self,
        configs: Dict[str, Any] = None,
        is_sagemaker: bool = False,
        region: str = None,
        profile_name: str = None,
        role_arn: str = None,
    ):
        """
        Create a singleton instance of AWSManager.

        Args:
            configs (dict, optional): A dictionary containing AWS credential configurations.
            is_sagemaker (bool): Flag to indicate if running in a SageMaker notebook.
            region (str, optional): AWS region. Uses config or default if not provided.
            profile_name (str, optional): AWS CLI profile name. Uses config or default if not provided.
            role_arn (str, optional): SageMaker execution role ARN. Auto-detected if not provided.

        Returns:
            AWSManager instance.
        """
        super(AWSManager, self).__init__()
        with self._lock:
            self.configs = configs
            self.sagemaker_session = None
            self.s3_client = None

            # Set region: parameter > config > default
            self.region = (
                region
                or (configs.get("aws_setup", {}).get("region_name") if configs else None)
                or self.DEFAULT_REGION
            )

            # Set profile: parameter > config > None
            self.profile_name = profile_name or (
                configs.get("aws_setup", {}).get("profile_name") if configs else None
            )

            self.bucket = configs["aws_setup"]["default_bucket"] if configs else None

            if is_sagemaker:
                # Priority: CLI argument > get_execution_role() > config > environment variable
                if role_arn:
                    self.role = role_arn
                    logger.info(f"Using role from CLI argument: {self.role}")
                else:
                    try:
                        from sagemaker import get_execution_role
                        self.role = get_execution_role()
                        logger.info(
                            f"Running in SageMaker environment. Retrieved execution role: {self.role}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Not running in SageMaker environment: {e}. Checking for role in config or environment."
                        )
                        # Try to get role from config
                        self.role = configs.get("aws_setup", {}).get("role") if configs else None
                        
                        # If not in config, try environment variable
                        if not self.role:
                            import os
                            self.role = os.environ.get("SAGEMAKER_ROLE")
                        
                        if self.role:
                            logger.info(f"Using role from config/environment: {self.role}")
                        else:
                            logger.error(
                                "No SageMaker execution role found. SageMaker training requires an IAM role. "
                                "Please provide role via: --role argument, 'role' in aws_setup.yaml, "
                                "or SAGEMAKER_ROLE environment variable."
                            )
                            raise ValueError(
                                "SageMaker execution role is required for training. "
                                "Provide via --role, config file, or SAGEMAKER_ROLE environment variable."
                            )
            else:
                self.role = None
                logger.info(
                    f"Running in non-SageMaker environment. Region: {self.region}, Profile: {self.profile_name or 'default'}"
                )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def create_sagemaker_session(self) -> Session:
        """
        Create a SageMaker session using the specified profile and region.
        Retries up to 3 times with exponential backoff on failure.

        Returns:
            sagemaker.Session: A SageMaker session object.

        Raises:
            SageMakerException: If session creation fails after retries
        """
        try:
            session = boto3.Session(region_name=self.region, profile_name=self.profile_name)

            self.sagemaker_session = sagemaker.Session(
                boto_session=session, default_bucket=self.bucket
            )
            logger.info(
                f"SageMaker session created successfully. Region: {self.region}, Profile: {self.profile_name or 'default'}"
            )
            return self.sagemaker_session
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to create SageMaker session: {e}")
            raise SageMakerException(f"Failed to create SageMaker session: {e}") from e

    def write_cmd_credentials(self):
        """
        Generate AWS credential environment variables for CLI commands.

        Returns:
            str: Environment variables string for AWS CLI, or empty string if using IAM roles.
        """
        # When running in SageMaker or with IAM roles, credentials are automatically handled
        # by the instance profile, so we don't need to pass explicit credentials
        if not hasattr(self, "credentials") or self.credentials is None:
            logger.info("Using IAM role credentials (no explicit credentials needed)")
            return ""

        # If explicit credentials are available, format them for AWS CLI
        try:
            credentials = {
                "aws_access_key_id": self.credentials["AccessKeyId"],
                "aws_secret_access_key": self.credentials["SecretAccessKey"],
                "aws_session_token": self.credentials["SessionToken"],
            }
            return " ".join([f"{key.upper()}={value}" for key, value in credentials.items()])
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to format credentials: {e}. Using default credentials.")
            return ""

    def connect_to_s3(self) -> boto3.client:
        """
        Connect to S3 using the specified profile and region.
        If the configured bucket does not exist, it is created automatically.

        Returns:
            boto3.client: S3 client object.
        """
        try:
            session = boto3.Session(region_name=self.region, profile_name=self.profile_name)
            self.s3_client = session.client("s3")

            logger.info(
                f"S3 client connected successfully. Region: {self.region}, Profile: {self.profile_name or 'default'}"
            )

            # Auto-create the bucket if it doesn't exist
            if self.bucket:
                self._ensure_bucket_exists()

            return self.s3_client
        except (BotoCoreError, ClientError) as e:
            err_msg = f"Failed to connect to S3: {e}"
            logger.error(err_msg)
            raise

    def _ensure_bucket_exists(self) -> None:
        """
        Check if the configured S3 bucket exists. If not, create it.
        For us-east-1 no LocationConstraint is needed; other regions require it.
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 bucket '{self.bucket}' already exists.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                logger.info(f"S3 bucket '{self.bucket}' not found. Creating it...")
                try:
                    if self.region == "us-east-1":
                        self.s3_client.create_bucket(Bucket=self.bucket)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.region
                            },
                        )
                    logger.info(f"S3 bucket '{self.bucket}' created successfully in {self.region}.")
                except ClientError as create_err:
                    logger.error(f"Failed to create S3 bucket '{self.bucket}': {create_err}")
                    raise
            else:
                # Some other error (e.g. 403 Forbidden) — don't swallow it
                logger.error(f"Error checking bucket '{self.bucket}': {e}")
                raise

    def upload_file_to_s3(self, file_name: str, s3_bucket: str, object_name: str = None) -> None:
        """
        Upload a file to an S3 bucket.

        Args:
            file_name (str): File to upload.
            object_name (str): S3 object name. If not specified, file_name is used.
        """
        if self.s3_client is None:
            logger.error("S3 client not connected. Call connect_to_s3() first.")
            raise ValueError("S3 client not connected. Call connect_to_s3() first.")

        if object_name is None:
            object_name = file_name

        try:
            # Bandit: B110 - Use of dangerous functions (check if file_name is sanitized)
            self.s3_client.upload_file(
                file_name, s3_bucket, object_name, ExtraArgs={"ACL": "bucket-owner-full-control"}
            )
            logger.info(f"File {file_name} uploaded to {s3_bucket}/{object_name}")
        except ClientError as e:
            err_msg = f"Failed to upload file to S3: {e}"
            logger.error(err_msg)
            raise

    def upload_data_to_s3(self, data: str, s3_bucket: str, s3_key: str):
        """
        Upload in-memory data to S3.

        Args:
            data (str): The data to be uploaded to S3.
            s3_bucket (str): The name of the S3 bucket where data will be uploaded.
            s3_key (str): The key (path) under which the data will be stored in S3.

        Raises:
            ValueError: If the data is empty or None.
            ClientError: If there is an error during the upload process.
        """
        if not data:
            logger.error("No data provided for upload.")
            raise ValueError("Data to upload cannot be empty or None.")

        try:
            self.s3_client.put_object(Body=data, Bucket=s3_bucket, Key=s3_key)
            logger.info(f"Successfully uploaded data to s3://{s3_bucket}/{s3_key}")
        except ClientError as e:
            err_msg = f"Failed to upload data to S3: {e}"
            logger.error(err_msg)
            raise

    def download_file_from_s3(self, object_name: str, file_name: str = None) -> None:
        """
        Download a file from an S3 bucket.

        Args:
            object_name (str): S3 object name.
            file_name (str): Local file path to save the downloaded file. If not specified, object_name is used.
        """
        if self.s3_client is None:
            logger.error("S3 client not connected. Call connect_to_s3() first.")
            raise ValueError("S3 client not connected. Call connect_to_s3() first.")

        if file_name is None:
            file_name = object_name

        try:
            # Bandit: B110 - Check if object_name is sanitized before use
            self.s3_client.download_file(self.bucket, object_name, file_name)
            logger.info(f"File {file_name} downloaded from {self.bucket}/{object_name}")
        except ClientError as e:
            err_msg = f"Failed to download file from S3: {e}"
            logger.error(err_msg)
            raise

    def read_file_from_s3(self, object_name: str) -> str:
        """
        Read a file's content directly from S3 without downloading it. First attempts to decode using 'utf-8',
        and if that fails with a UnicodeDecodeError, falls back to 'latin-1' with a log message.

        Args:
            object_name (str): S3 object name to read.

        Returns:
            str: Content of the file as a string.

        Raises:
            ClientError: If there is an issue reading the file from S3.
        """
        if self.s3_client is None:
            logger.error("S3 client not connected. Call connect_to_s3() first.")
            raise ValueError("S3 client not connected. Call connect_to_s3() first.")

        try:
            # Bandit: B110 - Check if object_name is sanitized before use
            response = self.s3_client.get_object(Bucket=self.bucket, Key=object_name)
            content = response["Body"].read()

            try:
                content_str = content.decode("utf-8")
                logger.info(f"File {object_name} successfully decoded using 'utf-8'.")
            except UnicodeDecodeError:
                content_str = content.decode("latin-1")
                logger.warning(
                    f"Failed to decode file {object_name} using 'utf-8', switched to 'latin-1' decoding."
                )

            return content_str

        except ClientError as e:
            err_msg = f"Failed to read file {object_name} from S3: {e}"
            logger.error(err_msg)
            raise
