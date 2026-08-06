"""Custom exception hierarchy for finetune package"""


class FinetuneException(Exception):
    """Base exception for all finetune errors"""


class AWSException(FinetuneException):
    """AWS-related errors"""


class SageMakerException(AWSException):
    """SageMaker-specific errors"""


class S3Exception(AWSException):
    """S3-specific errors"""


class ConfigException(FinetuneException):
    """Configuration-related errors"""


class TrainingException(FinetuneException):
    """Training-related errors"""


class DeploymentException(FinetuneException):
    """Deployment-related errors"""
