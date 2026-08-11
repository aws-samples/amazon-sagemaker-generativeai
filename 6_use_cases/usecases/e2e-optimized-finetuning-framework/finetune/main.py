import argparse
import os
import logging
import warnings

# Suppress warnings before any imports
os.environ['SAGEMAKER_INTERNAL_DISABLE_LOGGING'] = '1'
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ['sagemaker', 'sagemaker.config', 'sagemaker.jumpstart']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

from finetune.constants import TOKENIZERS_PARALLELISM
from finetune.launch_deployment import deployment_main
from finetune.launch_training import training_main
from finetune.utils.config_util import check_configs_and_files
from finetune.utils.logging_util import get_logger, setup_logging

# Setup logging
setup_logging("finetune", level=logging.INFO)
logger = get_logger(__name__)


def args_parse() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine Tune and Deploy Model")

    parser.add_argument("--train", action="store_true", help="Set this flag to enable training")
    parser.add_argument("--deploy", action="store_true", help="Set this flag to enable deployment")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS profile name (uses default profile if not specified).",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        help="SageMaker execution role ARN (required when running outside SageMaker).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="The name of the trained model (required for deployment without training).",
    )
    parser.add_argument(
        "--endpoint_name", type=str, default=None, help="The name of the endpoint being created."
    )
    parser.add_argument(
        "--s3_folder_existance",
        action="store_true",
        help="Set this flag to indicate the model has already been in S3.",
    )
    parser.add_argument(
        "--train_dataset_quality_check",
        action="store_true",
        help="Set this flag to indicate to check the quality of the train dateset.",
    )

    return parser.parse_args()


def set_environment_vars() -> None:
    """
    Set necessary environment variables for tokenizers and others.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
    logger.info("Environment variables for tokenizers parallelism set.")


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if not args.train and not args.deploy:
        raise ValueError("Either --train or --deploy flag must be specified")

    if args.deploy and not args.train and not args.model_name:
        raise ValueError("--model_name is required when deploying without training")


def main() -> None:
    """Main function to execute training and deployment."""
    args = args_parse()

    try:
        validate_arguments(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        raise

    check_configs_and_files()
    set_environment_vars()

    if args.train:
        training_job_name = training_main(args.train_dataset_quality_check, args.profile, args.role)

        if args.deploy:
            endpoint_name = f"{training_job_name}-endpoint"
            deployment_main(
                training_job_name, endpoint_name, s3_folder_existance=False, profile=args.profile, role=args.role
            )
    elif args.deploy:
        deployment_main(
            args.model_name, args.endpoint_name, args.s3_folder_existance, profile=args.profile, role=args.role
        )


if __name__ == "__main__":
    main()
