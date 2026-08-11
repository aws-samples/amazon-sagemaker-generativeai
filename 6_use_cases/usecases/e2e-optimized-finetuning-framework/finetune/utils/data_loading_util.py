import json
import logging
import os
from io import StringIO
from typing import Dict

import pandas as pd
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split

# Set up logging for debugging and error handling
from finetune.constants import SEED_DATA_BUCKET, SEED_DATA_ROOT_FOLDER
from finetune.utils.logging_util import get_logger
from finetune.utils.training_data_quality_check import NumberAlignmentProcessor

logger = get_logger(__name__)


def convert_to_json(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a JSON string.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        str: The formatted JSON string.
    """
    result_dict = df.to_dict(orient="index")
    formatted_dict_list = [
        {"Input_Info": value["input_text"], "Target_Summary": value["output_text"]}
        for value in result_dict.values()
    ]
    json_data = json.dumps(formatted_dict_list, indent=2)
    return json_data


def split_and_save_data(df: pd.DataFrame, test_size: float) -> Dict[str, str]:
    """
    Split the DataFrame into training and test sets, and return JSON strings for both.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        Dict[str, str]: Dictionary containing the training and evaluation data JSON strings.
    """
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=df["input_type"])
    df_train = df_train.drop(columns=["input_type"], axis=1)
    df_test = df_test.drop(columns=["input_type"], axis=1)

    train_json = convert_to_json(df_train)
    eval_json = convert_to_json(df_test)

    return {"train_data": train_json, "eval_data": eval_json}


def _seed_training_data_if_needed(configs: Dict, aws_manager) -> None:
    """
    Check if the training data file exists in the user's bucket. If not, copy
    the sample data from the canonical source bucket so first-time users can
    run end-to-end without manual data upload.

    Skips seeding if the user has customized the data file name or input folder
    away from the defaults, since that implies they intend to supply their own data.
    """
    s3_bucket = configs["aws_setup"]["default_bucket"]
    s3_key = os.path.join(
        configs["data"]["s3_root_folder"],
        configs["data"]["s3_input_data_folder"],
        configs["data"]["s3_input_data_file"],
    )

    # Check if the file already exists
    try:
        aws_manager.s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
        logger.info(f"Training data already exists at s3://{s3_bucket}/{s3_key}")
        return
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code != "404":
            raise  # Unexpected error, don't swallow it

    # Build the source key from the canonical seed bucket
    source_key = os.path.join(
        SEED_DATA_ROOT_FOLDER,
        configs["data"]["s3_input_data_folder"],
        configs["data"]["s3_input_data_file"],
    )

    logger.info(
        f"Training data not found in s3://{s3_bucket}/{s3_key}. "
        f"Seeding from s3://{SEED_DATA_BUCKET}/{source_key}..."
    )

    try:
        copy_source = {"Bucket": SEED_DATA_BUCKET, "Key": source_key}
        aws_manager.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=s3_bucket,
            Key=s3_key,
        )
        logger.info(
            f"Successfully seeded training data from "
            f"s3://{SEED_DATA_BUCKET}/{source_key} to s3://{s3_bucket}/{s3_key}"
        )
    except ClientError as e:
        logger.error(
            f"Failed to seed training data from s3://{SEED_DATA_BUCKET}/{source_key}: {e}. "
            f"Please upload your training data manually to s3://{s3_bucket}/{s3_key}"
        )
        raise


def data_generate_main(
    configs: Dict[str, Dict[str, str]], aws_manager, train_dataset_quality_check: bool
):
    """
    Main function to load data from S3, split it, and upload directly to S3 as JSON.

    Args:
        configs (dict): Configuration dictionary containing input file paths and S3 details.
        aws_manager (AWSManager): Instance of the AWSManager class for S3 interaction.
        train_dataset_quality_check (bool): the flag to indicate whether training dataset quality check is required.
    """
    try:
        # Seed sample data from canonical bucket if this is a fresh setup
        _seed_training_data_if_needed(configs, aws_manager)

        # Read CSV file from S3
        s3_key = os.path.join(
            configs["data"]["s3_root_folder"],
            configs["data"]["s3_input_data_folder"],
            configs["data"]["s3_input_data_file"],
        )
        s3_bucket = configs["aws_setup"]["default_bucket"]
        csv_content = aws_manager.read_file_from_s3(s3_key)

        # Convert the CSV content to a DataFrame
        df = pd.read_csv(StringIO(csv_content), encoding="latin-1")
        df.columns = ["input_type", "input_text", "output_text"]
    except (IOError, ClientError) as e:
        logging.error(f"Error reading the CSV file from S3: {e}")
        raise

    if train_dataset_quality_check:
        logger.info("Start check quality of the training dataset.")
        number_alignment_processor = NumberAlignmentProcessor()
        input_figures = df["input_text"].apply(number_alignment_processor.extract_numbers_with_sign)
        output_figures = df["output_text"].apply(
            number_alignment_processor.extract_numbers_with_sign
        )

        diff = [
            number_alignment_processor.compare_lists(input_figure, output_figure)
            for input_figure in input_figures
            for output_figure in output_figures
        ]
        for row_num in range(len(df)):
            input_figure = input_figures.iloc[row_num]
            output_figure = output_figures.iloc[row_num]
            diff = number_alignment_processor.compare_lists(input_figure, output_figure)
            if not diff[0]:
                logger.warning(
                    f"Please not, at row {row_num}, input_figure are: {input_figure} while output_figure are: {output_figure}. The difference are: {diff}"
                )

    # Shuffle data and split
    df = df.sample(frac=1)
    result_data = split_and_save_data(df, configs["data"]["test_size"])

    # Upload JSON data directly to S3
    train_s3_key = os.path.join(
        configs["data"]["s3_root_folder"], configs["data"]["s3_input_data_folder"], "train.json"
    )
    eval_s3_key = os.path.join(
        configs["data"]["s3_root_folder"], configs["data"]["s3_input_data_folder"], "eval.json"
    )

    aws_manager.upload_data_to_s3(result_data["train_data"], s3_bucket, train_s3_key)
    aws_manager.upload_data_to_s3(result_data["eval_data"], s3_bucket, eval_s3_key)

    logging.info("Data generation and S3 upload completed to successfully.")
