from typing import List
import logging
import warnings
import os

os.environ['SAGEMAKER_INTERNAL_DISABLE_LOGGING'] = '1'
warnings.filterwarnings('ignore')

# Configure logging before importing sagemaker
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ['sagemaker', 'sagemaker.config', 'sagemaker.jumpstart']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer


def predict_batch(prompt: List[str], endpoint: str, profile_name: str = None):
    boto_session = boto3.Session(region_name="us-west-2", profile_name=profile_name) if profile_name else boto3.Session(region_name="us-west-2")
    sess = sagemaker.Session(boto_session=boto_session)

    predictor = Predictor(
        endpoint_name=endpoint,
        sagemaker_session=sess,
        serializer=JSONSerializer(),
    )

    parameters = {
        "temperature": 0,
        "max_seq_len": 5000,
    }
    input_key = "text"
    trim_start = 5
    trim_end = -3

    prediction = predictor.predict({input_key: prompt, "parameters": parameters}).decode("utf-8")

    return prediction[trim_start:trim_end]
