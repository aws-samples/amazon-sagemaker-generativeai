import boto3, json, os
import httpx
import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SageMaker App")

@mcp.tool()
async def generate_prediction_with_sagemaker(test_sample: list):
    """
        Use Amazon SageMaker AI to generate predictions.
        Args:
            test_sample: a list of lists containing the inputs to generate predictions from
        Returns:
            predictions: an array of predictions
    """ 
    print(os.environ)
    endpoint_name = os.environ["SAGEMAKER_ENDPOINT_NAME"]
    boto_session = boto3.session.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        region_name=os.environ["AWS_REGION_NAME"]
    )
    sagemaker_runtime = boto_session.client("sagemaker-runtime")
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(test_sample),
        ContentType="application/json",
        Accept="application/json"
    )
    predictions = json.loads(response['Body'].read().decode("utf-8"))
    return np.array(predictions)

if __name__ == "__main__":
    mcp.run(transport="stdio")