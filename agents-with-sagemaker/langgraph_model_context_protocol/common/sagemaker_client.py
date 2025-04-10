import os
import json
import boto3
from typing import Dict
from dotenv import load_dotenv
from botocore.config import Config
from botocore.response import StreamingBody
from langchain_core.messages import AIMessageChunk
from langchain_aws.chat_models.sagemaker_endpoint import ChatSagemakerEndpoint, ChatModelContentHandler

# === Load .env ===
load_dotenv()

# === ENV CONFIG ===
REGION = os.getenv("AWS_REGION", "us-east-1")
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT")
assert ENDPOINT_NAME, "❌ SageMaker endpoint name is not set!"

session = boto3.Session(
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN")  # optional
)

sm = session.client("sagemaker-runtime", config=Config(region_name=REGION))

class ContentHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs: Dict) -> bytes:
        body = {
            "messages": prompt,
            "stream": True,
            **model_kwargs
        }
        return json.dumps(body).encode("utf-8")

    def transform_output(self, output: StreamingBody) -> AIMessageChunk:
        try:
            all_content = []
            for line in output.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    json_data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if json_data.get("choices", [{}])[0].get("delta", {}).get("content") == "[DONE]":
                    break
                content = json_data["choices"][0]["delta"].get("content", "")
                all_content.append(content)
            return AIMessageChunk(content="".join(all_content))
        except Exception as e:
            return AIMessageChunk(content=f"Error processing response: {str(e)}")

chat_content_handler = ContentHandler()

chat_llm = ChatSagemakerEndpoint(
    endpoint_name=ENDPOINT_NAME,
    client=sm,
    content_handler=chat_content_handler,
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 1200,
        "top_p": 0.95,
        "do_sample": True
    }
)

