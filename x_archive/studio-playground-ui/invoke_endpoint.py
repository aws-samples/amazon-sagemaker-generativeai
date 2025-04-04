import boto3
import json

sagemaker_runtime = boto3.client("runtime.sagemaker")


def generate_text(payload, endpoint_name):
    encoded_input = json.dumps(payload).encode("utf-8")

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read())
    # - this works for faster transformr and DJL containers
    for item in result:
        # print(f" Item={item}, type={type(item)}")
        if isinstance(item, list):
            for element in item:
                if isinstance(element, dict):
                    # print(f"List:element::is: {element['generated_text']} ")
                    return element["generated_text"]
        elif isinstance(item, str):
            # print(item["generated_text"])
            # return item["generated_text"]
            print(f"probably:Item:from:dict::result[item]={result[item]}")
            return result[item]
        else:
            return result[0]["generated_text"]


def generate_text_ai21(payload, endpoint_name):
    print("payload type: ", type(payload))
    print("payload: ", payload)
    encoded_input = json.dumps({
        "prompt": payload["inputs"],
        "maxTokens": payload["maxTokens"],
        "temperature": payload["temperature"],
        "numResults": payload["numResults"]}).encode("utf-8")
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read())
    print(result['completions'][0]['data']['text'])
    return result['completions'][0]['data']['text']


def generate_text_ai21_summarize(payload, endpoint_name):
    encoded_input = json.dumps({
        "source": payload["inputs"],
        "sourceType": "TEXT"}).encode("utf-8")
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read())

    return result['summary']


def generate_text_ai21_context_qa(payload, question, endpoint_name):
    print('----- Context -------', payload["inputs"])
    print('----- Question ------', question)
    encoded_input = json.dumps({
        "context": payload["inputs"],
        "question": question}).encode("utf-8")
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read())
    return result['answer']
