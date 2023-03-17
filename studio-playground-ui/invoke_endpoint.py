import streamlit as st
import boto3
import json
import io
import jinja2
import os
from pathlib import Path

sagemaker_runtime = boto3.client('runtime.sagemaker')
templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)

def read_template_dir(dir_path):
    # folder path
    dir_path = dir_path
    count = 0
    fileList = []
    fileNames = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
            fileList.append(path)
            fileNames.append(path.split('.')[0])
    return count, fileList, fileNames

def read_template(template_path):
    TEMPLATE_FILE = template_path
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render()
    return outputText

def generate_text(payload, endpoint_name):
    encoded_input = json.dumps(payload).encode("utf-8")
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response['Body'].read().decode()) # -
    for item in result:
        if isinstance(item, list):
            for element in item:
                if isinstance(element, dict):
                    print(element["generated_text"])
                    return element["generated_text"]
        else:
            print(item["generated_text"])
            return item["generated_text"]

def handle_stable_diffusion(response):
    print(response)
    img_res = io.BytesIO(response['Body'].read())
    placeholder  = st.image(img_res)
    return prompt

count, fileList, fileNames = read_template_dir('templates')


endpoint_name_radio = st.sidebar.selectbox(
    "Select the endpoint to run in SageMaker",
    tuple(fileNames)
)

output_text = read_template(f'templates/{endpoint_name_radio}.template.json')
output = json.loads(output_text)

parameters = output['payload']['parameters']

########## UI Code #############
st.sidebar.title("Model Parameters")
st.image('./ml_image.jpg')
st.header("Prompt Engineering Playground")
for x in parameters: 
    if isinstance(parameters[x], bool):
        parameters[x] = st.sidebar.selectbox(x,['True', 'False' ] )
    if isinstance(parameters[x], int):
        if x == 'max_new_tokens' or x == 'max_length':
            parameters[x] = st.sidebar.slider(x, min_value=0, max_value=500, value=100)
        else: 
            parameters[x] = st.sidebar.slider(x, min_value=0, max_value=10, value=2)
    if isinstance(parameters[x], float):
        if x == 'temperature':
            parameters[x] =  st.sidebar.slider(x, min_value=0.0, max_value=1.5, value=0.5,
                         help="Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature → 0, it results in greedy decoding. If specified, it must be a positive float.")
        else: 
            parameters[x] = st.sidebar.slider(x, min_value=0.0, max_value=10.0, value=2.1)
            

st.markdown("""

Example :red[For Few Shot Learning]

**:blue[List the Country of origin of food.]**
Pizza comes from Italy
Burger comes from USA
Curry comes from
""")

prompt = st.text_area("Enter your prompt here:", height=350)
placeholder = st.empty()

if st.button("Run"):
    placeholder = st.empty()
    endpoint_name = output['endpoint_name']
    payload = {"inputs": [prompt,],  "parameters": parameters}
    print(' generated payload for inference : ', payload)
    generated_text = generate_text(payload,endpoint_name)
    print(generated_text)
    st.write(generated_text)