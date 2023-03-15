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
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
            fileList.append(path)
    print('File count:', count)
    print('File list: ', fileList)
    return count, fileList

def read_template(template_path, curr_length, temp):
    TEMPLATE_FILE = template_path
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(prompt=prompt, curr_length=curr_length,temp=temp)
    return outputText

def generate_text(payload, parameters, endpoint_name):
    encoded_input = json.dumps(payload).encode("utf-8")
    
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(
        {
            "inputs": payload,
            "parameters": parameters,
        }
    )
    )
    result = json.loads(response['Body'].read().decode()) # -
    return result

def handle_stable_diffusion(response):
    print(response)
    img_res = io.BytesIO(response['Body'].read())
    placeholder  = st.image(img_res)
    return prompt

count, fileList = read_template_dir('templates')

endpoint_name_radio = st.sidebar.selectbox(
    "Select the endpoint to run in SageMaker",
    tuple(fileList)
)


########## UI Code #############
st.sidebar.title("LLM Model Parameters")
st.image('./ml_image.jpg')
st.header("Few Shot Playground")
# Length control
length_choice = st.sidebar.select_slider("Length",
                                         options=['very short', 'short', 'medium', 'long', 'very long'],
                                         value='medium',
                                         help="Length of the model response")

# early_stopping
early_stopping = st.sidebar.selectbox("Early Stopping",['True', 'False' ] )

# do_sample
do_sample_st = st.sidebar.selectbox("Sample Probabilities",['True', 'False' ] )

# Temperature control
temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.5,
                         help="The creativity of the model")

# Repetition penalty control 'no_repeat_ngram_size',
rep_penalty = st.sidebar.slider("Repetition penalty", min_value=1, max_value=5, value=2,
                                help="Penalises the model for repition, positive integer greater than 1")

# Repetition penalty control 'no_repeat_ngram_size',
beams_no = st.sidebar.slider("Beams Search For Greedy search", min_value=0, max_value=5, value=1,
                                help="Beams for optimization of search, positive integer")

# Repetition penalty control 'no_repeat_ngram_size',
seed_no = st.sidebar.slider("SEED for consistency", min_value=1, max_value=5, value=1,
                                help="Postive integer for consitent response, fix randomization")

#  Max length 'max_length'
#max_length = st.sidebar.text_input("Max length", value="50", max_chars=2)
max_length = {'very short':10, 'short':20, 'medium':30, 'long':40, 'very long':50}



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
    curr_length = max_length.get(length_choice, 10)
    curr_length = curr_length * 5 # for scaling
    output_text = read_template(f'templates/{endpoint_name_radio}', curr_length,temp)
    output = json.loads(output_text)
    parameters = output['parameter']
    endpoint_name = output['endpoint_name']
    generated_text = generate_text(prompt, parameters,endpoint_name)
    #print(generated_text)
    st.write(generated_text)