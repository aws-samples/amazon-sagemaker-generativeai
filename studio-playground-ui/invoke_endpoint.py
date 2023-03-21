import streamlit as st
from streamlit_ace import st_ace
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
    
    # TO DO: results are either dictionary or list
    print(result) 
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



code = """
{
  "modelName": "Example-Model",
  "endpoint_name": "huggingface-inference-endpoint",
  "payload": {
    "parameters": {
      "length_penalty": 2.0,
      "max_new_tokens": 20,
      "early_stopping": false
    }
  }
}
"""


def get_user_input():
    uploaded_file = st.file_uploader(label="Upload JSON Template", type=["json"])
    uploaded_file_location = st.empty()

    # user uploads an image
    if uploaded_file is not None:
        input_str = json.load(uploaded_file)
        if validate_json_template(input_str):
            user_file_path = os.path.join("templates", 
                                          input_str["modelName"] + ".template.json")
            with open(user_file_path, "wb") as user_file:
                user_file.write(uploaded_file.getbuffer())
            uploaded_file_location.write("Template Uploaded: " + str(user_file_path))
        else:
            uploaded_file_location.warning("Invalid Input: please upload a valid template.json")
    else:
        user_file_path = None
        
    return user_file_path


@st.cache_data
def validate_json_template(json_dictionary):
    expected_keys = {"modelName", "endpoint_name", "payload"}
    actual_keys = set(json_dictionary.keys())

    if not expected_keys.issubset(actual_keys):
        st.warning(
            "Invalid Input: template.json must contain a modelName, endpoint_name, and payload keys"
        )
        return False

    if not "parameters" in json_dictionary["payload"].keys():
        st.warning(
            "Invalid Input: template.json must contain a payload key with parameters listed"
        )
        return False
    return True


@st.cache_data
def handle_editor_content(input_str):
    if validate_json_template(input_str):
            try:
                model_name = input_str["modelName"]
                filename = model_name + ".template.json"
                user_file_path = os.path.join("templates", filename)
                with open(user_file_path, "w+") as f:
                    json.dump(input_str, f)
                st.write("json saved at " + str(user_file_path))
            except Exception as e:
                st.write(e)


def main():
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
    st.image('./ml_image_prompt.png')


    # Adding your own model
    st.header("Add a New Model")
    st.write(
            """Add a new model by uploading a template.json file or by pasting the dictionary in
            the editor. A model template
            is a json dictionary containing a modelName, endpoint_name, and payload with
            parameters.  \n \n Below is an
            example of a template.json"""
        )
    get_user_input()

    # Spawn a new Ace editor and isplay editor's content as you type
    content = st_ace(
        theme="tomorrow_night",
        wrap=True,
        show_gutter=True,
        language="json",
        value= code,
        keybinding = "vscode",
        min_lines = 15
    )

    if content != code:
        input_str = json.loads(content)
        handle_editor_content(input_str)



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
    

    
if __name__ == "__main__":
    main()
