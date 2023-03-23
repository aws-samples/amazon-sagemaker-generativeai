import streamlit as st
from streamlit_ace import st_ace
import boto3
import json
import io
import jinja2
import os
from pathlib import Path
import random
import string


N = 7
sagemaker_runtime = boto3.client("runtime.sagemaker")
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)


code_example = """{
  "model_name": "model",
  "endpoint_name": "huggingface-inference-endpoint",
  "payload": {
    "parameters": {
      "length_penalty": {
        "default": 2,
        "min": 0,
        "max": 100
      },
      "max_new_tokens": {
        "default": 20,
        "min": 0,
        "max": 100
      },
      "early_stopping": {
        "default": false,
        "min": true,
        "max": false
      }
    }
  }
}
"""


def list_templates(dir_path):
    # folder path
    templates = []
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            templates.append(path.split(".")[0])
    return templates


def read_template(template_path):
    template = template_env.get_template(template_path)
    output_text = template.render()
    return output_text


def generate_text(payload, endpoint_name):
    encoded_input = json.dumps(payload).encode("utf-8")

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read().decode())  # -

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
    img_res = io.BytesIO(response["Body"].read())
    placeholder = st.image(img_res)
    return prompt


def get_user_input():
    uploaded_file = st.file_uploader(label="Upload JSON Template", type=["json"])
    uploaded_file_location = st.empty()

    # user uploads an image
    if uploaded_file is not None:
        input_str = json.load(uploaded_file)
        if validate_json_template(input_str):
            user_file_path = os.path.join(
                "templates", input_str["model_name"] + ".template.json"
            )
            with open(user_file_path, "wb") as user_file:
                user_file.write(uploaded_file.getbuffer())
            uploaded_file_location.write("Template Uploaded: " + str(user_file_path))
            st.session_state["new_template_added"] = True
        else:
            uploaded_file_location.warning(
                "Invalid Input: please upload a valid template.json"
            )
    else:
        user_file_path = None

    return user_file_path


@st.cache_data
def validate_json_template(json_dictionary):
    expected_keys = {"model_name", "endpoint_name", "payload"}
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
            model_name = input_str["model_name"]
            filename = model_name + ".template.json"
            user_file_path = os.path.join("templates", filename)
            with open(user_file_path, "w+") as f:
                json.dump(input_str, f)

            st.write("json saved at " + str(user_file_path))
            st.session_state["new_template_added"] = True

        except Exception as e:
            st.write(e)


def handle_parameters(parameters):
    for p in parameters:
        minimum = parameters[p]["min"]
        maximum = parameters[p]["max"]
        default = parameters[p]["default"]
        if type(minimum) == int and type(maximum) == int and type(default) == int:
            parameters[p] = st.sidebar.slider(
                p, min_value=minimum, max_value=maximum, value=default, step= 1)
        if type(minimum) == bool and type(maximum) == bool and type(default) == bool:
            parameters[p] = st.sidebar.selectbox(p, ["True", "False"])
        if type(minimum) == float and type(maximum) == float and type(default) == float:
            parameters[p] = st.sidebar.slider(
                    p, min_value=float(minimum), 
                    max_value=float(maximum), 
                    value=float(default), step = 0.01
                )
    return parameters


def main():
    st.session_state["new_template_added"] = False
    sidebar_selectbox = st.sidebar.empty()
    selected_endpoint = sidebar_selectbox.selectbox(
        label="Select the endpoint to run in SageMaker",
        options=list_templates("templates"),
    )

    st.sidebar.title("Model Parameters")
    st.image("./ml_image_prompt.png")

    # Adding your own model
    with st.expander("Add a New Model"):
        st.header("Add a New Model")
        st.write(
                """Add a new model by uploading a template.json file or by pasting the dictionary
                in the editor. A model template is a json dictionary containing a modelName,
                endpoint_name, and payload with parameters. [TO DO: instructions for getting parameters] \n \n Below is an example of a
                template.json"""
            )
        res = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
        get_user_input()

        # Spawn a new Ace editor and display editor's content as you type
        content = st_ace(
            theme="tomorrow_night",
            wrap=True,
            show_gutter=True,
            language="json",
            value=code_example,
            keybinding="vscode",
            min_lines=15,
        )

        if content != code_example:
            input_str = json.loads(content)
            handle_editor_content(input_str)
            templates = list_templates("templates")

        if st.session_state["new_template_added"]:
            res = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
            selected_endpoint = sidebar_selectbox.selectbox(
                label="Select the endpoint to run in SageMaker",
                options=list_templates("templates"),
                key=res
            )
    
    # Prompt Engineering Playground
    st.header("Prompt Engineering Playground")
    output_text = read_template(f"templates/{selected_endpoint}.template.json")
    output = json.loads(output_text)

    parameters = output["payload"]["parameters"]
    parameters = handle_parameters(parameters)

    st.markdown(
    """
    Example:  :red[For Few Shot Learning]

    **:blue[List the Country of origin of food.]**
    Pizza comes from Italy
    Burger comes from USA
    Curry comes from
    """
    )

    prompt = st.text_area("Enter your prompt here:", height=350)
    placeholder = st.empty()

    if st.button("Run"):
        placeholder = st.empty()
        endpoint_name = output["endpoint_name"]
        payload = {
            "inputs": [
                prompt,
            ],
            "parameters": parameters,
        }
        generated_text = generate_text(payload, endpoint_name)
        st.write(generated_text)


if __name__ == "__main__":
    main()
