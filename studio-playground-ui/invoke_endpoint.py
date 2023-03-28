import boto3
from collections import defaultdict
import io
import jinja2
import json
import os
from pathlib import Path
import random
from streamlit_ace import st_ace
import streamlit as st
import string


N = 7
sagemaker_runtime = boto3.client("runtime.sagemaker")
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)


code_example = """{
  "model_name": "example",
  "endpoint_name": "jumpstart-example-infer-pytorch-textgen-2023-03-22-23-09-15-885",
  "payload": {
    "parameters": {
      "max_length": {
        "default": 200,
        "range": [
          10,
          500
        ]
      },
      "num_return_sequences": {
        "default": 10,
        "range": [
          0,
          10
        ]
      },
      "num_beams": {
        "default": 3,
        "range": [
          0,
          10
        ]
      },
      "temperature": {
        "default": 0.5,
        "range": [
          0,
          1
        ]
      },
      "early_stopping": {
        "default": true,
        "range": [
          true,
          false
        ]
      },
      "stopwords_list": {
        "default": [
          "stop",
          "dot"
        ],
        "range": [
          "a",
          "an",
          "the",
          "and",
          "it",
          "for",
          "or",
          "but",
          "in",
          "my",
          "your",
          "our",
          "stop",
          "dot"
        ]
      }
    }
  }
}
"""

parameters_help_map = {
    "max_length": "Model generates text until the output length (which includes the input context length) reaches max_length. If specified, it must be a positive integer.",
    "num_return_sequences": "Number of output sequences returned. If specified, it must be a positive integer.",
    "num_beams": "Number of beams used in the greedy search. If specified, it must be integer greater than or equal to num_return_sequences.",
    "no_repeat_ngram_size": "Model ensures that a sequence of words of no_repeat_ngram_size is not repeated in the output sequence. If specified, it must be a positive integer greater than 1.",
    "temperature": "Controls the randomness in the output. Higher temperature results in output sequence with low-probability words and lower temperature results in output sequence with high-probability words. If temperature -> 0, it results in greedy decoding. If specified, it must be a positive float.",
    "early_stopping": "If True, text generation is finished when all beam hypotheses reach the end of stence token. If specified, it must be boolean.",
    "do_sample": "If True, sample the next word as per the likelyhood. If specified, it must be boolean.",
    "top_k": "In each step of text generation, sample from only the top_k most likely words. If specified, it must be a positive integer.",
    "top_p": "In each step of text generation, sample from the smallest possible set of words with cumulative probability top_p. If specified, it must be a float between 0 and 1.",
    "seed": "Fix the randomized state for reproducibility. If specified, it must be an integer.",
}


parameters_help_map = defaultdict(str, parameters_help_map)


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


def is_valid_default(parameter, minimum, maximum):
    # parameter is a list
    if type(parameter) == list:
        return True

    # parameter is an int or float and is in valid range
    if parameter <= maximum and parameter >= minimum:
        return True

    # parameter is a bool
    if type(parameter) == bool and type(minimum) == bool and type(maximum) == bool:
        return True
    return False


def generate_text(payload, endpoint_name):
    encoded_input = json.dumps(payload).encode("utf-8")

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name, ContentType="application/json", Body=encoded_input
    )
    print("Model input: \n", encoded_input)
    result = json.loads(response["Body"].read().decode())

    # TO DO: results are either dictionary or list
    for item in result:
        if isinstance(item, list):
            for element in item:
                if isinstance(element, dict):
                    print(element["generated_text"])
                    return element["generated_text"]
        else:
            print(item["generated_text"])
            return item["generated_text"]


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

    if not "parameters" in json_dictionary["payload"].keys() and not type(
        json_dictionary["payload"]["parameters"] == list
    ):
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
        minimum = parameters[p]["range"][0]
        maximum = parameters[p]["range"][-1]
        default = parameters[p]["default"]
        parameter_range = parameters[p]["range"]
        parameter_help = parameters_help_map[p]
        if not is_valid_default(default, minimum, maximum):
            st.warning(
                "Invalid Default: "
                + p
                + " default value does not follow the convention default >= min and default <= max."
            )
        elif len(parameter_range) > 2:
            if not set(default).issubset(set(parameter_range)):
                st.warning(
                    "Invalid Default: "
                    + p
                    + " Every Multiselect default value must exist in options"
                )
            else:
                parameters[p] = st.sidebar.multiselect(
                    p, options=parameter_range, default=default
                )

        elif type(minimum) == int and type(maximum) == int and type(default) == int:
            parameters[p] = st.sidebar.slider(
                p,
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1,
                help=parameter_help,
            )
        elif type(minimum) == bool and type(maximum) == bool and type(default) == bool:
            parameters[p] = st.sidebar.selectbox(
                p, ["True", "False"], help=parameter_help
            )
        elif (
            type(minimum) == float and type(maximum) == float and type(default) == float
        ):
            parameters[p] = st.sidebar.slider(
                p,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default),
                step=0.01,
                help=parameter_help,
            )
        else:
            st.warning(
                "Invalid Parameter: "
                + p
                + " is not a valid parameter for this model or the parameter type is not supported in this demo."
            )
    return parameters


def main():
    default_endpoint_option = "Select"
    st.session_state["new_template_added"] = False
    sidebar_selectbox = st.sidebar.empty()
    selected_endpoint = sidebar_selectbox.selectbox(
        label="Select the endpoint to run in SageMaker",
        options=[default_endpoint_option] + list_templates("templates"),
    )

    st.sidebar.title("Model Parameters")
    st.image("./ml_image_prompt.png")

    # Adding your own model
    with st.expander("Add a New Model"):
        st.header("Add a New Model")
        st.write(
            """Add a new model by uploading a .template.json file or by pasting the dictionary
                in the editor. A model template is a json dictionary containing a model_name,
                endpoint_name, and payload with parameters.  \n \n Below is an example of a
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
                key=res,
            )

    # Prompt Engineering Playground
    st.header("Prompt Engineering Playground")
    if selected_endpoint != default_endpoint_option:
        output_text = read_template(f"templates/{selected_endpoint}.template.json")
        output = json.loads(output_text)

        parameters = output["payload"]["parameters"]
        parameters = handle_parameters(parameters)

    st.markdown(
        """
    Let's say we want to list the country of origin for foods. Example Input:  

    **:red[
    Pizza comes from Italy
    Burger comes from USA
    Curry comes from ...
    ]**
    """
    )

    prompt = st.text_area("Enter your prompt here:", height=350)
    placeholder = st.empty()

    if st.button("Run"):
        if selected_endpoint != default_endpoint_option:
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
        else:
            st.warning("Invalid Endpoint: Please select a valid endpoint")


if __name__ == "__main__":
    main()
