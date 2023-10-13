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
from io import StringIO
import re
from invoke_endpoint import *
from dict import *

N = 7
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)


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


def get_user_input():
    uploaded_file = st.file_uploader(
        label="Upload JSON Template", type=["json"])
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
            uploaded_file_location.write(
                "Template Uploaded: " + str(user_file_path))
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
            type(minimum) == float and type(
                maximum) == float and type(default) == float
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


def on_clicked():
    st.session_state.text = example_prompts_ai21[st.session_state.task]


def on_clicked_qa():
    st.session_state.text = example_context_ai21_qa[st.session_state.taskqa]


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
        res = "".join(random.choices(
            string.ascii_uppercase + string.digits, k=N))
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
            res = "".join(random.choices(
                string.ascii_uppercase + string.digits, k=N))
            selected_endpoint = sidebar_selectbox.selectbox(
                label="Select the endpoint to run in SageMaker",
                options=list_templates("templates"),
                key=res,
            )

    # Prompt Engineering Playground
    st.header("Prompt Engineering Playground")
    if selected_endpoint != default_endpoint_option:
        output_text = read_template(
            f"templates/{selected_endpoint}.template.json")
        output = json.loads(output_text)
        parameters = output["payload"]["parameters"]
        print("parameters ------------------ ", parameters)
        if parameters != "None":
            parameters = handle_parameters(parameters)

        st.markdown(
            output["description"]
        )
    if selected_endpoint == "AI21-J2-GRANDE-INSTRUCT":
        selected_task = st.selectbox(
            label="Example prompts",
            options=example_list,
            on_change=on_clicked,
            key="task"
        )
    if selected_endpoint == "AI21-CONTEXT-QA":
        selected_task = st.selectbox(
            label="Example context",
            options=example_context_ai21_qa,
            on_change=on_clicked_qa,
            key="taskqa"
        )
    if selected_endpoint == "AI21-SUMMARY" or selected_endpoint == "AI21-CONTEXT-QA":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # To read file as string:
            string_data = stringio.read()
            st.session_state.text = string_data
            prompt = st.session_state.text

    prompt = st.text_area("Enter your prompt here:", height=350, key="text")
    if selected_endpoint == "AI21-CONTEXT-QA":
        question = st.text_area(
            "Enter your question here", height=80, key="question")
    placeholder = st.empty()

    if st.button("Run"):
        final_text = ""
        if selected_endpoint != default_endpoint_option:
            placeholder = st.empty()
            endpoint_name = output["endpoint_name"]
            print(parameters)
            if parameters != "None":
                payload = {"inputs": prompt, "parameters": {**parameters}}
            else:
                payload = {"inputs": prompt}
            if output["model_type"] == "AI21":
                print('-------- Payload ----------', payload)
                generated_text = generate_text_ai21(payload, endpoint_name)
                final_text = f''' {generated_text} '''  # to take care of multi line prompt
                st.write(final_text)
            elif output["model_type"] == "AI21-SUMMARY":
                generated_text = generate_text_ai21_summarize(
                    payload, endpoint_name)
                summaries = generated_text.split("\n")
                for summary in summaries:
                    st.markdown("- " + summary)
                    final_text += summary
            elif output["model_type"] == "AI21-CONTEXT-QA":
                generated_text = generate_text_ai21_context_qa(
                    payload, question, endpoint_name)
                final_text = f''' {generated_text} '''  # to take care of multi line prompt
                st.write(final_text)
            else:
                generated_text = generate_text(payload, endpoint_name)
                final_text = f''' {generated_text} '''  # to take care of multi line prompt
                st.write(final_text)
        else:
            st.warning("Invalid Endpoint: Please select a valid endpoint")
        st.download_button("Download", final_text, file_name="output.txt")


if __name__ == "__main__":
    main()
