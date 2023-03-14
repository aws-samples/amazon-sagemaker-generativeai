import streamlit as st
import boto3
import json
import io

sagemaker_runtime = boto3.client('runtime.sagemaker')

def generate_text(payload, endpoint_name_str):
    #- payload = {"inputs": prompt, "parameters": params}
    encoded_inp = json.dumps(payload).encode("utf-8")
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name_str,
        ContentType='application/json',
        Body=encoded_inp
    )
    if endpoint_name_radio == 'STABLE-DIFFUSION':
        return handle_stable_diffusion(response)

    result = json.loads(response['Body'].read().decode()) # -
    print(result)
    if endpoint_name_radio == 'ALEXA-20B':
        text = parse_alexa_response(result)
    elif endpoint_name_radio == 'GPT-NEOX-20B':
        text = parse_gpt_neox_response(result)
    elif endpoint_name_radio == 'FLAN-T5':
        text = parse_flan_t5_response(result)
    else:
        text = parse_gpt_response(result) # - result[0]['generated_text']
    return text 
def handle_stable_diffusion(response):
    print(response)
    img_res = io.BytesIO(response['Body'].read())
    placeholder  = st.image(img_res)
    return prompt

st.image('./ml_image.jpg')


st.header("Few Shot Playground")
endpoint_name='gpt-j-deploy-2023-01-21-20-01-49-923'
length=50 # max length variation

# End Point names
endpoint_name_radio = st.sidebar.selectbox(
    "Select the endpoint to run in SageMaker",
    (
        'GPT-J',
        'ALEXA-20B',
        'GPT-NEOX-20B',
        'FLAN-T5',
        'STABLE-DIFFUSION',
        'BLOOM-176B'
    ),
    index=2
)

dict_endpoint = {
    "GPT-J" : "gptj-ds-2023-03-06-18-19-50-643", #"gpt-j-deploy-2023-01-21-20-01-49-923",
    "ALEXA-20B" : "jumpstart-example-infer-pytorch-textgen-2023-03-10-03-43-03-082",
    "GPT-NEOX-20B" : "gpt-neox-djl20-ds-2023-03-10-03-41-25-326-endpoint",
    "STABLE-DIFFUSION" : "stable-diffusion-2023-02-10-21-25-33-687",
    "FLAN-T5" : "jumpstart-example-huggingface-text2text-2023-03-14-21-49-46-617",


}
# Sidebar title
st.sidebar.title("LLM Model Parameters")


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

def parse_alexa_response(query_response):
    #generated_text = query_response["generated_texts"][0]
    #json.loads(query_response)["generated_texts"]

    # Trim using the delimiter
    #return generated_text.split("<br>")[0].strip()

    return query_response["generated_texts"]

def parse_gpt_response(query_response):
    return query_response[0][0]['generated_text']

def parse_gpt_neox_response(query_response):
    return query_response['outputs'][0]

def parse_flan_t5_response(query_response):
    return query_response["generated_texts"]


def get_params(curr_length, endpoint_name_radio):
    if endpoint_name_radio == 'ALEXA-20B':
        return get_params_alexa(curr_length)
    elif endpoint_name_radio == 'GPT-NEOX-20B':
        return get_params_gptneox(curr_length)
    elif endpoint_name_radio == 'STABLE-DIFFUSION':
        return get_params_stable_diffusion(curr_length)
    elif endpoint_name_radio == 'FLAN-T5':
        return get_params_flant(curr_length)
    else:
        return get_params_gptj(curr_length)

def get_params_alexa(curr_length):
    params = {
        'text_inputs': prompt,
        'max_length': curr_length, #len(prompt) // 4 + curr_length + 5,
        'num_return_sequences': int(beams_no) -1,
        'num_beams': int(beams_no),
        'top_p':0.5, # equal probability
        'top_k':0, #20,
        'early_stopping': 'True' == early_stopping,
        #early_stopping=True so that generation is finished when all beam hypotheses reached the EOS token.
        'do_sample': 'True' == do_sample_st , #True,
        'no_repeat_ngram_size': int(rep_penalty),
        'temperature':temp,
        'seed':int(seed_no)
    }
    if temp == 0:
        print("ALEXA-20B: No temperature so no Diversity or rare tokens in generation")
        params.pop('temperature')
        params['do_sample'] = False
    if int(beams_no) == 0:
        print("ALEXA-20B:: No BEEMS")
        params.pop('num_beams')
        params.pop('num_return_sequences')
    if int(beams_no) == 1:
        params['num_return_sequences'] = 1

    print("ALEXA-20B", endpoint_name_radio, params, curr_length)
    return params

def get_params_flant(curr_length):
    print(do_sample_st,early_stopping)
    params = {
        "text_inputs": prompt,
        "max_length": curr_length,
        "num_return_sequences": 3,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": 'True' == do_sample_st,
        "temperature" : temp
    }
    if temp < 0.25 :
        print("FLAN-T5: No temperature so no Diversity or rare tokens in generation")
        params.pop('temperature')
        params['do_sample'] = False


    print("FLANT-T5::", endpoint_name_radio, params,curr_length)

    return params

# - https://www.fidelity.com/learning-center/personal-finance/personal-inflation-rate

def get_params_gptj(curr_length):
    print(do_sample_st,early_stopping)
    params = {
              "return_full_text": True,
              "temperature": temp,
              #"min_length": len(prompt), #len(prompt) // 4 + length - 5,
              "max_length": curr_length, #len(prompt) // 4 + length + 5,
              'early_stopping': 'True' == early_stopping,
              'num_beams': int(beams_no),
              #'num_return_sequences' : int(beams_no) - 1,
              'no_repeat_ngram_size': int(rep_penalty),
              "do_sample": 'True' == do_sample_st , #True,
              "repetition_penalty": float(rep_penalty),
              #"top_k": 0, #20,
              #"seed":int(seed_no)
    }
    if temp == 0:
        print("GPT: No temperature so no Diversity or rare tokens in generation")
        params.pop('temperature')
        params['do_sample'] = False
    if int(beams_no) == 0:
        print("GPT: No BEEMS")
        params.pop('num_beams')
        #params.pop('num_return_sequences')


    payload = {"inputs": [prompt,],  "parameters": params}
    print("GPT::", endpoint_name_radio, payload,curr_length)

    return payload

def get_params_gptneox(curr_length):
    #print(do_sample_st,early_stopping)
    params = {
              #"return_full_text": True,
              "temperature": temp,
              #"min_length": len(prompt), # len(prompt) // 4 + length - 5,
              "max_length": curr_length, #len(prompt) // 4 + length + 5,
              'early_stopping': 'True' == early_stopping,
              'num_beams': int(beams_no),
              #'num_return_sequences' : int(beams_no) - 1,
              'no_repeat_ngram_size': int(rep_penalty),
              "do_sample": 'True' == do_sample_st, #True,
              "repetition_penalty": float(rep_penalty),
              "top_k": 0, #20,
              #"seed":int(seed_no)
    }
    #print(params)
    if temp == 0:
        print("GPT: No temperature so no Diversity or rare tokens in generation")
        params.pop('temperature')
        params['do_sample'] = False
    if int(beams_no) == 0:
        print("GPT: No BEEMS")
        params.pop('num_beams')
        #params.pop('num_return_sequences')


    payload = {"inputs": [prompt,],  "parameters": params}
    print("GPT-NEOX::", endpoint_name_radio, payload,curr_length)

    return payload

def get_params_stable_diffusion(curr_length):
    #print(do_sample_st,early_stopping)
    params = {
        "num_inference_steps": 50, 
        "guidance_scale":7.5,
        "negative_prompt": "low quality poorly lit",
        "seed": "9080980",
        "num_images_per_prompt": 4
    }

    payload = {"prompt": f"""{prompt}""",  "parameters": params}
    print("STABLE-DIFFUSION::", endpoint_name_radio, payload,curr_length)

    return payload


if st.button("Run"):
    placeholder = st.empty()
    curr_length = max_length.get(length_choice, 10)
    curr_length = curr_length * 5 # for scaling 
    payload = get_params(curr_length,endpoint_name_radio)
    endpoint_name_s = dict_endpoint.get(endpoint_name_radio, "gptj-ds-2023-02-11-02-56-05-104")

    generated_text = generate_text(payload,endpoint_name_s)
    #print(generated_text)
    st.write(generated_text)


