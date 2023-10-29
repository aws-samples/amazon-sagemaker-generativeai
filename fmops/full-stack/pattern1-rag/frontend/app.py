import streamlit as st
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import os
import json
import uuid
import logging
from typing import Dict
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import SagemakerEndpoint, PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.chains.question_answering import load_qa_chain

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

opensearch_domain_endpoint = os.environ['OPENSEARCH_ENDPOINT']
opensearch_index = os.environ['OPENSEARCH_INDEX']
ddb_table_name = os.environ['DDB_TABLE_NAME']
fh_name = os.environ['FIREHOSE']
text_model_endpoint = os.environ['TEXT_MODEL_ENDPOINT']
embed_model_endpoint = os.environ['EMBEDDINGS_MODEL_ENDPOINT']


cwclient = boto3.client('cloudwatch')
fhclient = boto3.client('firehose')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(ddb_table_name)
credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name

def put_ddb_item(item):
    try:
        table.put_item(Item=item)
    except ClientError as err:
        logger.error(err.response['Error']['Code'], err.response['Error']['Message'])
        raise

def get_ddb_item(id):
    try:
        items = table.query(KeyConditionExpression=Key('id').eq(id))['Items'][0]
        return items
    except ClientError as err:
        logger.error(err.response['Error']['Code'], err.response['Error']['Message'])
        raise

def put_cw_metric(cwclient, score):
    try:
        cwclient.put_metric_data(
            Namespace='rag',
            MetricData=[
                {
                    'MetricName': 'similarity',
                    'Value': score,
                    'Unit': 'None',
                    'StorageResolution': 1
                },
            ]
        )
    except ClientError as err:
        logger.error(err.response['Error']['Code'], err.response['Error']['Message'])
        raise

class TextContentHandler(LLMContentHandler):
    """
    encode input string as utf-8 bytes, read the generated text
    from the output
    """
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs = {}) -> bytes:
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]['generated_text']

class EmbeddingsContentHandler(EmbeddingsContentHandler):
    """
    encode input string as utf-8 bytes, read the embeddings
    from the output
    """
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"text_inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes):
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]

def create_sagemaker_embeddings(endpoint_name):
    # create a content handler object which knows how to serialize
    # and deserialize communication with the model endpoint
    content_handler = EmbeddingsContentHandler()

    # read to create the Sagemaker embeddings, we are providing
    # the Sagemaker endpoint that will be used for generating the
    # embeddings to the class
    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=endpoint_name,
        region_name=region, 
        content_handler=content_handler
    )

    return embeddings

# Functiion to do vector search and get context from opensearch. Returns list of documents
def get_context_from_opensearch(query, endpoint_name, opensearch_domain_endpoint, opensearch_index):

    opensearch_endpoint = f"https://{opensearch_domain_endpoint}"
    docsearch = OpenSearchVectorSearch(
        index_name=opensearch_index,
        embedding_function=create_sagemaker_embeddings(endpoint_name),
        opensearch_url=opensearch_endpoint,
        is_aoss=False
    )
    docs_with_scores = docsearch.similarity_search_with_score(query, k=3, vector_field="embedding", text_field="passage")
    for d in docs_with_scores:
        score = d[1]
        put_cw_metric(cwclient, score)
    docs = [doc[0] for doc in docs_with_scores]
    logger.info(f"docs received from opensearch:\n{docs}")
    return docs # return list of matching docs

# Function to combine the context from vector search, combine with question and query sage maker deployed model
def call_sm_text_generation_model(query, context, endpoint_name):

    # create a content handler object which knows how to serialize
    # and deserialize communication with the model endpoint
    content_handler = TextContentHandler()
    
    ## Query to sagemaker endpoint to generate a response from query and context
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,
        content_handler=content_handler,
        endpoint_kwargs={'CustomAttributes': 'accept_eula=true'}
    )
    prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    logger.info(f"prompt sent to llm = \"{prompt}\"")
    chain = load_qa_chain(llm=llm, prompt=prompt)
    answer = chain({"input_documents": context, "question": query}, return_only_outputs=True)['output_text']
    logger.info(f"answer received from llm,\nquestion: \"{query}\"\nanswer: \"{answer}\"")
    
    return answer

question_data = {}
conversation_id = uuid.uuid4()
st.session_state.id = conversation_id
question_data['id'] = str(conversation_id)
st.set_page_config(page_title="Embedding Analysis for Drift Detection")
st.markdown("# Embedding Analysis")
input_text = st.text_input('Whats your question?', key='text')
st.session_state.query =input_text
question_data['query'] = input_text
get_answer_button = st.button('Answer', type="primary")

if get_answer_button:
    with st.spinner('Searching for similar documents for context...'):
        context = get_context_from_opensearch(st.session_state.query, embed_model_endpoint, opensearch_domain_endpoint, opensearch_index)
        context_formatted =  [{"page_content": doc.page_content} for doc in context]
        st.session_state.context = context
        question_data['context'] = context_formatted
        st.success(f"Found {str(len(context))} similar documents")

    with st.spinner('Generating Answer...'):
        answer = call_sm_text_generation_model(st.session_state.query, st.session_state.context, text_model_endpoint)
        st.write(answer)
        st.success(f"Conversation ID is {conversation_id}")
        question_data['answer'] = answer
        st.session_state.answer = answer
    
    fh_stream_records = []
    embedding = create_sagemaker_embeddings(embed_model_endpoint).embed_query(st.session_state.query)
    fh_stream_records.append({'Data': (str(embedding)+ "\n").encode('utf-8')})
    fhclient.put_record_batch( DeliveryStreamName=fh_name, Records=fh_stream_records)
    put_ddb_item(question_data)

