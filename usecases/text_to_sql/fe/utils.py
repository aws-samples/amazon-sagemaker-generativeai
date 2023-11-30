
import boto3
import const
import json
import psycopg2
import pandas as pd
from langchain.llms.sagemaker_endpoint import SagemakerEndpoint, LLMContentHandler
from langchain.chains import create_sql_query_chain
from langchain.sql_database import SQLDatabase
import streamlit as st
from typing import Dict


##################### FUNCTION DEFINITIONS ###############################

def _get_db_params():
    """Retrieves database connection parameters.

    This function retrieves the database connection parameters 
    including port, database name, username, password and endpoint
    from AWS Systems Manager Parameter Store.

    Returns:
        dict: A dictionary containing the database connection parameters 
            with keys 'port', 'db_name', 'user', 'password' and 'endpoint'.
    """
    ssm_client = boto3.client("ssm")
    port=5432
    db_name = "CompanyDatabase" 
    user=ssm_client.get_parameter(Name="text2sql_db_user", WithDecryption=True)["Parameter"]["Value"]
    password=ssm_client.get_parameter(Name="text2sql_db_password", WithDecryption=True)["Parameter"]["Value"]
    endpoint = ssm_client.get_parameter(Name="text2sql_db_endpoint", WithDecryption=True)["Parameter"]["Value"]
    db_params = {
        "port": port,
        "db_name": db_name,
        "user": user,
        "password": password,
        "endpoint": endpoint
    }
    return db_params

def _init_connection():
    """Initializes a read-only connection with our RDS PostgreSQL instance.

    Returns:
        psycopg2.connection: connection object to our RDS instance.
    """    
    db_params = _get_db_params()
    conn = psycopg2.connect(host=db_params["endpoint"], port=db_params["port"], user=db_params["user"], password=db_params["password"], dbname=db_params["db_name"])
    conn.set_session(readonly=True)
    return conn

# Perform query.
def execute_query(query):
    """Executes a SQL query and returns the result.

    Args:
        query (str): The SQL query to execute

    Returns:
        DataFrame: The result of the SQL query execution  
    """

    conn = _init_connection()
    try:
        with conn.cursor() as cursor:
            try:
                cursor.execute(query)
                df = pd.DataFrame(cursor.fetchall())
                df.columns = [desc[0] for desc in cursor.description]
                conn.close()
                return df
            except Exception as e:
                st.error('An error occured! - {}'.format(e), icon="🚨")
                conn.rollback()
    except psycopg2.InterfaceError as e:
        st.error('{} - Error connecting with the database'.format(e))
        conn.close()


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(
            {"inputs" : prompt,
            "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_text"]

    
def _populate_db():
    """Populates the database if tables are empty.

    Checks if the 'employees' table exists in the database. If not, 
    populates the database by executing SQL from the 'populate.sql' 
    file. Returns the result of the population query as a Pandas 
    DataFrame.

    Raises:
        RuntimeError: If the database is already populated.

    Returns: 
        pandas.DataFrame: The result of the population query.
    """

    # populate the db if empty
    check_empty_query = """
    SELECT 
        table_name
    FROM
        information_schema.tables;
    """
    check_empty_result = execute_query(check_empty_query)
    if "employees" not in check_empty_result.values:
        with open("populate.sql") as f:
            populate_query = f.read()
        commands = populate_query.split(";")
        del commands[-1]  # remove the last empty command
        conn = _init_connection()
        conn.set_session(readonly=False)
        with conn.cursor() as cursor:
            for command in commands:
                cursor.execute(command)
                conn.commit()
            conn.close()
    else:
        raise RuntimeError("Database already populated!")

def text2sql(query):
    """Converts a natural language query to SQL.

    Uses the SageMaker endpoint to call a large language model 
    that generates SQL from the input text query. Retrieves the
    database connection parameters and connects to the PostgreSQL
    database using the connection URL before returning the SQL.

    Args:
        query (str): The natural language query.

    Returns: 
        str: The generated SQL query.
    """

    # connect to our model hosted on a SageMaker endoint
    content_handler = ContentHandler()

    parameters = {"max_new_tokens": 200, "temperature": 0.2, "top_p": 0.9}

    llm_codellama = SagemakerEndpoint(
        endpoint_name=const.ENDPOINT_NAME,
        region_name=const.REGION,
        model_kwargs=parameters,
        endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
        content_handler=content_handler
    )

    # get the database url
    db_params = _get_db_params()
    rds_url = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['endpoint']}:{db_params['port']}/{db_params['db_name']}"

    try:
        # connect to the database
        db = SQLDatabase.from_uri(rds_url,
                           include_tables=["employees", "projects", "timelog"],
                           sample_rows_in_table_info=4)
    except ValueError:
        # populate the database
        _ = _populate_db()
        # connect to the database
        db = SQLDatabase.from_uri(rds_url,
                           include_tables=["employees", "projects", "timelog"],
                           sample_rows_in_table_info=4)

    # curate the prompt
    prompt = const.CUSTOM_PROMPT.format(
        input=query,
        table_info=db.table_info,
        dialect="PostgreSQL",
        few_shot_examples=const.FEW_SHOT_EXAMPLES
    )

    # create and invoke the chain to create the SQL query
    chain = create_sql_query_chain(llm_codellama, db)
    result = chain.invoke({"question": prompt})

    # format the query and return result
    return result.split(";")[0] + ";"

def explain_result(query, result):
    """Generates a natural language explanation for a SQL result.

    Takes the original query and result as input. Returns a  
    multi-line string with the query and result formatted for
    human readability, along with an explanation in natural
    language of what the result means.

    Args:
    query (str): The original SQL query
    result: The result of executing the SQL query

    Returns:
    str: A multi-line string with the formatted query, result  
        and natural language explanation
    """

    instruction = f"""
    I am building a Natural Language to SQL project. Please formulate an answer to my question in natural language in a human readable format.

    Query: 
    List all the software engineers. 
    Response: 
    [('Peter', 'Kabel', 'Software Engineer'), ('Max', 'Mustermann', 'Software Engineer'), ('Fidel', 'Wind', 'Software Engineer')]
    Explanation:
    The Software Engineers are Peter Kabel, Max Mustermann and Fidel Wind.

    ##

    Query: 
    How many hours did Peter work in August 2022?
    Response: 
    [(119,)]
    Explanation:
    Peter worked a total of 119 hours in August 2022.

    ##

    Query: 
    List all the projects.
    Response: 
    [(283921, 'Restaurant Management App', 'The Mozzarella Fellas'), (131032, 'Garden Planner', 'Flamingo Gardens'), (933012, 'Music generator', 'ElvisAI'), (311092, 'Weather forecasting system', 'Flamingo Gardens')]
    Explanation:
    Above is a list of all the projects of the company.

    ##

    Query:
    {query}
    Response: 
    {result}
    Explanation:
    """
    content_handler = ContentHandler()

    parameters = {"max_new_tokens": 100, "temperature": 0.2, "top_p": 0.9}

    llm_codellama = SagemakerEndpoint(
        endpoint_name=const.ENDPOINT_NAME,
        region_name=const.REGION,
        model_kwargs=parameters,
        endpoint_kwargs={"CustomAttributes": 'accept_eula=true'},
        content_handler=content_handler
    )
    explanation = llm_codellama(instruction)
    if "##" in explanation:
        return explanation.split("##")[0]
    return explanation