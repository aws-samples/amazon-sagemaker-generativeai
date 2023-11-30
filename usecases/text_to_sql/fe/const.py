
import os 
from langchain.prompts import PromptTemplate

##################### CONSTANTS ####################################

# Endpoint name used by the SageMaker model hosting
ENDPOINT_NAME = "code-llama-7b" # change depending on your endpoint name

# Example prompts for the user to try out
EXAMPLE_PROMPTS = [
    "What is Velma's employee id?",
    "How many hours did Peter work in August 2022?",
    "Who worked the most hours in May 2022?",
    "How many Software Engineers does the company have?",
    "Who are the SDEs?",
    "Who are the Employees of the company?",
    "List all Software Engineers who have Peter as their manager",
    "Who are the Software Engineers working on the 'Restaurant Management App' project?"
]

REGION = 'us-east-1'
os.environ['AWS_DEFAULT_REGION'] = REGION # set region to us-east-1 because endpoint is there

# Prompt template string
TEMPLATE = """Given an input question, create a syntactically correct {dialect} query to run.
Use the following format:

Question: "Question here"
SQLQuery:
"SQL Query to run"

Only use the following tables:

{table_info}.

Some examples of SQL queries that correspond to questions are:

{few_shot_examples}

Question: {input}"""

# curate prompt template
CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "few_shot_examples", "table_info", "dialect"], template=TEMPLATE
)

# few shot examples for the user to try out
FEW_SHOT_EXAMPLES = """

Question: Find what is Peter's email adress.
SQL Query:
SELECT email FROM employees WHERE first_name='Peter';

##

Question: How many Software Engineers does the company have?
SQL Query:
SELECT COUNT(*) from employees
WHERE designation='Software Engineer';

##

Question: How many hours did Velma work in July 2022?
SQL Query:
SELECT SUM(t.entered_hours) AS total_hours_worked
FROM employees e
JOIN timelog t ON e.employee_id = t.employee_id
WHERE e.first_name = 'Velma'
  AND EXTRACT(YEAR FROM t.working_day) = 2022
  AND EXTRACT(MONTH FROM t.working_day) = 7;

##

Question: Who is working on the Music generator project?
SQL Query:
SELECT * FROM employees
WHERE project_id=(
SELECT project_id FROM projects
WHERE project_name = 'Music generator'
);

##

Question: Who works under Max?
SQL Query:
SELECT * FROM employees
WHERE manager_id=(
SELECT employee_id FROM employees
WHERE first_name = 'Max');

##

Question: Who worked the most hours in April 2022?
SQL Query:
SELECT e.first_name, e.last_name, SUM(t.entered_hours) AS total_hours_worked
FROM employees e
JOIN timelog t ON e.employee_id = t.employee_id
WHERE EXTRACT(YEAR FROM t.working_day) = 2022
  AND EXTRACT(MONTH FROM t.working_day) = 4
GROUP BY e.employee_id, e.first_name, e.last_name
ORDER BY total_hours_worked DESC
LIMIT 1;

##

"""