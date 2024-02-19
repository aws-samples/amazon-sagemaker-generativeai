"""
Text to SQL generator demo
"""

####################### IMPORTS ####################################
import streamlit as st
import random
import const
import utils

######################## STREAMLIT UI ################################### 

st.set_page_config(layout="wide") # set layout to wide so we can use 2 columns
# set the title
st.title("Natural language to SQL:")

# create a container so we can split in containers
app_container = st.container()
col1, col2 = st.columns([1, 1])
with app_container:
    with col1: # left column
        st.caption("""
           This Generative AI demo lets you query structured data from Amazon Relational Database Service (RDS) with natural language.
            You can now easily consume your data without any knowledge of SQL in a chat like manner. \n
            Give it a try with our pre-configured examples!
           """)
        
        # Load premade example for easier usability
        if st.button(label="Example"):
            st.session_state.prompt = random.choice(const.EXAMPLE_PROMPTS)

        # first textbox where the user will enter their query in natural language
        st.text_area("Your query in natural language :female-technologist: :male-technologist:", key="prompt")


        # call the function to generate a sql query
        if st.button(label="Generate SQL") and st.session_state.prompt:
            # use a prompt to generate an sql query from the given user input
            prompt = st.session_state.prompt 
            try:
                with st.spinner("Generating SQL query..."):
                    query = utils.text2sql(prompt)
                    st.session_state.results = query
                    st.session_state.query = query
            except Exception as e:
                err=f"Unable to call model or interact with database - {e}"
                st.error(err, icon="🚨")
        try:
            if not st.session_state.prompt or not st.session_state.query:
                st.session_state.query = ""
        except AttributeError:
            st.session_state.query = ""


        # Display the query and let the user edit it per needs
        st.write("Generated SQL query :computer:")
        st.code(st.session_state.query, language="sql", line_numbers=True)


        # Let the user download the query
        st.download_button(
            label="Download query",
            data=st.session_state.query.encode('utf-8'),
            file_name='query.sql',
            mime='text/plain',
            )

    with col2: # right column
        # demo db schema
        expander = st.expander("See DemoDB schema")
        expander.write("This is a demo database, where your query can be executed on!")

        expander.image("./schema.png")

        if st.button(label="Execute query") and st.session_state.query:
            df = utils.execute_query(st.session_state.query)
            if df is not None:
                st.subheader("Query results:")
                table = df.style.format(precision=0, thousands='') # set this so streamlit doesn't mistake integers for floats
                st.table(table)
                exp = utils.explain_result(query=st.session_state.prompt, result=df.values.tolist())
                st.subheader("Answer:")
                st.write(exp)
            else:
                st.error("Unable to execute query.", icon="🚨")