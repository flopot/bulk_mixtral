import streamlit as st
import pandas as pd
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import AutoModelForCausalLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Custom CSS for styling
st.markdown("""
    <style>
        body {font-family: "Raleway", Sans-serif;}
    </style>
""", unsafe_allow_html=True)

# Title and Setup
st.title('Bulk Mixtral')

# File upload
uploaded_file = st.file_uploader("Choose your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    columns = df.columns.tolist()

    st.write("Map each column to a variable name for the prompts:")
    column_to_variable = {}
    for column in columns:
        variable_name = st.text_input(f"Enter a variable name for {column}", value=column)
        column_to_variable[column] = variable_name

    system_prompt = st.text_area("Edit the system prompt", value="Edit the system prompt.")
    user_prompt_template = st.text_area("Edit the user prompt", value="Edit the user prompt.")

    progress_text = st.empty()

    if st.button("Generate Responses"):
        # Load model and tokenizer
        tokenizer = MistralTokenizer.v1()
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

        all_responses = []

        def generate_response(row):
            formatted_user_prompt = user_prompt_template.format(**{var: row[col] for col, var in column_to_variable.items()})
            completion_request = ChatCompletionRequest(messages=[UserMessage(content=formatted_user_prompt)])
            tokens = tokenizer.encode_chat_completion(completion_request).tokens
            outputs = model.generate(torch.tensor([tokens]), max_new_tokens=100)
            response = tokenizer.decode(outputs[0].tolist())
            return response

        for index, row in df.iterrows():
            try:
                response = generate_response(row)
                all_responses.append([row[col] for col in columns] + [response])
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")

        response_df = pd.DataFrame(all_responses, columns=columns + ['Response'])
        csv = response_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download as CSV", data=csv, file_name="responses.csv", mime="text/csv")
