import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate,load_prompt
# Load environment variables from .env file
load_dotenv()

# Fetch the Hugging Face token from the environment variable
hf_token = os.getenv("HF_TOKEN")

# Set up the Hugging Face model endpoint
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

# Initialize the model
model = ChatHuggingFace(llm=llm)

chat_history=[]
while True:
    
    user_input=input('You: ')
    chat_history.append(user_input)
    if user_input =='exit':
        break
    result=model.invoke(chat_history)
    print("AI: ",result.content)
    
print(chat_history)