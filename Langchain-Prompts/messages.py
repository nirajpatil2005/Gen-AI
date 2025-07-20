import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate,load_prompt

from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

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

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about Langchain")
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)