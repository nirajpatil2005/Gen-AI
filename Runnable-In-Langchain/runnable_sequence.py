from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence
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

prompt1=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()

# chain=RunnableSequence(prompt1,model,parser)

# print(chain.invoke({'topic':'india'}))

prompt2=PromptTemplate(
    template="Explain the folowing joke - {text}",
    input_variables=['text']
)

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))