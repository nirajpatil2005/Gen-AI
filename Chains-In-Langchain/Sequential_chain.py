from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
hf_token=os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='generate a 5 pointer summary from the following test \n {text}',
    input_variables=['text']
)
parser=StrOutputParser()
chain=prompt1|model|parser|prompt2|model|parser

result=chain.invoke({'topic':"unemployment in india"})
print(result)