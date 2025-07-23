from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
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

template1=PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)
template2=PromptTemplate(
    template="write a 5 line summary on following ./n {text}",
    input_variables=['text']
)

parser=StrOutputParser()
chain=template1 | model | parser |template2|parser
result=chain.invoke({'topic':'black hole'})

print(result)

