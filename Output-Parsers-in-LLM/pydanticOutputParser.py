from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
load_dotenv()

hf_token=os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    
    name:str=Field(description='Name of the person')
    age:int=Field(gt=18,description="Age of the person")
    city:str=Field("name of the city the person belongs to ")
    
parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="""
    Generate the name,age and city of a fictional {place}
    person \n{format_instructions}
    """,
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt=template.invoke({'place':'indian'})
print(prompt)
result=model.invoke(prompt)
result=parser.parse(result.content)
print(result)

chain=template|model|parser
result=chain.invoke({'place':'indian'})
print(result)
#output
