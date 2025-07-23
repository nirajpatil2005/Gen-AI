from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
load_dotenv()

hf_token=os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template=PromptTemplate(
    template="give me the name,age,city of a fictional person \n{format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
    
)

chain=template | model |parser

# prompt=template.format()
# #print(prompt)
result=chain.invoke({})
# result=model.invoke(prompt)

print(result) #{'name': 'Lila Moreau', 'age': 29, 'city': 'Montpellier'}