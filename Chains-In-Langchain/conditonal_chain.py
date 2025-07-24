from altair import condition
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation",
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or negative.\n"
        "Feedback: {feedback}\n\n"
        "{format_instructions}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# print(classifier_chain.invoke({'feedback': "this is wonderful smartphone"}))

prompt2=PromptTemplate(
    template="write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template="write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser2),
    (lambda x:x.sentiment=='negative',prompt3|model|parser2),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain=classifier_chain|branch_chain

print(chain.invoke({'feedback':'this is a terrible phone'}))
