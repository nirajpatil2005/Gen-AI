from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Set up HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation",
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model = ChatHuggingFace(llm=llm)

# Define expected schema
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Define prompt with format instructions
template = PromptTemplate(
    template="""
Give 3 facts about the topic: {topic}

{format_instructions}
""",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# Chain together prompt, model, and parser
chain = template | model | parser

# Invoke with topic
final_result = chain.invoke({'topic': 'Nikola Tesla'})
print(final_result)

#output
# {
# 'fact_1': 'Nikola Tesla was born during a lightning storm at midnight between July 9 and 10, 1856, in Smiljan, Austrian Empire (modern-day Croatia).',
# 'fact_2': 'He held over 300 patents worldwide, including the first practical AC induction motor and the Tesla coil, which became fundamental to radio technology.', 
# 'fact_3': 'Tesla once envisioned a global wireless communication system in the early 1900s, constructing the Wardenclyffe Tower in New York to transmit wireless signals and power across the planet.'
# 
# }