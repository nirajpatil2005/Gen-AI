from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()
hf_token=os.getenv("HF_TOKEN")

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model1=ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)
model2=ChatHuggingFace(llm=llm2)

prompt1=PromptTemplate(
    template='Generate short and simple notes from the following text \n {text} ',
    input_variables=['text']
)
prompt2=PromptTemplate(
    template='Generate 5 short questions answers from the following text \n {text} ',
    input_variables=['text']
)
prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes ->{notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser}
)

merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain
text="""
The tanh function, or hyperbolic tangent function, is a mathematical function that maps real numbers to the range of -1 to 1. It's commonly used in neural networks as an activation function, particularly in recurrent neural networks and some layers of feedforward networks. Its zero-centered output can speed up training by helping gradient descent converge faster. 
Key aspects of the tanh function:
Output Range:
It maps input values to a range between -1 and 1.
Zero-Centered:
Negative inputs result in negative outputs, and positive inputs result in positive outputs. This can be beneficial for optimization.
S-shaped curve:
Similar to the sigmoid function, the tanh function has an S-shaped curve when graphed.
Formula:
The tanh function is defined as (e^x - e^(-x)) / (e^x + e^(-x)), where 'e' is Euler's number (approximately 2.71828). 
Use in Neural Networks: 
Activation Function:
Tanh is frequently used as an activation function in neural networks, especially when dealing with inputs and outputs that are naturally negative or when zero-centered activations are preferred. 
Hidden Layers:
It's commonly found in hidden layers of neural networks, helping to introduce non-linearity. 
Recurrent Networks:
Tanh is particularly useful in recurrent neural networks (RNNs), which are designed to process sequential data. 
Comparison with other Activation Functions:
ReLU:
While tanh is a valuable activation function, ReLU and its variations (like Leaky ReLU) have become more popular in many modern neural network architectures due to their computational efficiency and ability to mitigate the vanishing gradient problem, according to Soulpage IT Solutions. 
Sigmoid:
Tanh is similar to the sigmoid function (which also outputs values between 0 and 1) but has the advantage of being zero-centered. 
Tanh Function - Soulpage IT Solutions
The tanh function is commonly used as an activation function in neural network architectures, especially in recurrent neural netwo...
Soulpage IT Solutions
What is the tanh activation function? - Educative.io
The tanh activation function, also called the hyperbolic tangent activation function, is a mathematical function commonly used in ...

Educative
Tanh (Hyperbolic Tangent) Explained - Ultralytics
Tanh (Hyperbolic Tangent) is a widely used activation function in neural networks. It is a mathematical function that squashes inp...
"""
result=chain.invoke({'text':text})
print(result)

#output