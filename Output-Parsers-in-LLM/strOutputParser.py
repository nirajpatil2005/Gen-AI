from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

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

prompt1=template1.invoke({'topic':'black hole'})
result=model.invoke(prompt1)

prompt2=template2.invoke({'text':result.content})
result1=model.invoke(prompt2)

print(result1.content)

#output:

# Black holes are spacetime regions where gravity traps everything, born from massive-star collapse or primordial fluctuations.  
# They come in Schwarzschild, Kerr, Reissner–Nordström and Kerr–Newman types, each defined by mass, spin and charge.  
# Gravitational waves, X-ray disks, stellar orbits and horizon images confirm their existence.  
# Hawking radiation, black-hole thermodynamics and the information paradox link gravity with quantum theory.  
# Future missions like LISA and upgraded EHT aim to resolve singularities, firewalls and the origin of supermassive black holes.