from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
load_dotenv()
# Fetch the Hugging Face token from the environment variable
hf_token = os.getenv("HF_TOKEN")

# Set up the Hugging Face model endpoint
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

def word_count(text):
    return len(text.split())
# Initialize the model
model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

prompt=PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)
joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        # 'word_count':RunnableLambda(word_count())
        'word_count':RunnableLambda(lambda x:len(x.split()))
    }
)

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({"topic":'AI'}))

# {'joke': 'Why did the AI break up with its calculator girlfriend?\n\nBecause every time it said “I love you,” she replied, “Syntax error—undefined variable.”', 'word_count': 23}