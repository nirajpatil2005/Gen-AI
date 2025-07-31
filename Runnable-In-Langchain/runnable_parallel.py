from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence,RunnableParallel
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
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="generate a linkedin post about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()
parallel_chain=RunnableParallel(
    {
        'tweet':RunnableSequence(prompt1,model,parser),
         'linkedin':RunnableSequence(prompt2,model,parser)
    }
)

result=parallel_chain.invoke({'topic':'India'})
print(result)

# {'tweet': 'India: where ancient temples hum with history and startups spark tomorrow—chaos, color, and chai in every breath.', 'linkedin': '🇮🇳 India isn’t just a country—it’s a 1.4-billion-person masterclass in reinvention.  \n\nIn the last 12 months alone:  \n• We became the world’s most populous nation—and turned that scale into the fastest-growing major economy.  \n• UPI crossed 10 billion transactions a month, moving more money digitally in a week than most countries do in a year.  \n• We landed on the moon’s south pole—on a budget smaller than a Hollywood blockbuster.  \n• And we hosted the G20, showing the world how ancient heritage and cutting-edge tech can share the same stage.  \n\nFrom semiconductor fabs in Gujarat to AI research in Bengaluru, from drone deliveries in Arunachal to green hydrogen hubs in Tamil Nadu, the story isn’t “emerging” anymore—it’s arriving.  \n\nIf you’re still asking “Why India?” you’re asking the wrong question. The real question is: “Why not yesterday?”  \n\n#India #Innovation #DigitalEconomy #SpaceTech #FutureIs sHere'}