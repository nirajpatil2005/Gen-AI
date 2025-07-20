from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,Optional
# Load environment variables from .env file
load_dotenv()

# Fetch the Hugging Face token from the environment variable
hf_token = os.getenv("HF_TOKEN")

# Set up the Hugging Face model endpoint
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

model=ChatHuggingFace(llm=llm)

#schema
class Review(TypedDict):
    
    key_themes:Annotated[str,"Write down all the key themes discussed in the review in a list"]
    summary:Annotated[str,"A brief summay of the review"]
    sentiment:Annotated[str,"Return Sentiment of the review either positve,negative or neutral"]
    pros:Annotated[Optional[list[str]],"Write down all pros inside a list"]
    cons:Annotated[Optional[list[str]],"Write down all cons inside a list"]
    name:Annotated[Optional[str],"Write name of the reviewer"]
    
    
structured_model=model.with_structured_output(Review)

review="""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by jcv
"""

result =structured_model.invoke(review)

print(result)
print(type(result))
print(result['summary'])
print(result['sentiment'])

#output
# {'summary': 'Upgraded to the Samsung Galaxy S24 Ultra and finds it to be an absolute powerhouse.', 'sentiment': 'positive'}
# <class 'dict'>
# Upgraded to the Samsung Galaxy S24 Ultra and finds it to be an absolute powerhouse.
# positive