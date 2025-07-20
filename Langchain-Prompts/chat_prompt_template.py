from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

chat_template=ChatPromptTemplate.from_messages([
    ('System',"you are a helpful {domain} expert"),
    ('human',"explain in simple terms, what is {topic}")
])

prompt=chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)
