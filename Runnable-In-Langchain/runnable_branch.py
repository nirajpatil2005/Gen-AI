from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain.schema.runnable import RunnableBranch
load_dotenv()
# Fetch the Hugging Face token from the environment variable
hf_token = os.getenv("HF_TOKEN")

# Set up the Hugging Face model endpoint
llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",  # Use the correct model repo_id
    task="text-generation",  # You can adjust this based on your model task
    model_kwargs={"headers": {"Authorization": f"Bearer {hf_token}"}}
)

prompt1=PromptTemplate(
template='Write a detailed report on {topic}',
input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)
model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()

report_gen_chain=RunnableSequence(prompt1,model,parser)
branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia Vs Ukrain'}))

# Russia-Ukraine War – June 2024 Snapshot

# • Europe’s largest interstate war since 1945 is now a 1,200 km positional slog. Ukraine still controls ~82 % of its territory; Russia holds ~18 %, including Crimea and a land bridge to it.

# • Casualties: 30,000+ confirmed civilian dead (likely under-counted); ~150,000+ military fatalities estimated on each side.

# • Forces: Russia fields 470 k+ troops, 2 k+ tanks, 4 k artillery pieces, 1,200+ aircraft, and a $140 bn defense budget. Ukraine has 800 k+ mobilized, 1.5 k tanks (many Western), 1,200+ Western artillery pieces, and a $43 bn budget.

# • Asymmetric tools: Ukraine relies on long-range drones, Storm Shadow/ATACMS, and Starlink-enabled ISR; Russia on glide-bombs, Shahed drones, and nuclear threats.

# • Economy & energy: EU/G7 sanctions and a $60 oil cap have not prevented Russia from earning $183 bn in 2023 via a “shadow fleet.” Ukraine’s GDP rebounded 5 % in 2023 but still needs $38 bn in external financing; reconstruction cost estimated at $486 bn over 10 years.

# • Humanitarian toll: 14.6 m Ukrainians need aid; 6.5 m refugees abroad; 3.7 m IDPs; 90 % of children show PTSD symptoms.

# • Diplomacy frozen: ICC warrant for Putin, NATO enlarged to Finland & Sweden, Ukraine an EU candidate, but no ceasefire talks. Swiss-led peace summit (June 2024) excluded Russia and drew no Chinese participation.

# • Five plausible futures (2024-26): frozen conflict (45 %), gradual Ukrainian rollback (25 %), Russian breakthrough (15 %), NATO-Russia escalation (10 %), or negotiated settlement (5 %).        

# Bottom line: absent a decisive shift, the war is settling into a protracted, high-casualty stalemate with global economic and security spillovers.