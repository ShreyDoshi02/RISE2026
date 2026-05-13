from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=1
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["situation", "batsman_name", "bowler_name"],
    template="""
You are an expert cricket commentator with an exciting and energetic voice.

Generate a thrilling cricket commentary for the following situation:

Situation: {situation}
Batsman: {batsman_name}
Bowler: {bowler_name}

The commentary should be lively, descriptive, and capture the excitement of the moment. Use cricketing terminology.
"""
)

# User Input
situation = input("Situation: ")
batsman_name = input("Batsman name: ")
bowler_name = input("Bowler name: ")

# Create Final Prompt
final_prompt = prompt.format(
    situation=situation,
    batsman_name=batsman_name,
    bowler_name=bowler_name
)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nGenerated Commentary:\n")
print(response.content)
