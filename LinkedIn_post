from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

prompt = PromptTemplate(
    input_variables=["topic", "tone", "audience"],
    template="""
You are an AI assistant optimized to write concise, professional, and friendly linkedin post.

You are a professional LinkedIn content writer.

Write a highly engaging LinkedIn post about: {topic}

Target Audience:
{audience}

Tone:
{tone}

Requirements:
- Start with a strong hook
- Keep the post concise and readable
- Use short paragraphs
- Add storytelling when relevant
- Include actionable insights
- End with an engaging question or CTA
- Add relevant emojis naturally
- Include relevant hashtags at the end

Output only the LinkedIn post.
"""
)

topic = input("Topic Name: ")
tone = input("What should be the tone: ")
audience = input("Who are the target audience: ")

final_prompt = prompt.format(
    topic=topic,
    tone=tone,
    audience=audience
)

response = llm.invoke(final_prompt)
print("")
print(response.content)
