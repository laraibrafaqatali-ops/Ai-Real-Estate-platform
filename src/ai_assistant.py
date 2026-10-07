import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class AIAssistant:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI Real Estate Consultant. Provide clear, concise, and accurate advice regarding property valuation, investment strategies, neighborhood analysis, and market trends."),
            ("human", "{user_query}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def ask(self, query: str) -> str:
        return self.chain.invoke({"user_query": query})