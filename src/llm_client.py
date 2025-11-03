# src/llm_client.py
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

# Load environment variables first (.env in repo root)
load_dotenv()


class LLMClient:
    def __init__(self):
        # Initialize with API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Helpful warning — but don't raise so code can still run with fallback planner
            print("[llm_client] WARNING: OPENAI_API_KEY not set; LLM calls will fail if attempted.")

        # Chat-capable model (chat interface)
        try:
            self.chat = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=openai_api_key
            )
        except Exception as ex:
            # Fallback: keep attribute but note failure
            print(f"[llm_client] ChatOpenAI creation failed: {ex}")
            self.chat = None

        # Non-chat LLM (text) — keep it small by default
        try:
            self.llm = OpenAI(
                model="gpt-3.5-turbo-instruct",
                temperature=0.2,
                max_tokens=200,
                api_key=openai_api_key
            )
        except Exception as ex:
            print(f"[llm_client] OpenAI creation failed: {ex}")
            self.llm = None

    def get_chat_model(self):
        return self.chat

    def get_llm_model(self):
        return self.llm


# Singleton instance used by other modules (llm_planner expects this name)
llm_client = LLMClient()
