import openai
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_client():
    client = openai.OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    logger.info(f"OpenAI client initialized. OpenAI base URL: {os.environ.get('OPENAI_BASE_URL')}")
    return client

def  get_model_real_name(model,proxy="LiteLLM"):
      if proxy == "GROK" and model:
          return f"openai/{model}"
      return model