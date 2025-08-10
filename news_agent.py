import os
import requests
import json
import yfinance as yf
from datetime import datetime
import openai
import yaml
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory

# Load API keys from config
def load_yaml(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        return {}

config_data = load_yaml('./config.yaml')
os.environ["OPENAI_API_KEY"] = config_data["OPENAI_API_KEY"]
os.environ["NEWSAPI_API_KEY"] = config_data["NEWSAPI_API_KEY"]
os.environ["OPENWEATHER_API_KEY"] = config_data["OPENWEATHER_API_KEY"]

# Define tools using LangChain
@tool
def fetch_news(params: str) -> str:
    """Fetches news articles based on a topic. Params should be a JSON string with 'topic', 'language', and 'page_size'."""
    try:
        inputs = json.loads(params)
        topic = inputs.get("topic", "")
        language = inputs.get("language", "en")
        page_size = inputs.get("page_size", 5)
    except json.JSONDecodeError:
        return "Invalid input. Provide a JSON string with 'topic', 'language', and 'page_size'."

    api_key = os.getenv("NEWSAPI_API_KEY")
    base_url = "https://newsapi.org/v2/everything"
    query_params = {
        "q": topic,
        "language": language,
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        response = requests.get(base_url, params=query_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            return f"No articles found for topic: {topic}."
        return "\n".join([f"{i+1}. {article['title']}" for i, article in enumerate(articles)])
    except Exception as e:
        return f"Error fetching news: {str(e)}"

@tool
def get_weather(params: str) -> str:
    """Fetches current weather for a city. Params should be a JSON string with 'city' and optional 'units'."""
    try:
        inputs = json.loads(params)
        city = inputs.get("city", "")
        units = inputs.get("units", "metric")
    except json.JSONDecodeError:
        return "Invalid input. Provide a JSON string with 'city' and optional 'units'."

    api_key = os.getenv("OPENWEATHER_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    query_params = {"q": city.strip(), "appid": api_key, "units": units}
    try:
        response = requests.get(base_url, params=query_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The weather in {city} is {temperature}°C with {description}."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def translate_text(params: str) -> str:
    """Translates text using OpenAI. Params should be a JSON string with 'text' and 'target_lang'."""
    try:
        inputs = json.loads(params)
        text = inputs.get("text", "")
        target_lang = inputs.get("target_lang", "en")
    except json.JSONDecodeError:
        return "Invalid input. Provide a JSON string with 'text' and 'target_lang'."

    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        response = llm.predict(f"Translate this text to {target_lang}: {text}")
        return response
    except Exception as e:
        return f"Error translating text: {str(e)}"

@tool
def get_commodity_prices(params: str) -> str:
    """Fetches commodity prices using Yahoo Finance. Params should be a JSON string with 'commodities_list'."""
    try:
        inputs = json.loads(params)
        commodities_list = inputs.get("commodities_list", [])
    except json.JSONDecodeError:
        return "Invalid input. Provide a JSON string with 'commodities_list'."

    available_commodities = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Copper': 'HG=F',
        'Crude Oil': 'CL=F',
    }
    results = {}
    for commodity in commodities_list:
        symbol = available_commodities.get(commodity)
        if not symbol:
            results[commodity] = "Not supported"
            continue
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            current_price = data['Close'].iloc[-1]
            results[commodity] = f"${current_price:.2f}"
        except Exception as e:
            results[commodity] = f"Error: {str(e)}"
    return "\n".join([f"{key}: {value}" for key, value in results.items()])

# Initialize LangChain components
tools = [
    fetch_news,
    get_weather,
    translate_text,
    get_commodity_prices,
]

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Add memory to the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",  # Zero-shot agent
    memory=memory,
    verbose=True
)

# Chatbot loop
def chatbot():
    print("Welcome to the LangChain Chatbot!")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            response = agent.run(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
