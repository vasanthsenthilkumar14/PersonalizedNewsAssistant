# Personalized News Assistant

An interactive, terminal-based assistant that fetches and summarizes the latest news, translates responses, shows current weather, and retrieves commodity prices. It uses OpenAI for summarization/translation, NewsAPI for news, OpenWeatherMap for weather, and Yahoo Finance for commodities. An optional LangChain-powered agent is also included.

## Features
- News: fetch top articles for any topic and get concise summaries
- Translation: translate responses into many languages
- Weather: current conditions for any city (metric/imperial)
- Commodities: prices for Gold, Silver, Copper, Platinum, Palladium, Crude Oil, Brent Crude, and Natural Gas
- Moderation: basic content moderation on inputs/outputs
- Optional: LangChain agent variant (`news_agent.py`)

## Project Structure
```
.
├── news_chatbot.py                # Main interactive chatbot (OpenAI + APIs)
├── news_agent.py                  # Optional LangChain-based agent
├── config.yaml                    # API keys (do NOT commit real secrets)
├── requirements.txt               # Core dependencies
├── command_line_interface_test.txt# Sample interactive transcript
├── toRun.md                       # Minimal run notes
└── README.md                      # This file
```

## Prerequisites
- Python 3.9+ (tested with 3.9)
- Accounts/API keys for:
  - OpenAI API
  - NewsAPI
  - OpenWeatherMap

## Installation
You can use Conda or venv. Example with Conda:

```bash
conda create -n news_bot python=3.9 -y
conda activate news_bot
pip install -r requirements.txt
```

Optional (only if you plan to run `news_agent.py`):

```bash
pip install langchain langchain-community
```

## Configuration
Create or update `config.yaml` in the project root with your keys. Example:

```yaml
OPENAI_API_KEY: "sk-..."
NEWSAPI_API_KEY: "your_newsapi_key"
OPENWEATHER_API_KEY: "your_openweather_key"
```

Notes:
- Do not commit real keys to GitHub. Use placeholders locally and add `config.yaml` to your `.gitignore` in your own fork if needed.
- The app reads keys from `config.yaml` and sets the corresponding environment variables at runtime.
- Alternatively, you may export the variables yourself before running:

```bash
export OPENAI_API_KEY=sk-...
export NEWSAPI_API_KEY=...
export OPENWEATHER_API_KEY=...
```

## Usage (Recommended: `news_chatbot.py`)
Run the interactive chatbot:

```bash
python news_chatbot.py
```

You will see a Welcome message and can type commands or natural-language queries. Core commands:
- `help`: show available features and examples
- `trending`: fetch current trending topics
- `exit`: quit the chatbot

Examples you can try:
- "What's the latest news about AI?"
- "Show me the weather in Tokyo"
- "Get gold and silver prices"
- "Translate the last response to Spanish"
- "Show me 5 articles about tech"
- "Summarize article 3"

The transcript in `command_line_interface_test.txt` shows a full sample session.

## Optional: LangChain Agent (`news_agent.py`)
There is an alternative agent-powered interface using LangChain tools. To use it, first install extras:

```bash
pip install langchain langchain-community
```

Then run:

```bash
python news_agent.py
```

This variant supports similar capabilities (news, translation, weather, commodities) via LangChain tools and memory.

## Configuration Details and APIs
- OpenAI: Used for summarization, translation, and content moderation. The default model in `news_chatbot.py` is `gpt-4-0613`. You may change the model if needed.
- NewsAPI: Used to fetch articles for arbitrary topics (free tiers have limits).
- OpenWeatherMap: Used for current weather by city.
- Yahoo Finance (`yfinance`): Used for commodities pricing.

## Troubleshooting
- Missing API key errors:
  - Ensure `config.yaml` contains `OPENAI_API_KEY`, `NEWSAPI_API_KEY`, and `OPENWEATHER_API_KEY` with valid values, or export them as environment variables before running.
- News fetch returns 401/429:
  - Check NewsAPI key validity and quota limits; free tier limitations may apply.
- Weather fetch errors:
  - Verify your `OPENWEATHER_API_KEY` and city spelling; try `units="metric"` or `units="imperial"` explicitly.
- Commodities show no data:
  - This can happen if Yahoo Finance temporarily fails or the symbol has no recent data. Try again later.
- Model access errors:
  - The default `gpt-4-0613` requires appropriate access on your OpenAI account. You can change the model string in the code if needed.
- LangChain agent import errors:
  - Install the optional dependencies: `pip install langchain langchain-community`.

## Development Notes
- Main entrypoint: `news_chatbot.py` (function `chatbot()`)
- Helper functions include: `fetch_news`, `summarize_article`, `translate_text`, `get_trending_topics`, `get_commodity_prices`, `get_weather`, and a function registry passed to OpenAI tool-calling.
- Debug helpers: `test_translate()`, `test_weather()` are available but commented out in `__main__`.

## License
Add your preferred license here before publishing to GitHub (e.g., MIT). Make sure your `config.yaml` with real keys is not committed.

## Acknowledgements
- OpenAI API
- NewsAPI
- OpenWeatherMap
- Yahoo Finance (`yfinance`)
