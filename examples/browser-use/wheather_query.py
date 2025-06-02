"""
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" 

If you use Azure;

export AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY" 
export AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
export AZURE_OPENAI_MODEL_NAME="YOUR_AZURE_OPENAI_MODEL_NAME"
export AZURE_OPENAI_DEPLOYMENT="YOUR_AZURE_OPENAI_DEPLOYMENT"
export AZURE_OPENAI_API_VERSION="YOUR_AZURE_OPENAI_API_VERSION"

# 📰 Stocks News Scraper
AI-powered agent to scrape and summarize the latest stock market news.

## 🚀 Features
- Fetches the latest stock-related news
- Extracts headlines, sources, and article links

## 🔧 Installation
```bash
pip install upsonic browser-use playwright
```
"""
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from upsonic import Agent, Task, ObjectResponse
from upsonic.client.tools import BrowserUse

# Initialize the AI agent
news_agent = Agent("Weather Query", model="openai/qwen-turbo-latest", reflection=True) # If using Azure, set model="azure/gpt-4o"


class Weather(ObjectResponse):
    temperature: str
    conditions: str
    location: str
    humidity: str
    wind_speed: str
    precipitation: str
    pressure: str
    visibility: str
    uv_index: str
    sunrise: str
    sunset: str


# Task to search for the latest stock news using Browser Use
news_task = Task(
    "go to bing.com search the wheather of nanjing todyay",
    tools=[BrowserUse],
    response_format=Weather,
)
news_agent.print_do(news_task)
