import os

from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

# --- Tools -----------------------------------------------------------------


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city name, e.g. 'Cairo' or 'New York'."""
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return f"Could not find location: {city}"

        location = results[0]
        lat, lon = location["latitude"], location["longitude"]
        label = f"{location['name']}, {location.get('country', '')}".strip(", ")

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json()["current_weather"]

        return (
            f"Weather in {label}: {current['temperature']} C, "
            f"wind {current['windspeed']} km/h, "
            f"observed at {current['time']}"
        )
    except Exception as exc:
        return f"Error fetching weather: {exc}"

@tool
def tavily_search_tool(query: str) -> str:
    """Search the web for the query and return a concise summary of the results."""
    try:
        tavaly = TavilySearch(max_results=3)
        results = tavaly.invoke(query)
        if not results or not results.get("results"):
            return "No results found."
        summary = "\n".join(
            f"- {res.get('title', 'No title')} (score: {res.get('score', 'N/A')}): {res.get('content', 'No content')}"
            for res in results["results"]
        )
        return f"Search results for '{query}':\n{summary}"
    except Exception as exc:
        return f"Error performing search: {exc}"
TOOLS = [add, subtract, multiply, divide, get_weather, tavily_search_tool]

# --- Agent -------------------------------------------------------------------


def create_tool_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))
    return create_agent(
        model=llm,
        system_prompt=("""
                       You are a helpful assistant that can perform calculations, 
                       fetch weather information, 
                       and search the web to answer user queries. Use the provided tools to assist with user requests. Always choose the most appropriate tool based on the user's input, and provide clear and concise responses.
                       tools:
                       - add(a: float, b: float) -> float: Add two numbers together.
                       - subtract(a: float, b: float) -> float: Subtract b from a.
                       - multiply(a: float, b: float) -> float: Multiply two numbers together.
                       - divide(a: float, b: float) -> float: Divide a by b.
                        - get_weather(city: str) -> str: Get the current weather for a city name, e.g. 'Cairo' or 'New York'.
                        - tavily_search_tool(query: str) -> str: Search the web for the query and return a concise summary of the results.
                       """),
          tools=TOOLS)


# Module-level graph so `langgraph dev` / LangGraph Studio can load it directly.
graph = create_tool_agent()
