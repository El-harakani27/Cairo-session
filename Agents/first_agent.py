import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
import operator
load_dotenv()


class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    sentiment: str
    response_count: int


def create_conversation_graph():
    llm = ChatGroq(model= "openai/gpt-oss-120b",api_key = os.getenv('GROQ_API_KEY'))

    # Define node function
    def analyze_sentiment(state: ConversationState) -> dict:
        """Analyze the sentiment of the last message."""
        last_message = state["messages"][-1]

        response = llm.invoke(
            [
                SystemMessage(
                    content="Classify sentiment as: positive, negative, or neutral. Reply with just the word."
                ),
                HumanMessage(content=last_message),
            ]
        )

        return {"sentiment": response.content.lower().strip()}

    def generate_response(state: ConversationState) -> dict:
        """Generate appropriate response based on sentiment."""
        sentiment = state["sentiment"]
        last_message = state["messages"][-1]

        system_prompts = {
            "positive": "Respond enthusiastically and build on their positive energy.",
            "negative": "Respond empathetically and offer support.",
            "neutral": "Respond helpfully and informatively.",
        }

        prompt = system_prompts.get(sentiment, system_prompts["neutral"])

        response = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=last_message)]
        )

        return {"messages": [f"AI: {response.content}"], "response_count": 1}

    # Create graph
    graph = StateGraph(ConversationState)

    # Add nodes
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("generate_response", generate_response)

    # Add edges
    graph.add_edge(START, "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "generate_response")
    graph.add_edge("generate_response", END)

    app = graph.compile()

    return app



 
app = create_conversation_graph() 
    

    
