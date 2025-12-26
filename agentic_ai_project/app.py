
# Tools: Arxiv, Wikipedia, Tavily

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Configure tool wrappers (limit results & content length for speed)
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arXiv papers")
print(arxiv.name) 

arxiv.invoke("Attention is all you need")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
print(wiki.name)  

# Env setup for Tavily & Groq
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") or ""
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""

# Tavily search tool (news / web search)
from langchain_community.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults()

# Quick dry-run call 
tavily.invoke("Provide me the recent AI news")

# Combine all tools in a list; order doesn’t matter
tools = [arxiv, wiki, tavily]


from langchain_groq import ChatGroq

llm = ChatGroq(model_name="qwen-qwq-32b")

# Bind tools so the LLM can decide to call them via function/tool calls
llm_with_tools = llm.bind_tools(tools=tools)

# Try a couple of queries where different tools may be selected
llm_with_tools.invoke("What is the latest research on quantum computing?")
llm_with_tools.invoke("What is machine learning?")


# LangGraph workflow
# State schema: messages list (AnyMessage) with add_messages reducer
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    # messages accumulates over the graph; reducer merges turns
    messages: Annotated[list[AnyMessage], add_messages]


from IPython.display import Image, display  # Jupyter-friendly display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Node: call LLM (which may or may not request a tool)
def tool_calling_llm(state: State):
    """
    Given the current conversation state, call the LLM.
    If the LLM returns a tool call, tools_condition will route to ToolNode.
    Otherwise, we will route to END.
    """
    # Pass the messages history directly to the LLM
    ai_msg = llm_with_tools.invoke(state["messages"])
    # Return an appended assistant message (reducer will add it to state)
    return {"messages": [ai_msg]}

# Build graph
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)  # LLM node
builder.add_node("tools", ToolNode(tools))              # Runs the requested tool

# Edges: START → LLM; conditional → Tools or END; Tools → END
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If assistant's latest message contains a tool call → go to 'tools'
    # Otherwise → go to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile executable graph
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
messages = graph.invoke({"messages":"Hi give me latest ai news"})

# Print outputs
for m in messages["messages"]:
    try:
        m.pretty_print()  # nicely formatted (if available)
    except Exception:
        print(getattr(m, "content", m))

