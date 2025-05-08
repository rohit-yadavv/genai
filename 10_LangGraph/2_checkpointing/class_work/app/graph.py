from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition

# https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html

load_dotenv()

# os.getenv("OPENAI_API_KEY")
# https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-4-human-in-the-loop

@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human."""
    # this above comment is a description for an ai
    human_response = interrupt({"query": query}) # Graph will exit out after saving data in db
    return human_response["data"] # when support team give answer then this tool resume with the data.  the "data" includes human ne jo likha hai
    # this function call by ai

tools = [human_assistance_tool]

llm = init_chat_model(model_provider='openai', model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools=tools)


class State(TypedDict):
    messages: Annotated[list, add_messages] # Messages have the type "list". The `add_messages` function in the annotation defines how this state key should be updated (in this case, it appends messages to the list, rather than overwriting them)


# creating node. in langgraph we call chatbot() function as node and not function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    # return {"messages": [llm.invoke(state["messages"])]} OR 
    messages = state.get("messages")
    # response = llm.invoke(messages)
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")  # tool k output ko wapis se chatbot me daalo
graph_builder.add_edge("chatbot", END)

# without any memory
# graph = graph_builder.compile()

# * TOPIC: Checkpointing
# jab graph execute karna hai tab checkpointer pass kardo
# creates a new graph with given checkpointer and it has memory
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
