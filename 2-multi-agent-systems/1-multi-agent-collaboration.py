# =============================================================================
# Import Libraries
# =============================================================================
import getpass
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Annotated, Sequence, TypedDict


import operator

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from  langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# =============================================================================
# Load env variables
# =============================================================================
load_dotenv()

def _set_if_undefined(var:str):
    if var not in os.environ.keys():
        os.environ[var] = getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# =============================================================================
# Create agents
# =============================================================================
# this helper functions will create agents. These agents will then be nodes in
# the graph.

def create_agent(llm, tools, system_message: str):
    """Create an agent"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful AI assistant, collaborating with other assistants."
             " Use the provided tools to progress towards answering the question."
             " If you are unable to fully answer, that's OK, another assistant with different tools "
             " will help where you left off. Execute what you can to make progress."
             " If you or any of the other assistants have the final answer or deliverable,"
             " prefix your response with FINAL ANSWER so the team knows to stop."
             " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names = ", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


# =============================================================================
# Define Tools
# =============================================================================
# Search tool
tavily_tool = TavilySearchResults(max_results=5)


# python code executer tool
# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()

@tool
def python_repl(code: Annotated[str, "The python code to execute to generate the chart."]):
    """Use this to execute python code. If you want to see the output of a variable, you
    should print it out with print(...). This is visible to the user.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return "Failed to execute. error: {}".format(repr(e))
    
    result_str = "Successfully executed:\n```python\n{}\n```\nStdout: {}".format(code, result)
    return (result_str+"\n\nIf you have completed all tasks, respond with FINAL ANSWER.")


# =============================================================================
# =============================================================================
# # Create Graph
# =============================================================================
# =============================================================================
# Now that we've defined our tools and made some helper functions, will create the 
# individual agents below and tell them how to talk to each other using LangGraph.

# =============================================================================
# Define State
# =============================================================================
# We first define the state of the graph. This will just a list of messages,
# along with a key to track the most recent sender

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# =============================================================================
# Define Agent Nodes
# =============================================================================















































































