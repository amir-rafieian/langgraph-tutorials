# =============================================================================
# Agent Supervisor
# The collaboration example routed messages automatically based on the output of the initial 
# researcher agent.
# 
# We can also choose to use an LLM to orchestrate the different agents.
# Below, we will create an agent group, with an agent supervisor to help delegate tasks.
# =============================================================================

# =============================================================================
# Import Libraries
# =============================================================================
import getpass
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

from  langchain_openai import ChatOpenAI

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent

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
# Create Tools
# =============================================================================
# Tool1: Tavily Search
tavily_tool = TavilySearchResults(max_results=5)

# Tools2: Python Executer. This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()


# =============================================================================
# Helper Utilities
# =============================================================================
# Define a helper function below, which make it easier to add new agent worker nodes
def create_agent(llm:ChatOpenAI, tools:list, system_prompt:str):
    """
    Creates an agent using the specified language model, tools, and system prompt.

    This function initializes an agent by creating a prompt template with the system's instructions,
    and placeholders for incoming messages and the agent's scratchpad. It then constructs the agent
    using the provided language model and tools, wrapping it within an executor for execution.
    each worker node will be given a name and some tools.

    Args:
        llm (ChatOpenAI): The language model to be used by the agent.
        tools (list): A list of tools that the agent can use to perform its tasks.
        system_prompt (str): A string that defines the initial system prompt for the agent, 
                             setting the context or behavior for the conversation.

    Returns:
        AgentExecutor: An executor that manages the agent's operation, enabling it to process 
                       incoming messages and execute tasks with the provided tools.
    """
    # prompt = ChatPromptTemplate.from_messages(
    # [
    #     ("system", "You are a helpful assistant"),
    #     MessagesPlaceholder("chat_history", optional=True),
    #     ("human", "{input}"),
    #     MessagesPlaceholder("agent_scratchpad"),
    # ]
    # )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    
    agent = create_openai_tools_agent(llm, tools, prompt)   #create_react_agent(llm, tools, prompt) )
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


# Define a function to be used as nodes, and convert agent response to human message.
# we will add this human message to the global state of the graph
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': [HumanMessage(content=result['output'], name=name)]}

# =============================================================================
# Create Agent Supervisor
# =============================================================================




































