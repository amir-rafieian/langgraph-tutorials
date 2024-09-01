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

from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser

import functools  # The functools module is for higher-order functions: functions that act on or return other functions.
import operator

from langgraph.graph import END, StateGraph, START
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
# Define a helper function below, which make it easier to add new agent worker nodes.
# instead of this function, we can use create_react_agent. refer to 3-multi-agent-hierarchical-supervision.py.
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
# We use function calling to choose the next worker node OR finish processing.
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed

members = ['Researcher', 'Coder']
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
    )

options = ['FINISH'] + members

#Parsing output using openai function calling:
function_def = {
    "name" : "route",
    "description": "Select the next role",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next":{
                "title": "Next",
                "anyOf":[{"enum": options}]
                }
            },
        "required":["next"]
        }
    }

prompt = ChatPromptTemplate.from_messages(
    [
         ("system", system_prompt),
         MessagesPlaceholder(variable_name="messages"),
         ("system",
          "Given the conversation above, who should act next?"
          "Or should we FINISH? Select one of: {options}"
         )
     ]
    )
prompt = prompt.partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model = 'gpt-4o-mini')

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
    )


# =============================================================================
# Construct Graph
# =============================================================================
# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next:str


research_agent = create_agent(llm=llm, tools=[tavily_tool], system_prompt="You are a web researcher.")
research_node = functools.partial(agent_node, agent=research_agent, name = "Researcher")


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION. PROCEED WITH CAUTION OR USE SANDBOX
code_agent = create_agent(llm=llm, tools=[python_repl_tool], system_prompt="You may generate safe python code to analyze data and generate charts using matplotlib or seaborn.")
code_node = functools.partial(agent_node, agent=code_agent, name = "Coder")

# Create the graph and add nodes
workflow = StateGraph(AgentState)
workflow.add_node('Researcher', research_node)
workflow.add_node('Coder', code_node)
workflow.add_node('Supervisor', supervisor_chain)

# Add the edges
for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(start_key=member, end_key='Supervisor')

# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges(source="Supervisor", path= lambda x: x['next'],path_map= conditional_map)

# Finally, add entrypoint
workflow.add_edge(START, "Supervisor")

graph = workflow.compile()


# =============================================================================
# Draw the graph
# =============================================================================
# print(graph.get_graph().draw_mermaid())
# we can copy the output of this print command, then go to mermaid.live and paste
# it there to visualize it.

# or generate the graph in ascii form:
graph.get_graph().print_ascii()


# Visualize the graph in the ipython console:
# try:
#     Image(graph.get_graph(xray=True).draw_mermaid_png())
# except:
#     # display(Image(graph.get_graph().draw_mermaid_png()))
#     # or it may need extra dependencies
#     pass

# =============================================================================
# Invoke-1
# =============================================================================
for s in graph.stream({"messages":[HumanMessage(content="Code hello world in python and print it to the terminal")]}):
    if "__end__" not in s:
        print(s)
        print("-----")



# =============================================================================
# Invoke-2
# =============================================================================
for s in graph.stream({"messages":[HumanMessage(content="Write a brief research report on pikas.")]},
                      {"recursion_limit":100}):
    if "__end__" not in s:
        print(s)
        print("-----")









































































