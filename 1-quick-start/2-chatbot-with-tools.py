# =============================================================================
# Import Libraries
# =============================================================================
import os
from getpass import getpass 
from dotenv import load_dotenv
import json
from IPython.display import Image, display

from typing import Annotated, Literal
from typing_extensions import TypedDict

from  langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic  # models from ChatOpenAI


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import ToolMessage, BaseMessage

# =============================================================================
# Load env variables
# =============================================================================
load_dotenv()

if 'OPENAI_API_KEY' not in os.environ.keys():
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")

if 'TAVILY_API_KEY' not in os.environ.keys():
    os.environ["TAVILY_API_KEY"] = getpass("OpenAI API Key:")
    

# =============================================================================
# The Tools
# =============================================================================
# Tavily
tool = TavilySearchResults(max_result=2)
tools = [tool]

# test Tavily
# tool.invoke('Whats a node in LangGraph?')
# output is a list of dictionaries with url & content as keys

for t in tools:
    print(tool.name)  # tavily_search_results_json


# =============================================================================
# The LLM model with Tool
# =============================================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatAnthropic(model="claude-3-haiku-20240307")

llm_with_tool = llm.bind_tools(tools)


# =============================================================================
# Define the state
# =============================================================================
class State(TypedDict):
    '''
    messgaes have the type "list".  the add_messages function in the annotation
    defines how this state key should be updated. In this case, it appends 
    messages to the list, rather than overwriting them.
    we've defined our State as a TypedDict with a single key: messages.
    The messages key is annotated with the add_messages function, which tells
    LangGraph to append new messages to the existing list, rather than overwriting it.
    Every node we define will receive the current State as input and return a 
    value that updates that state.
    '''
    messages: Annotated[list, add_messages]


# =============================================================================
# Define the nodes
# =============================================================================
# Every node we define will receive the current State as input and return a 
# value that updates that state.

def chatbot(state:State):
    '''
    It takes the current State as input and returns an updated messages list.
    '''
    res = llm_with_tool.invoke(state['messages'])
    return {'messages': [res]}


# BasicToolNode checks the most recent message in the state and calls tools
# if the message contains tool_calls. It relies on the LLM's tool_calling 
# support, which is available in Anthropic, OpenAI, Google Gemini,  etc.
# LangGraph's prebuilt ToolNode will do this. but for now we implement it 
# ourselves.
# The Walrus ':=' operator in this class which is introduced in python 3.8
# allows you to assign a value to a variable as part of an expression
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage"""
    
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name for tool in tools}
        
    def __call__(self, inputs: dict):
        if messages := inputs.get('messages', []):  # if the key 'messages' does not exists, it will return an empty list
            message = messages[-1]
        else:
            raise ValueError("No Message found in the input")
            
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
            
            return {'messages': outputs}
        
tool_node = BasicToolNode(tools=[tool])


# =============================================================================
# Define the Router Function
# =============================================================================
# route_tools checks for tool_calls in the chatbot's output.
# Provide this function to the graph by calling add_conditional_edges,
# which tells the graph that whenever the chatbot node completes to check this 
# function to see where to go next.
# The condition will route to tools if tool calls are present and "__end__" if not.
# function returns "tools" if the chatbot asks to use a tool, and "__end__" if
# it is fine directly responding. This conditional routing defines the main agent loop.
# Later, we will replace this with the prebuilt tools_condition to be more concise.

def route_tools(state: State) -> Literal['tools', '__end__']:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
        
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"





# =============================================================================
# Define the graph
# =============================================================================
graph_builder = StateGraph(State)
graph_builder.add_node(node='chatbot', action=chatbot)  # unique node name & its function or object
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(start_key=START, end_key='chatbot')  # add the entry point
graph_builder.add_edge('tools', 'chatbot')

# graph_builder.add_edge(start_key='chatbot', end_key=END)  # set the finish point

graph_builder.add_conditional_edges('chatbot', route_tools,
                                    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
                                    # It defaults to the identity function, but if you
                                    # want to use a node named something else apart from "tools",
                                    # You can update the value of the dictionary to something else
                                    # e.g., "tools": "my_tools"
                                    {'tools': 'tools', '__end__':'__end__'})

# Notice that conditional edges start from a single node. This tells the 
# graph "any time the 'chatbot' node runs, either go to 'tools' if it calls 
# a tool, or end the loop if it responds directly.


# =============================================================================
# Compile the graph
# =============================================================================
graph = graph_builder.compile()


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
#     Image(graph.get_graph().draw_mermaid_png())
# except:
#     # display(Image(graph.get_graph().draw_mermaid_png()))
#     # or it may need extra dependencies
#     pass


# =============================================================================
# Main Routine
# =============================================================================
if __name__ == '__main__':
    while True:
        user_input = input('User: ')
        if user_input.lower() in ['quit', 'exit', 'q']:
            print('Goodbye')
            break
        
        for event in graph.stream({'messages': ('user', user_input) }):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    print('Assistant: ', value['messages'][-1].content)
                


