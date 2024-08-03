# =============================================================================
# Work in progress...
# This script is not complete!
# =============================================================================

# =============================================================================
# Import Libraries
# =============================================================================
import os
from getpass import getpass 
from dotenv import load_dotenv
from IPython.display import Image, display

from typing import Annotated
from typing_extensions import TypedDict

from  langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic  # models from ChatOpenAI


from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.messages import BaseMessage

from langgraph.checkpoint.sqlite import SqliteSaver


from langchain_core.messages import AIMessage, ToolMessage

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

# for t in tools:
#     print(tool.name)  # tavily_search_results_json

# =============================================================================
# The Checkpoint Memory
# =============================================================================
memory = SqliteSaver.from_conn_string(":memory:")
# we've specified :memory: as the Sqlite DB path. it saves it all in-memory.
# In a production application we connect to our own DB and use other checkpoint
# classes.

# We need to provide a checkpointer when compiling the graph and a thread_id when 
# calling the graph, LangGraph automatically saves the state after each step. 
# When you invoke the graph again using the same thread_id, the graph loads its 
# saved state, allowing the chatbot to pick up where it left off.
config = {'configurable':{'thread_id':"1"}}

# =============================================================================
# The LLM model with Tool
# =============================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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


# Tools Node        
tool_node = ToolNode(tools=[tool])


# =============================================================================
# Define the graph
# =============================================================================
graph_builder = StateGraph(State)
graph_builder.add_node(node='chatbot', action=chatbot)  # unique node name & its function or object
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge('tools', 'chatbot')

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Notice that conditional edges start from a single node. This tells the 
# graph "any time the 'chatbot' node runs, either go to 'tools' if it calls 
# a tool, or end the loop if it responds directly.

graph_builder.set_entry_point("chatbot")
# or
# graph_builder.add_edge(START, "chatbot")

# =============================================================================
# Compile the graph
# =============================================================================
# specifying to interrupt_before the action node or after
graph = graph_builder.compile(checkpointer=memory,
                              interrupt_before=["tools"],
                              # Note: can also interrupt __after__ actions, if desired.
                              )


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
# Test
# =============================================================================
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
events = graph.stream({'messages': [("user", user_input)] }, config)
for event in events:
    for value in event.values():
        if isinstance(value["messages"][-1], BaseMessage):
            # print('Assistant: ', value['messages'][-1].content)
            value['messages'][-1].pretty_print()

snapshot = graph.get_state(config)
print('Next state: {}'.format(snapshot.next))  

existing_message = snapshot.values["messages"][-1]
print('existing message: {}'.format(existing_message))

print(existing_message.tool_calls[0]['id'])  # name, args, id


# =============================================================================
# Make the answer without tool calling
# =============================================================================
answer = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs."
)

new_messages = [
    # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
    ToolMessage(content=answer,tool_call_id = existing_message.tool_calls[0]['id']),
    # And then directly "put words in the LLM's mouth" by populating its response.
    AIMessage(content=answer)
    ]

new_messages[-1].pretty_print()

print("\n\nLast 2 messages before updating the state:")
print(graph.get_state(config).values["messages"][-2:])

graph.update_state(
    # Which state to update
    config,
    # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
    # to the existing state. We will review how to update existing messages in the next section!
    {'messages': new_messages}
    )

print("\n\nLast 2 messages after updating the state:")
print(graph.get_state(config).values["messages"][-2:])

graph.update_state(
    config,
    {"messages": [AIMessage(content="I'm an AI expert!")]},
    # Which node for this function to act as. It will automatically continue
    # processing as if this node just ran.
    as_node="chatbot",
)

print("\n\nLast 2 messages after updating the state:")
print(graph.get_state(config).values["messages"][-3:])
print(snapshot.next)


# =============================================================================
# Main Routine
# =============================================================================
if __name__ == '__main__':
    while True:
        user_input = input('User: ')
        if user_input.lower() in ['quit', 'exit', 'q']:
            print('Goodbye')
            break
        
        for event in graph.stream({'messages': ('user', user_input) }, config):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    # print('Assistant: ', value['messages'][-1].content)
                    value['messages'][-1].pretty_print()
                
                    
                    # Get a snapshot of the graph state
                    # The snapshot above contains the current state values, corresponding config,
                    # and the next node to process.
                    snapshot = graph.get_state(config)
                    print('Next state: {}'.format(snapshot.next))  
                    
                    existing_message = snapshot.values["messages"][-1]
                    print('existing message: {}'.format(existing_message))
                    
                    
                    
                    
                    
                    
                    
                    