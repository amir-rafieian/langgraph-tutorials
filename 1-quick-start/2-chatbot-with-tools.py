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


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# =============================================================================
# Load env variables
# =============================================================================
load_dotenv()

if 'OPENAI_API_KEY' not in os.environ.keys():
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key:")

if 'TAVILY_API_KEY' not in os.environ.keys():
    os.environ["TAVILY_API_KEY"] = getpass("OpenAI API Key:")
    

# =============================================================================
# The LLM model
# =============================================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = ChatAnthropic(model="claude-3-haiku-20240307")


# =============================================================================
# Classes for the graph
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
    res = llm.invoke(state['messages'])
    return {'messages': [res]}




# =============================================================================
# Define the graph
# =============================================================================
graph_builder = StateGraph(State)
graph_builder.add_node(node='chatbot', action=chatbot)  # unique node name & its function or object

graph_builder.add_edge(start_key=START, end_key='chatbot')  # add the entry point
graph_builder.add_edge(start_key='chatbot', end_key=END)  # set the finish point


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
                print('Assistant: ', value['messages'][-1].content)
                


