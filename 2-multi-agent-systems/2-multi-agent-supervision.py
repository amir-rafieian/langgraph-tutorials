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

