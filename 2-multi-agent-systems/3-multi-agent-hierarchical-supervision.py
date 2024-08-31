# =============================================================================
# Hierarchical Agent Teams
# =============================================================================
# In 2-multi-agent-supervision, We had a single supervisor node to route work between different worker nodes.
# But what if the job for a single worker becomes too complex? What if the number of workers becomes too large?
# For some applications, the system may be more effective if work is distributed hierarchically.

# We can do this by composing different subgraphs and creating a top-level supervisor,
# along with mid-level supervisors.

# =============================================================================
# Import Libraries
# =============================================================================
import getpass
import os
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Annotated, Sequence, TypedDict, Literal

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

