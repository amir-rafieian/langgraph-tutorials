# Langgraph Tutorials

This repo is contains samples codes & projects based on Langgraph official documents.

## Directories
+ **quick-start:** A chatbot graph with tool & memory
+ **multi-agent-systems:** Multi-agent Collaboration/Supervision/Hierarchical Teams



## Environment Setup
Create a Python 3.11.9 Environment:
   ```bash
   $conda create -n <env_name> python=3.11.9
   ```

Activate the environment:
```bash
$conda activate <env_name>
```

If you are using Vertex AI Virtual Machines, you need to install ipykernel and add user to the environment:
```bash
$conda install ipykernel
$python -m ipykernel install --user --name=<env_name>
```

Clone the project repository and install the required packages:
```bash
$pip install -r requirements.txt
```

