# AI-Powered Rule Generation for Automated Fault Detection and Diagnostics
This project lets you generate rules for rule-based fault detection and diagnostics systems in the context of building automation and control systems. 
The current project is set up to follow rules that are defined in the paper *Data Integrity Checks for Building Automation and Control Systems* by Gwerder et al. (https://doi.org/10.34641/clima.2022.271).
Some adjustments are necessary to use the system with your own rule-based fault detection.

## Installation
First, all necessary packages must be installed. This can be done by running the following commands:
```bash
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup
To be able to use this project, add PDF documents to the doc folder that describe your fault detection system.
Afterwards, run the RAG pipeline from inside the `src` folder:

```bash
python3 rag.py
```

As a next step, examples must be added to an examples directory. The examples must be in text form and should be existing examples of your rule-based fault detection system.

Finally, the Pydantic schema must be adjusted according to the rules of your fault detection system. This is done in the `few_shot_crew.py` file.

## Run
To run the project, execute from inside the `src` folder:
```bash
python3 main.py
```
