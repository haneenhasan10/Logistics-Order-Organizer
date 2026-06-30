# Logistics Order Extractor (LLM-Powered)

A logistics tool that utilizes a local Large Language Model (`llama3` via Ollama) to parse unstructured, messy order text and convert it into a clean, structured delivery plan using Python and Pandas.

# The tool was tested using the following messy logistics text input:

> *"3x pipes to site B asap; deliver gloves (2 boxes) + 1 helmet to A tomorrow am; URGENT cement to site D; pickup empty pallets from C; H needs 5 vests by end of day"*

## Key Functions
- **Text Parsing:** Extracts data from conversational text using `Ollama`.
- **Data Structuring:** Converts responses safely into JSON and loads them into a `Pandas DataFrame`.
- **Priority Sorting:** Sorts deliveries automatically by urgency weights.

## The pipeline is as follows:
INPUT_TEXT -> build_prompt -> call_ollama -> extract_json -> clean data ->

DataFrame -> sort -> print

## What the AI Was Asked to Help Build
- the AI was asked to write the prompt with strict constraints
- Baseline Extraction

## What I Changed, Fixed, or Added Myself
- **Enhancing LLM prompt**
- **Dynamic Priority Scoring:** Designed a dictionary-based numerical scoring system (`PRIORITY_SCORE`) to  map text priorities to weights, enabling reliable Pandas sorting.
