# Logistics Order Extractor (LLM-Powered)

A logistics tool that utilizes a local Large Language Model (`llama3` via Ollama) to parse unstructured, messy order text and convert it into a clean, structured delivery plan using Python and Pandas.

# The tool was tested using the following messy logistics text input:

> *"3x pipes to site B asap; deliver gloves (2 boxes) + 1 helmet to A tomorrow am; URGENT cement to site D; pickup empty pallets from C; H needs 5 vests by end of day"*

## Key Functions
- **Text Parsing:** Extracts data from conversational text using `Ollama`.
- **Data Structuring:** Converts responses safely into JSON and loads them into a `Pandas DataFrame`.
- **Priority Sorting:** Sorts deliveries automatically by urgency weights.

## What the AI Was Asked to Help Build
the AI was asked to write the prompt with strict constraints:
- Convert the raw text into a raw JSON array.
- Avoid summarizing or omitting any items, quantities, or specific site letters.
- Split multi-item sentences for the same stop into separate, individual rows.
- Map and normalize time-based urgency phrases into standard priority tiers (`urgent`, `high`, `medium`, `low`).

## What I Changed, Fixed, or Added Myself
- **Robust JSON Regex Parsing:** Added an advanced extraction function (`extract_json_from_response`) utilizing Python’s `re` module to isolate JSON blocks and fallback formats from the raw LLM output.
- **Dynamic Priority Scoring:** Designed a dictionary-based numerical scoring system (`PRIORITY_SCORE`) to  map text priorities to weights, enabling reliable Pandas sorting.
