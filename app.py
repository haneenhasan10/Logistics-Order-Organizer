import json
import re
import subprocess
import pandas as pd
import streamlit as st

# Name of the local LLM deployed via Ollama
MODEL_NAME = "llama3:latest"

# Sample unstructured, conversational logistics text input
INPUT_TEXT = (
    "3x pipes to site B asap; deliver gloves (2 boxes) + 1 helmet to A tomorrow am; "
    "URGENT cement to site D; pickup empty pallets from C; H needs 5 vests by end of day"
)

# Mapping of normalized priority strings to numerical scores
PRIORITY_SCORE = {
    "urgent": 100,
    "high": 80,
    "medium": 60,
    "low": 30,
}


def call_ollama(prompt: str) -> str:
    """
    Executes a subprocess command to run the local Ollama model with the given prompt
    and retrieves the raw text response.
    """
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        capture_output=True,
        text=True
    )

    # Raise an exception if the Ollama command execution fails
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Ollama run failed.")

    return result.stdout.strip()


def extract_json_from_response(text: str):
    """
    Safely extracts and parses a valid JSON object or array from the LLM's response
    using strict try-except blocks and Regex fallbacks.
    """
    text = text.strip()

    # Attempt 1: Direct JSON parsing of the entire text response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract JSON from markdown code blocks (```json ... ```) via Regex
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        candidate = code_block.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"(\[.*\])", text, re.DOTALL)
    if array_match:
        candidate = array_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Raise an error if no valid JSON is found
    raise ValueError("Could not parse JSON from Ollama response.")


def build_prompt(order_text: str) -> str:
    return f"""
You are a deterministic logistics extractor.

Your job is to convert the Messy text into a complete JSON array.
Do not summarize.
Do not infer missing stops incorrectly.
Do not merge separate items into one row.
Do not omit any job.
Do not omit any quantity.
Do not change site letters.

Return JSON only.

The output must contain one row per job item.
If a sentence contains two items for the same stop, return two rows.

Use exactly these keys:
- stop: A, B, C, D or H. if letters not specified, write unspecified.
- item
- quantity: it is the number of items. if not specified, write 1.
- priority
- when: the temporal context, time frame,time abbreviation(like: asap) or deadline.prioity keywords, urgency keywords. It must only capture the time-related aspect and must never contain actions, verbs, or task types. if not specified, write unspecified.


Normalize priority to one of:
- urgent
- high
- medium
- low

Rules:
- "urgent" or "asap" => urgent
- "by end of day" => high
- "tomorrow am" => medium
- no urgency word => low

Messy text:
{order_text}
""".strip()


def extract_jobs(order_text: str):
    """
    Extracts logistics jobs using the LLM,
    validates the output,
    and converts it into clean structured data.
    """
    # build prompt 
    prompt = build_prompt(order_text)

    # send prompt to ollama
    raw = call_ollama(prompt)

    # parse returned JSON
    jobs = extract_json_from_response(raw)

    # Ensure the response is a JSON array
    if not isinstance(jobs, list):
        raise ValueError("Expected a JSON array.")

    cleaned = []

    # Process each extracted job
    for job in jobs:

        # normalize stop names
        stop = str(job.get("stop", "")).strip().replace("site ", "").replace("Site ", "").upper()
        
        # extracts fields 
        item = str(job.get("item", "")).strip()
        quantity = str(job.get("quantity", "unspecified")).strip()
        priority = str(job.get("priority", "low")).strip().lower()
        when = str(job.get("when", "unspecified")).strip().lower()

        # Skip invalid stops
        if stop not in {"A", "B", "C", "D", "H"}:
            continue

        # Default invalid priorities to low
        if priority not in {"urgent", "high", "medium", "low"}:
            priority = "low"

        # Ignore empty items
        if not item:
            continue

        # store cleaned record
        cleaned.append({
            "Stop": stop,
            "Item": item,
            "Quantity": quantity if quantity else "unspecified",
            "Priority": priority,
            "When": when if when else "unspecified",
            "Score": PRIORITY_SCORE[priority]
        })

    return cleaned


# Streamlit UI

st.set_page_config(
    page_title="Logistics Job Extractor",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Logistics Job Extractor")

st.write("Convert messy logistics text into structured jobs using Ollama.")

default_text = (
    "3x pipes to site B asap; deliver gloves (2 boxes) + 1 helmet to A tomorrow am; "
    "URGENT cement to site D; pickup empty pallets from C; "
    "H needs 5 vests by end of day"
)

user_text = st.text_area(
    "Enter logistics text:",
    value=default_text,
    height=180
)

if st.button("Extract Jobs", type="primary"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Extracting jobs..."):

            try:

                jobs = extract_jobs(user_text)

                if len(jobs) == 0:
                    st.warning("No jobs found.")

                else:

                    df = pd.DataFrame(jobs)

                    df = df.sort_values(
                        by="Score",
                        ascending=False
                    ).drop(columns=["Score"])

                    st.success(f"{len(df)} jobs extracted.")

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv = df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "⬇ Download CSV",
                        csv,
                        file_name="logistics_jobs.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(str(e))