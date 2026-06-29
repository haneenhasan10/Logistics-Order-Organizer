import json
import re
import subprocess
import pandas as pd

MODEL_NAME = "llama3:latest"

INPUT_TEXT = (
    "3x pipes to site B asap; deliver gloves (2 boxes) + 1 helmet to A tomorrow am; "
    "URGENT cement to site D; pickup empty pallets from C; H needs 5 vests by end of day"
)


PRIORITY_SCORE = {
    "urgent": 100,
    "high": 80,
    "medium": 60,
    "low": 30,
}


def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Ollama run failed.")

    return result.stdout.strip()


def extract_json_from_response(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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
- stop
- item
- quantity
- priority
- when

Normalize priority to one of:
- urgent
- high
- medium
- low

Rules:
- "urgent" or "asap" => urgent
- "by end of day" => high
- "tomorrow am" => medium
- pickup with no urgency => low

Messy text:
{order_text}
""".strip()


def extract_jobs(order_text: str):
    prompt = build_prompt(order_text)
    raw = call_ollama(prompt)
    jobs = extract_json_from_response(raw)

    if not isinstance(jobs, list):
        raise ValueError("Expected a JSON array.")

    cleaned = []
    for job in jobs:
        stop = str(job.get("stop", "")).strip().replace("site ", "").replace("Site ", "").upper()
        item = str(job.get("item", "")).strip()
        quantity = str(job.get("quantity", "unspecified")).strip()
        priority = str(job.get("priority", "low")).strip().lower()
        when = str(job.get("when", "unspecified")).strip().lower()

        if stop not in {"A", "B", "C", "D", "H"}:
            continue

        if priority not in {"urgent", "high", "medium", "low"}:
            priority = "low"

        if not item:
            continue

        cleaned.append({
            "Stop": stop,
            "Item": item,
            "Quantity": quantity if quantity else "unspecified",
            "Priority": priority,
            "When": when if when else "unspecified",
            "Score": PRIORITY_SCORE[priority]
        })

    return cleaned


def main():
    jobs = extract_jobs(INPUT_TEXT)

    df = pd.DataFrame(jobs)
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    display_df = df[["Stop", "Item", "Quantity", "Priority", "When"]]

    print("\nMessy text converted into structured data:\n")
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
