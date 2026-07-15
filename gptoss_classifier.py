import ollama
import csv
import json
import time
import os
import glob

from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CONFIG
# =========================================================

INPUT_FOLDER = "..."
OUTPUT_FOLDER = "..."

MODEL = "gpt-oss:20b"

MAX_WORKERS = 2
BATCH_SIZE = 40
MAX_RETRIES = 3
RETRY_DELAY = 2

TEMPERATURE = 0

PRINT_LIVE = False

TEXT_COLUMN = "text"

# =========================================================
# TAXONOMY
# =========================================================

TAXONOMY_PROMPT = """
PARENT: A Procedural Grievances

A1 Political/Legal Procedural Injustice
Definition: Perception that political or legal systems operate in a biased, unethical, or manipulated manner.

A2 Institutional Procedural Injustice
Definition: Perception that formal institutions apply rules unfairly or inconsistently.

A3 Organizational Grievances
Definition: Perception of unfairness within workplaces or organizations.

A4 Customer/Service-Related Grievances
Definition: Perception of unfair or inadequate service delivery.


PARENT: B Relational and Interpersonal Grievances

B1 Interpersonal Victimization & Betrayal
Definition: Harmful or exploitative behavior within personal relationships.

B2 Social Exclusion & Disconnection
Definition: Experiences of being socially excluded, ignored, or isolated.

B3 Status Threats and Symbolic Marginalization
Definition: Perceived lowering of social status or identity-based disrespect.

B4 Relational Free-Riding and Cooperation Failure
Definition: Perceived unfairness in reciprocity or shared effort.


PARENT: C Distributional Grievances

C1 Economic Inequality and Insecurity
Definition: Financial hardship, poverty, economic precarity, or class inequality.

C2 Workplace Distributional Injustice
Definition: Perceived unfair distribution of pay, promotion, or labor rewards.

C3 Service and Social Distributional Grievances
Definition: Unequal access to healthcare, education, welfare, or essential services.


PARENT: D Structural and Systemic Grievances

D1 Identity-Based Discrimination
Definition: Unequal treatment based on race, gender, ethnicity, age, or identity.

D2 Institutional Betrayal and Neglect
Definition: Institutions failing to protect or support individuals.

D3 Systemic Exclusion and Legal Barriers
Definition: Structural or legal systems restricting participation or access.

D4 Historical and Generational Grievances
Definition: Historical, inherited, or intergenerational injustice.


PARENT: E Geographically-rooted Grievances

E1 Environmental Inequality
Definition: Unequal exposure to environmental harm or risk.

E2 Territorial Marginalization
Definition: Geographic or regional exclusion and neglect.


PARENT: F Personal and Intimate Grievances

F1 Intimate Life Disruption
Definition: Major disruption in personal or family life.

F2 Life Transitions and Missed Expectations
Definition: Regret, burnout, disappointment, or unmet life expectations.

F3 Social Identity and Role Pressure
Definition: Stress arising from social or family role expectations.

F4 Self-perceived Shortcomings
Definition: Shame, inadequacy, low self-worth, or self-blame.

F5 Generational Disconnection and Challenges
Definition: Conflict or misunderstanding between generations.
"""

# =========================================================
# VALID LABELS
# =========================================================

VALID_SUBCATEGORIES = {
    "A1 Political/Legal Procedural Injustice",
    "A2 Institutional Procedural Injustice",
    "A3 Organizational Grievances",
    "A4 Customer/Service-Related Grievances",

    "B1 Interpersonal Victimization & Betrayal",
    "B2 Social Exclusion & Disconnection",
    "B3 Status Threats and Symbolic Marginalization",
    "B4 Relational Free-Riding and Cooperation Failure",

    "C1 Economic Inequality and Insecurity",
    "C2 Workplace Distributional Injustice",
    "C3 Service and Social Distributional Grievances",

    "D1 Identity-Based Discrimination",
    "D2 Institutional Betrayal and Neglect",
    "D3 Systemic Exclusion and Legal Barriers",
    "D4 Historical and Generational Grievances",

    "E1 Environmental Inequality",
    "E2 Territorial Marginalization",

    "F1 Intimate Life Disruption",
    "F2 Life Transitions and Missed Expectations",
    "F3 Social Identity and Role Pressure",
    "F4 Self-perceived Shortcomings",
    "F5 Generational Disconnection and Challenges",

    "None"
}

# =========================================================
# HELPERS
# =========================================================

def safe_json_load(text):

    try:
        return json.loads(text)

    except Exception:

        try:

            start = text.find("[")

            end = text.rfind("]") + 1

            if start != -1 and end != -1:

                return json.loads(
                    text[start:end]
                )

        except Exception:
            pass


        try:

            start = text.find("{")

            end = text.rfind("}") + 1

            if start != -1 and end != -1:

                return json.loads(
                    text[start:end]
                )

        except Exception:
            pass


    return None

def normalize_labels(labels):

    if labels is None:
        return ["None"]

    if isinstance(labels, str):
        labels = [labels]

    cleaned = []

    for label in labels:

        label = str(label).strip()

        # exact match
        if label in VALID_SUBCATEGORIES:
            cleaned.append(label)
            continue

        # shorthand like "C1"
        for valid in VALID_SUBCATEGORIES:
            if valid.startswith(label + " "):
                cleaned.append(valid)
                break

    return sorted(list(set(cleaned))) if cleaned else ["None"]


def infer_categories(subcategories):

    if subcategories == ["None"]:
        return ["None"]

    return sorted(list(set(
        sub[0] for sub in subcategories
    )))

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = f"""
You are a strict multi-label grievance classifier.

The taxonomy below is the complete and only allowed classification scheme.
Never create new labels.
Never rename labels.
Never classify outside this taxonomy.

TASK:
Assign ALL applicable grievance subcategories to the SINGLE comment provided.

IMPORTANT:
- You MUST ONLY use labels from the approved label list.
- NEVER invent labels.
- NEVER summarize themes.
- NEVER create new categories.
- NEVER output labels such as:
  "economic discourse"
  "inequality commentary"
  "moral judgment"

A grievance may be:
- explicit
- implicit
- ideological
- indirect
- hypothetical

However:
You MUST map everything ONLY onto existing taxonomy labels.

If no grievance applies:
return ["None"]

VALID LABELS:

{chr(10).join(sorted(VALID_SUBCATEGORIES))}

TAXONOMY:

{TAXONOMY_PROMPT}

OUTPUT FORMAT (JSON ONLY, no preamble, no markdown):

{{
  "subcategory": [
    "C1 Economic Inequality and Insecurity"
  ],
  "confidence": 0.93
}}
"""

# =========================================================
# IO
# =========================================================

def read_comments(file_path):

    comments = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            return comments

        # case-insensitive column lookup
        column_map = {
            col.strip().lower(): col
            for col in reader.fieldnames
            if col is not None
        }

        if TEXT_COLUMN.lower() not in column_map:

            raise ValueError(
                f"Could not find a '{TEXT_COLUMN}' column in {file_path}. "
                f"Columns found: {reader.fieldnames}"
            )

        text_column = column_map[TEXT_COLUMN.lower()]

        for row in reader:

            value = row.get(text_column)

            if value is None:
                continue

            text = str(value).strip()

            if text:

                comments.append(
                    {
                        "text": text,
                        "source": "text"
                    }
                )

    return comments
# =========================================================
# PROMPT
# =========================================================

def build_batch_prompt(items, context):

    texts = [
        {
            "id": i,
            "text": item["text"]
        }
        for i, item in enumerate(items)
    ]

    return f"""
Dataset context:
{context}


Classify each comment below according to the taxonomy provided in the system instructions.


COMMENTS:

{json.dumps(
    texts,
    ensure_ascii=False,
    indent=2
)}


Return JSON ONLY.

Required format:

[
  {{
    "id": 0,
    "subcategory": [
      "C1 Economic Inequality and Insecurity"
    ],
    "confidence": 0.93
  }}
]


Rules:
- Return exactly one object per comment.
- The id must correspond to the input id.
- Use ONLY labels from the approved taxonomy.
- Multi-label classification is allowed.
- If no grievance applies, return ["None"].
"""
# =========================================================
# MODEL
# =========================================================

def call_ollama(prompt):

    response = ollama.chat(

        model=MODEL,

        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":prompt
            }
        ],

        format="json",

        options={
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 6000
        }

    )

    return response["message"]["content"]
# =========================================================
# ROW WORKER
# =========================================================

def process_batch(batch_id, items, context):

    for attempt in range(MAX_RETRIES+1):

        try:

            prompt = build_batch_prompt(
                items,
                context
            )

            raw = call_ollama(prompt)

            data = safe_json_load(raw)

            if isinstance(data, dict):
                data = data.get("results", [])

            if not isinstance(data, list):
                print(
                    f"Batch {batch_id}: invalid JSON structure returned"
                )
                return []

            results = []


            for obj in data:

                idx = obj.get("id")

                if idx is None:
                    continue

                if idx >= len(items):
                    continue

                item = items[idx]


                subcategories = normalize_labels(
                    obj.get("subcategory")
                )


                categories = infer_categories(
                    subcategories
                )


                confidence = float(
                    obj.get(
                        "confidence",
                        0
                    )
                )


                results.append(
                    {
                    "row_id": batch_id*BATCH_SIZE+idx,
                    "text":item["text"],
                    "source":item["source"],
                    "categories":categories,
                    "subcategories":subcategories,
                    "confidence":confidence
                    }
                )


            return results


        except Exception as e:

            print(
                "Batch failed:",
                batch_id,
                e
            )

            time.sleep(RETRY_DELAY)


    return []
# =========================================================
# FILE PROCESSING
# =========================================================

def process_file(file_path):

    comments = read_comments(file_path)

    if not comments:
        print(f"No comments found in {file_path}")
        return []

    base_name = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    context = base_name.replace("_", " ")

    print("\n" + "#" * 80)
    print(f"PROCESSING: {base_name}")
    print(f"COMMENTS: {len(comments)}")
    print(f"WORKERS: {MAX_WORKERS}")
    print("#" * 80)

    all_results = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        batches = [
            comments[i:i + BATCH_SIZE]
            for i in range(
                0,
                len(comments),
                BATCH_SIZE
            )
        ]

        futures = [
            executor.submit(
                process_batch,
                batch_id,
                batch,
                context
            )
            for batch_id, batch in enumerate(batches)
        ]

        for i, future in enumerate(as_completed(futures), start=1):

            try:
                batch_results = future.result()

                all_results.extend(
                    batch_results
                )

                elapsed = time.time() - start_time

                rate = i / elapsed if elapsed > 0 else 0

                completed_rows = min(
                    i * BATCH_SIZE,
                    len(comments)
                )

                remaining = len(comments) - completed_rows

                eta = remaining / (completed_rows / elapsed)

                print(
                    f"\nCompleted {completed_rows}/{len(comments)} rows "
                    f"| ETA: {eta / 60:.1f} min"
                )

            except Exception as e:
                print("Future failed:", e)

    # keep original row order
    all_results.sort(key=lambda r: r["row_id"])

    # =====================================================
    # SAVE
    # =====================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"{base_name}_classified.csv"
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "categories",
                "subcategories",
                "confidence",
                "source",
                "row_id"
            ]
        )

        writer.writeheader()

        for row in all_results:

            writer.writerow({
                "text": row["text"],
                "categories": "; ".join(row["categories"]),
                "subcategories": "; ".join(row["subcategories"]),
                "confidence": row["confidence"],
                "source": row["source"],
                "row_id": row["row_id"]
            })

    print(f"\nSAVED → {output_path}")

    return all_results

# =========================================================
# MAIN
# =========================================================

def main():

    files = glob.glob(
        os.path.join(INPUT_FOLDER, "*.csv")
    )

    print(f"\nFound {len(files)} files")

    all_results = []

    total_start = time.time()

    for file_path in files:

        try:
            results = process_file(file_path)
            all_results.extend(results)

        except Exception as e:

            print(f"\nFILE FAILED: {file_path}")
            print(str(e))

    total_time = time.time() - total_start

    print("\n" + "#" * 80)
    print("DONE")
    print(f"TOTAL CLASSIFICATIONS: {len(all_results)}")
    print(f"TOTAL TIME: {total_time/60:.2f} minutes")
    print("#" * 80)

if __name__ == "__main__":
    main()
