#!/usr/bin/env python3
"""
Advanced Gemini Medical Q&A Pipeline
Merged Version

Features:
✔ Multiple API keys rotation
✔ Resume from last medicine
✔ Topic-aware extraction
✔ Medicine tag in output
✔ JSON recovery logic
✔ Confidence filtering
✔ Gemini 2.5 Flash support
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List

from google import genai


# =========================================================
# CONFIG
# =========================================================

BASE_FOLDER = r"C:/Users/Satya Sri/Desktop/gemini qa generation/medicines_data"

API_KEYS_FILE = "api_keys.txt"

OUTPUT_FOLDER = "C:/Users/Satya Sri/Desktop/gemini qa generation/Gemini_qa_pairs"

MODEL_NAME = "gemini-2.5-flash"

CONFIDENCE_THRESHOLD = 0.6

PROGRESS_FILE = "progress.json"

BATCH_DELAY = 7


# =========================================================
# TOPICS (from second code)
# =========================================================

COMPULSORY_TOPICS = [
    "constitution",
    "temperament_emotional_profile",
    "age_group",
    "gender_specific_symptoms",
    "central_pathology_theme",
    "infection_miasm",
    "prognosis_or_course",
    "modalities_better",
    "modalities_worse",
    "seasonal_variations",
    "time_aggravations",
    "discharges_character",
    "ulcer_character",
    "odour_characteristics",
    "bone_periosteum_pattern",
    "nervous_manifestations",
    "gastrointestinal_pattern",
    "female_reproductive_pattern",
    "male_reproductive_pattern",
    "respiratory_pattern",
    "cardiovascular_pattern",
    "key_mental_phrases",
    "emotional_state",
    "characteristic_pain_type",
    "unique_physical_signs",
    "therapeutic_indications",
    "treatment_modalities_suggestions",
    "medicine_potency_response",
    "complementary_remedies"
]


# =========================================================
# LOAD API KEYS
# =========================================================

def load_api_keys():

    if not os.path.exists(API_KEYS_FILE):
        raise FileNotFoundError(
            "api_keys.txt not found"
        )

    with open(API_KEYS_FILE) as f:

        keys = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(f"Loaded {len(keys)} API keys")

    return keys


# =========================================================
# PROGRESS SYSTEM
# =========================================================

def load_progress():

    if not os.path.exists(PROGRESS_FILE):

        return {
            "last_index": 0,
            "api_index": 0
        }

    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save_progress(index, api_index):

    data = {
        "last_index": index,
        "api_index": api_index
    }

    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# =========================================================
# GEMINI CLIENT
# =========================================================

class GeminiClient:

    def __init__(self, api_keys):

        self.api_keys = api_keys
        self.key_index = 0
        self.call_count = 0

        self.set_key()

    def set_key(self):

        key = self.api_keys[self.key_index]

        self.client = genai.Client(
            api_key=key
        )

        print(
            f"\nUsing API Key #{self.key_index + 1}"
        )

    def rotate_key(self):

        self.key_index += 1

        if self.key_index >= len(self.api_keys):

            raise RuntimeError(
                "ALL API KEYS USED — RESET AFTER 1 DAY"
            )

        print(
            f"\nSwitching to API key #{self.key_index + 1}"
        )

        self.set_key()

    def call_api(self, prompt):

        for attempt in range(3):

            try:

                self.call_count += 1

                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )

                text = response.text

                if not text:
                    raise ValueError(
                        "Empty response"
                    )

                return text

            except Exception as e:

                err = str(e).lower()

                if "quota" in err:

                    print(
                        "Quota exceeded → rotating key"
                    )

                    self.rotate_key()
                    continue

                print(
                    f"Retrying... ({attempt+1})"
                )

                time.sleep(5)

        return None


# =========================================================
# JSON RECOVERY
# =========================================================

def extract_json(text):

    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)

    start = text.find('{')
    end = text.rfind('}')

    if start == -1:
        return None

    try:
        return json.loads(
            text[start:end+1]
        )
    except:
        return None


# =========================================================
# PROMPT WITH TOPICS
# =========================================================

def build_prompt(text, medicine):

    topics_text = "\n".join(
        f"- {t}" for t in COMPULSORY_TOPICS
    )

    return f"""
Extract Question-Answer pairs from homeopathic remedy text.

MEDICINE:
{medicine}

TEXT:
{text}

Ensure coverage of these topics:
{topics_text}

Return STRICT JSON:

{{
 "qa_pairs":[
  {{
   "question":"...",
   "answer":"...",
   "excerpt":"...",
   "topic":"...",
   "confidence":0.9
  }}
 ]
}}

Rules:
- One Q&A per fact
- No invented information
- Confidence between 0.5–1.0
"""


# =========================================================
# FORMAT OUTPUT
# =========================================================

def format_triplet(
        medicine,
        question,
        answer,
        excerpt
):

    return {
        "text":
f"<medicine>\n{medicine}\n</medicine>\n\n"
f"<context>\n{excerpt}\n</context>\n\n"
f"<user>\n{question}\n</user>\n\n"
f"<assistant>\n"
f"<actual response>\n"
f"{answer}\n"
f"</actual response>\n"
f"</assistant>"
    }


# =========================================================
# MAIN PROCESS
# =========================================================

def process_all():

    api_keys = load_api_keys()

    progress = load_progress()

    client = GeminiClient(api_keys)

    client.key_index = progress["api_index"]
    client.set_key()

    base = Path(BASE_FOLDER)

    files = sorted(
        base.glob("*.txt")
    )

    start = progress["last_index"]

    print(
        f"\nResuming from medicine #{start + 1}"
    )

    output_dir = base / OUTPUT_FOLDER

    output_dir.mkdir(
        exist_ok=True
    )

    for idx in range(start, len(files)):

        file = files[idx]

        medicine = file.stem.replace(
            "-", " "
        ).title()

        print(
            f"\nProcessing: {medicine}"
        )

        text = file.read_text(
            encoding="utf-8"
        )

        text = re.sub(
            r'\s+',
            ' ',
            text
        ).strip()

        prompt = build_prompt(
            text,
            medicine
        )

        response = client.call_api(
            prompt
        )

        if not response:

            print("No response.")
            continue

        data = extract_json(
            response
        )

        if not data:

            print("JSON failed.")
            continue

        qa_pairs = data.get(
            "qa_pairs",
            []
        )

        triplets = []

        for qa in qa_pairs:

            if qa.get(
                "confidence",
                0.5
            ) >= CONFIDENCE_THRESHOLD:

                triplets.append(
                    format_triplet(
                        medicine,
                        qa["question"],
                        qa["answer"],
                        qa["excerpt"]
                    )
                )

        output_file = (
            output_dir /
            f"{file.stem}.jsonl"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            for t in triplets:

                f.write(
                    json.dumps(
                        t,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(
            f"Saved {len(triplets)} Q&A pairs"
        )

        save_progress(
            idx + 1,
            client.key_index
        )

        time.sleep(BATCH_DELAY)

    print("\nALL MEDICINES COMPLETED")

    print(
        f"Total API Calls: {client.call_count}"
    )


# =========================================================

if __name__ == "__main__":

    try:

        process_all()

    except RuntimeError as e:

        print("\n" + "="*50)
        print(str(e))
        print("All API keys used.")
        print("Resume tomorrow.")
        print("="*50)





# #!/usr/bin/env python3
# """
# Gemini Medical Q&A Pipeline
# NEW Gemini SDK Version (google.genai)

# Features:
# ✔ Reads API keys from api_keys.txt
# ✔ Rotates keys automatically
# ✔ Resume from last medicine
# ✔ Uses latest Gemini SDK
# ✔ Uses stable model (gemini-2.0-flash)
# """

# import os
# import re
# import json
# import time
# from pathlib import Path

# from google import genai   # ← NEW SDK


# # =========================================================
# # CONFIG — EDIT THESE IF NEEDED
# # =========================================================

# BASE_FOLDER = r"C:\Users\Satya Sri\Desktop\gemini qa generation\medicines_data"

# API_KEYS_FILE = "api_keys.txt"

# OUTPUT_FOLDER = "outputs"

# MODEL_NAME = "gemini-2.5-flash"   # ← WORKING MODEL

# CONFIDENCE_THRESHOLD = 0.6

# PROGRESS_FILE = "progress.json"

# BATCH_DELAY = 7   # safer for quota


# # =========================================================
# # LOAD API KEYS
# # =========================================================

# def load_api_keys():

#     if not os.path.exists(API_KEYS_FILE):
#         raise FileNotFoundError(
#             "api_keys.txt not found"
#         )

#     with open(API_KEYS_FILE) as f:

#         keys = [
#             line.strip()
#             for line in f
#             if line.strip()
#         ]

#     print(f"Loaded {len(keys)} API keys")

#     return keys


# # =========================================================
# # PROGRESS SYSTEM
# # =========================================================

# def load_progress():

#     if not os.path.exists(PROGRESS_FILE):

#         return {
#             "last_index": 0,
#             "api_index": 0
#         }

#     with open(PROGRESS_FILE) as f:
#         return json.load(f)


# def save_progress(index, api_index):

#     data = {
#         "last_index": index,
#         "api_index": api_index
#     }

#     with open(PROGRESS_FILE, "w") as f:
#         json.dump(data, f, indent=2)


# # =========================================================
# # GEMINI CLIENT (NEW SDK)
# # =========================================================

# class GeminiClient:

#     def __init__(self, api_keys):

#         self.api_keys = api_keys
#         self.key_index = 0
#         self.call_count = 0

#         self.set_key()

#     def set_key(self):

#         key = self.api_keys[self.key_index]

#         self.client = genai.Client(
#             api_key=key
#         )

#         print(
#             f"\nUsing API Key #{self.key_index + 1}"
#         )

#     def rotate_key(self):

#         self.key_index += 1

#         if self.key_index >= len(self.api_keys):

#             raise RuntimeError(
#                 "ALL API KEYS USED — RESET AFTER 1 DAY"
#             )

#         print(
#             f"\nSwitching to API key #{self.key_index + 1}"
#         )

#         self.set_key()

#     def call_api(self, prompt):

#         for attempt in range(3):

#             try:

#                 self.call_count += 1

#                 response = self.client.models.generate_content(
#                     model=MODEL_NAME,
#                     contents=prompt
#                 )

#                 text = response.text

#                 if not text:
#                     raise ValueError(
#                         "Empty response"
#                     )

#                 return text

#             except Exception as e:

#                 err = str(e).lower()

#                 if "quota" in err:

#                     print(
#                         "Quota exceeded → rotating key"
#                     )

#                     self.rotate_key()
#                     continue

#                 print(
#                     f"Retrying... ({attempt+1})"
#                 )

#                 time.sleep(5)

#         return None


# # =========================================================
# # JSON EXTRACTION
# # =========================================================

# def extract_json(text):

#     text = re.sub(r'```json', '', text)
#     text = re.sub(r'```', '', text)

#     start = text.find('{')
#     end = text.rfind('}')

#     if start == -1:
#         return None

#     try:
#         return json.loads(
#             text[start:end+1]
#         )
#     except:
#         return None


# # =========================================================
# # PROMPT
# # =========================================================

# def build_prompt(text, medicine):

#     return f"""
# Extract Q&A pairs from this remedy text.

# MEDICINE:
# {medicine}

# TEXT:
# {text}

# Return STRICT JSON:

# {{
#  "qa_pairs":[
#   {{
#    "question":"...",
#    "answer":"...",
#    "excerpt":"...",
#    "topic":"...",
#    "confidence":0.9
#   }}
#  ]
# }}
# """


# # =========================================================
# # FORMAT OUTPUT
# # =========================================================

# def format_triplet(
#         medicine,
#         question,
#         answer,
#         excerpt
# ):

#     return {
#         "text":
# f"<medicine>\n{medicine}\n</medicine>\n\n"
# f"<context>\n{excerpt}\n</context>\n\n"
# f"<user>\n{question}\n</user>\n\n"
# f"<assistant>\n"
# f"<actual response>\n"
# f"{answer}\n"
# f"</actual response>\n"
# f"</assistant>"
#     }


# # =========================================================
# # MAIN PROCESS
# # =========================================================

# def process_all():

#     api_keys = load_api_keys()

#     progress = load_progress()

#     client = GeminiClient(api_keys)

#     # Resume previous API index
#     client.key_index = progress["api_index"]
#     client.set_key()

#     base = Path(BASE_FOLDER)

#     files = sorted(
#         base.glob("*.txt")
#     )

#     start = progress["last_index"]

#     print(
#         f"\nResuming from medicine #{start + 1}"
#     )

#     output_dir = base / OUTPUT_FOLDER

#     output_dir.mkdir(
#         exist_ok=True
#     )

#     for idx in range(start, len(files)):

#         file = files[idx]

#         medicine = file.stem.replace(
#             "-", " "
#         ).title()

#         print(
#             f"\nProcessing: {medicine}"
#         )

#         text = file.read_text(
#             encoding="utf-8"
#         )

#         text = re.sub(
#             r'\s+',
#             ' ',
#             text
#         ).strip()

#         prompt = build_prompt(
#             text,
#             medicine
#         )

#         response = client.call_api(
#             prompt
#         )

#         if not response:

#             print("No response.")
#             continue

#         data = extract_json(
#             response
#         )

#         if not data:

#             print("JSON failed.")
#             continue

#         qa_pairs = data.get(
#             "qa_pairs",
#             []
#         )

#         triplets = []

#         for qa in qa_pairs:

#             if qa.get(
#                 "confidence",
#                 0.5
#             ) >= CONFIDENCE_THRESHOLD:

#                 triplets.append(
#                     format_triplet(
#                         medicine,
#                         qa["question"],
#                         qa["answer"],
#                         qa["excerpt"]
#                     )
#                 )

#         output_file = (
#             output_dir /
#             f"{file.stem}.jsonl"
#         )

#         with open(
#             output_file,
#             "w",
#             encoding="utf-8"
#         ) as f:

#             for t in triplets:

#                 f.write(
#                     json.dumps(
#                         t,
#                         ensure_ascii=False
#                     ) + "\n"
#                 )

#         print(
#             f"Saved {len(triplets)} Q&A pairs"
#         )

#         save_progress(
#             idx + 1,
#             client.key_index
#         )

#         time.sleep(BATCH_DELAY)

#     print("\nALL MEDICINES COMPLETED")

#     print(
#         f"Total API Calls: {client.call_count}"
#     )


# # =========================================================

# if __name__ == "__main__":

#     try:

#         process_all()

#     except RuntimeError as e:

#         print("\n" + "="*50)
#         print(str(e))
#         print("All API keys used.")
#         print("Resume tomorrow.")
#         print("="*50)
