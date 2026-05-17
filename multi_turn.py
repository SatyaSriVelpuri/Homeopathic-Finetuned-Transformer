# # #!/usr/bin/env python3
# # """
# # Homeopathy Multi-Turn Dialogue Generator
# # =========================================
# # - Reads all .txt files from medicines_data folder
# # - Generates multi-turn doctor-patient conversations using Gemini API
# # - Converts to {"messages": [...]} format for fine-tuning
# # - Rotates API keys automatically on quota exhaustion
# # - Resumes from last processed medicine via progress.json
# # - Output: multiturn.jsonl (single file, all medicines)
# # """

# # import os
# # import re
# # import json
# # import time
# # from pathlib import Path
# # from google import genai


# # # =========================================================
# # # CONFIG — EDIT THESE
# # # =========================================================

# # MEDICINES_FOLDER = r"C:\Users\Satya Sri\Desktop\gemini multi turn\medicines_data"

# # API_KEYS_FILE    = "api_keys.txt"

# # OUTPUT_FILE      = "multiturn.jsonl"

# # MODEL_NAME       = "gemini-2.0-flash"

# # PROGRESS_FILE    = "progress.json"

# # BATCH_DELAY      = 7   # seconds between calls (quota safety)


# # # =========================================================
# # # PROMPT
# # # =========================================================

# # SYSTEM_PROMPT = """Role:
# # You are an expert homeopathic medical dialogue generator.

# # Task:
# # Generate realistic multi-turn doctor–patient conversations strictly based on the provided Materia Medica entry.
# # The goal is to create medically grounded conversations that help train a model to map patient symptoms to the correct homeopathic remedy.

# # Primary Objective:
# # Simulate a natural clinical consultation where a patient reports symptoms, the doctor gathers details, identifies the condition, recommends the correct remedy, and provides usage advice.

# # Strict Medical Grounding Rules:
# # 1. Only use symptoms explicitly mentioned in the provided Materia Medica.
# # 2. Do not invent new symptoms.
# # 3. Use symptom combinations that match the remedy indications.
# # 4. The recommended medicine must match the symptom pattern.
# # 5. Diagnosis reasoning must be based on listed symptoms.

# # Required Dialogue Structure:
# # You MUST generate three ordered phases:

# # 1. <diagnosis>
# # Doctor gathers symptoms.
# # Patient reports complaints.
# # Doctor asks symptom-specific questions.
# # Requirements:
# # - At least 5 symptom-related exchanges.
# # - Alternate <user> and <assistant>.
# # - Focus on appetite, digestion, pain location, bowel symptoms, and other relevant remedy features.

# # 2. <treatment>
# # Doctor explains condition.
# # Doctor connects symptoms to diagnosis.
# # Doctor recommends the correct remedy.
# # Requirements:
# # - 2–3 assistant responses.
# # - Must include remedy name explicitly.

# # 3. <consultation>
# # Patient asks about medicine usage.
# # Doctor gives dosage instructions.
# # Doctor suggests diet or lifestyle advice.
# # Doctor recommends follow-up.
# # Requirements:
# # - At least 4 responses.
# # - Include dosage guidance.
# # - Include precaution advice.

# # Conversation Length Requirements:
# # - Diagnosis phase: 10–14 turns
# # - Treatment phase: 2–3 turns
# # - Consultation phase: 4–6 turns
# # - Total turns: 16–22

# # Output Format (MANDATORY):
# # Generate output strictly in this exact format:

# # {"text": "<medicine>
# # medicine_name
# # </medicine>
# # <context>
# # Summarize the key symptoms and clinical indications from the Materia Medica that are relevant to the dialogue case.
# # Include only important symptoms.
# # </context>
# # <dialogue>
# # <diagnosis>
# # <user>
# # Patient describes first symptom naturally.
# # </user>
# # <assistant>
# # Doctor asks about specific symptom.
# # </assistant>
# # </diagnosis>
# # <treatment>
# # <assistant>
# # Doctor explains condition using gathered symptoms.
# # </assistant>
# # <assistant>
# # Doctor recommends the correct remedy by name.
# # </assistant>
# # </treatment>
# # <consultation>
# # <user>
# # Patient asks how to take the medicine.
# # </user>
# # <assistant>
# # Doctor gives dosage guidance.
# # </assistant>
# # <assistant>
# # Doctor suggests dietary or lifestyle precautions.
# # </assistant>
# # <assistant>
# # Doctor recommends follow-up monitoring.
# # </assistant>
# # </consultation>
# # </dialogue>
# # "}

# # Formatting Rules (Strict):
# # - Always include <medicine>.
# # - Always include <context>.
# # - Always include <dialogue>.
# # - Always include <diagnosis>, <treatment>, <consultation>.
# # - Always alternate <user> and <assistant>.
# # - Use natural conversational language.
# # - Do NOT include annotations.
# # - Do NOT add markdown formatting.
# # - Do NOT output explanations.
# # - Output must be valid JSON with the text field.
# # - Do not leave empty sections.

# # Language Rules:
# # - Use realistic patient speech.
# # - Avoid medical jargon unless spoken by the doctor.
# # - Keep tone conversational.
# # - Avoid repetitive sentences.
# # - Ensure symptom flow feels natural."""


# # def build_prompt(medicine_name, medicine_text):
# #     return f"""{SYSTEM_PROMPT}

# # Now generate a dialogue for the following medicine:

# # MEDICINE NAME: {medicine_name}

# # MATERIA MEDICA TEXT:
# # {medicine_text}"""


# # # =========================================================
# # # API KEY MANAGEMENT
# # # =========================================================

# # def load_api_keys():
# #     if not os.path.exists(API_KEYS_FILE):
# #         raise FileNotFoundError(
# #             f"'{API_KEYS_FILE}' not found. "
# #             "Create this file with one API key per line."
# #         )
# #     with open(API_KEYS_FILE, encoding="utf-8") as f:
# #         keys = [line.strip() for line in f if line.strip()]
# #     if not keys:
# #         raise ValueError("No API keys found in api_keys.txt")
# #     print(f"Loaded {len(keys)} API key(s)")
# #     return keys


# # # =========================================================
# # # PROGRESS SYSTEM
# # # =========================================================

# # def load_progress():
# #     if not os.path.exists(PROGRESS_FILE):
# #         return {"last_index": 0, "api_index": 0}
# #     with open(PROGRESS_FILE, encoding="utf-8") as f:
# #         return json.load(f)


# # def save_progress(medicine_index, api_index):
# #     with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
# #         json.dump(
# #             {"last_index": medicine_index, "api_index": api_index},
# #             f, indent=2
# #         )


# # # =========================================================
# # # GEMINI CLIENT WITH KEY ROTATION
# # # =========================================================

# # class GeminiClient:

# #     def __init__(self, api_keys, start_key_index=0):
# #         self.api_keys  = api_keys
# #         self.key_index = start_key_index
# #         self.call_count = 0
# #         self._init_client()

# #     def _init_client(self):
# #         key = self.api_keys[self.key_index]
# #         self.client = genai.Client(api_key=key)
# #         print(f"  [KEY] Using API key #{self.key_index + 1}")

# #     def _rotate_key(self):
# #         self.key_index += 1
# #         if self.key_index >= len(self.api_keys):
# #             raise RuntimeError(
# #                 "ALL API KEYS EXHAUSTED.\n"
# #                 "All keys have hit their daily quota.\n"
# #                 "Please wait 24 hours and resume tomorrow.\n"
# #                 "Progress has been saved — just run the script again tomorrow."
# #             )
# #         print(f"  [ROTATE] Switching to API key #{self.key_index + 1}")
# #         self._init_client()

# #     def call(self, prompt):
# #         """Call Gemini with automatic retry and key rotation."""
# #         for attempt in range(3):
# #             try:
# #                 self.call_count += 1
# #                 response = self.client.models.generate_content(
# #                     model=MODEL_NAME,
# #                     contents=prompt
# #                 )
# #                 text = response.text
# #                 if not text:
# #                     raise ValueError("Empty response from API")
# #                 return text

# #             except Exception as e:
# #                 err_msg = str(e).lower()

# #                 # Quota / rate limit → rotate key
# #                 if any(kw in err_msg for kw in ["quota", "rate limit", "429", "resource exhausted"]):
# #                     print(f"  [QUOTA] Key #{self.key_index + 1} exhausted.")
# #                     self._rotate_key()
# #                     continue  # retry with new key immediately

# #                 # Other errors → wait and retry
# #                 wait = 5 * (attempt + 1)
# #                 print(f"  [ERROR] {str(e)[:80]} — retrying in {wait}s (attempt {attempt+1}/3)")
# #                 time.sleep(wait)

# #         print("  [FAIL] All 3 attempts failed for this medicine.")
# #         return None


# # # =========================================================
# # # PARSING: XML dialogue → messages format
# # # =========================================================

# # def extract_text_from_response(raw):
# #     """Pull the text value out of the JSON response."""
# #     # Strip markdown code fences if present
# #     raw = re.sub(r'```json\s*', '', raw)
# #     raw = re.sub(r'```\s*', '', raw)
# #     raw = raw.strip()

# #     # Try standard JSON parse
# #     try:
# #         data = json.loads(raw)
# #         return data.get("text", "")
# #     except json.JSONDecodeError:
# #         pass

# #     # Fallback: extract text value with regex
# #     match = re.search(r'"text"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
# #     if match:
# #         return match.group(1)

# #     # Last resort: treat the whole thing as the text
# #     return raw


# # def parse_to_messages(text):
# #     """
# #     Convert the XML-tagged dialogue text into
# #     {"messages": [{"role": ..., "content": ...}, ...]} format.
# #     """
# #     # Extract medicine and context
# #     med_match = re.search(r'<medicine>(.*?)</medicine>', text, re.DOTALL)
# #     ctx_match  = re.search(r'<context>(.*?)</context>',  text, re.DOTALL)

# #     medicine = med_match.group(1).strip() if med_match else ""
# #     context  = ctx_match.group(1).strip()  if ctx_match  else ""

# #     if not medicine and not context:
# #         return None

# #     system_content = (
# #         f"<medicine>{medicine}</medicine>"
# #         f"<context>{context}</context>"
# #     )
# #     messages = [{"role": "system", "content": system_content}]

# #     # Extract dialogue block
# #     dial_match = re.search(r'<dialogue>(.*?)</dialogue>', text, re.DOTALL)
# #     if not dial_match:
# #         return None

# #     dialogue_text = dial_match.group(1)

# #     # Extract all user/assistant turns in order
# #     turns = re.findall(
# #         r'<(user|assistant)>(.*?)<\/\1>',
# #         dialogue_text,
# #         re.DOTALL
# #     )

# #     if not turns:
# #         return None

# #     # Merge consecutive same-role messages (handles treatment's double <assistant>)
# #     merged = []
# #     for role, content in turns:
# #         content = content.strip()
# #         if not content:
# #             continue
# #         if merged and merged[-1]["role"] == role:
# #             merged[-1]["content"] += " " + content
# #         else:
# #             merged.append({"role": role, "content": content})

# #     if not merged:
# #         return None

# #     messages.extend(merged)
# #     return messages


# # # =========================================================
# # # MAIN PIPELINE
# # # =========================================================

# # def process_all():
# #     # Load keys and progress
# #     api_keys = load_api_keys()
# #     progress = load_progress()
# #     start_index     = progress["last_index"]
# #     start_key_index = progress["api_index"]

# #     # Init Gemini client
# #     client = GeminiClient(api_keys, start_key_index)

# #     # Gather all medicine files
# #     base  = Path(MEDICINES_FOLDER)
# #     files = sorted(base.glob("*.txt"))
# #     total = len(files)

# #     print(f"\nFound {total} medicine files.")
# #     print(f"Resuming from medicine #{start_index + 1}\n")

# #     # Stats
# #     converted = 0
# #     skipped   = 0

# #     # Open output file in append mode so we don't overwrite on resume
# #     with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

# #         for idx in range(start_index, total):
# #             file = files[idx]
# #             medicine_name = file.stem.replace("-", " ").title()

# #             print(f"[{idx+1}/{total}] Processing: {medicine_name}")

# #             # Read medicine text
# #             try:
# #                 medicine_text = file.read_text(encoding="utf-8")
# #                 medicine_text = re.sub(r'\s+', ' ', medicine_text).strip()
# #             except Exception as e:
# #                 print(f"  [SKIP] Could not read file: {e}")
# #                 skipped += 1
# #                 save_progress(idx + 1, client.key_index)
# #                 continue

# #             if not medicine_text:
# #                 print("  [SKIP] Empty file.")
# #                 skipped += 1
# #                 save_progress(idx + 1, client.key_index)
# #                 continue

# #             # Build prompt and call API
# #             prompt   = build_prompt(medicine_name, medicine_text)
# #             response = client.call(prompt)

# #             if not response:
# #                 print("  [SKIP] No response from API.")
# #                 skipped += 1
# #                 save_progress(idx + 1, client.key_index)
# #                 time.sleep(BATCH_DELAY)
# #                 continue

# #             # Extract text from response
# #             dialogue_text = extract_text_from_response(response)

# #             if not dialogue_text:
# #                 print("  [SKIP] Could not extract text from response.")
# #                 skipped += 1
# #                 save_progress(idx + 1, client.key_index)
# #                 time.sleep(BATCH_DELAY)
# #                 continue

# #             # Parse to messages format
# #             messages = parse_to_messages(dialogue_text)

# #             if not messages or len(messages) < 4:
# #                 print(f"  [SKIP] Parsed only {len(messages) if messages else 0} messages — too few.")
# #                 skipped += 1
# #                 save_progress(idx + 1, client.key_index)
# #                 time.sleep(BATCH_DELAY)
# #                 continue

# #             # Write to output
# #             record = {"messages": messages}
# #             outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
# #             outfile.flush()  # ensure it's written immediately

# #             converted += 1
# #             print(f"  [OK] Saved {len(messages)} messages ({len(messages)-1} turns)")

# #             # Save progress after each successful medicine
# #             save_progress(idx + 1, client.key_index)

# #             time.sleep(BATCH_DELAY)

# #     # Done
# #     print("\n" + "=" * 55)
# #     print("ALL MEDICINES PROCESSED")
# #     print(f"  Converted : {converted}")
# #     print(f"  Skipped   : {skipped}")
# #     print(f"  Output    : {OUTPUT_FILE}")
# #     print(f"  API calls : {client.call_count}")
# #     print("=" * 55)


# # # =========================================================

# # if __name__ == "__main__":
# #     try:
# #         process_all()
# #     except RuntimeError as e:
# #         print("\n" + "=" * 55)
# #         print(str(e))
# #         print("=" * 55)
# #     except KeyboardInterrupt:
# #         print("\n[STOPPED] Script interrupted by user.")
# #         print("Progress has been saved. Run again to resume.")


# #!/usr/bin/env python3
# """
# Homeopathy Multi-Turn Dialogue Generator
# =========================================
# - Reads all .txt files from medicines_data folder
# - Generates multi-turn doctor-patient conversations using Gemini API
# - Converts to {"messages": [...]} format for fine-tuning
# - Rotates API keys automatically on quota exhaustion
# - Resumes from last processed medicine via progress.json
# - Output: multiturn.jsonl (single file, all medicines)
# """

# import os
# import re
# import json
# import time
# from pathlib import Path
# from google import genai


# # =========================================================
# # CONFIG — EDIT THESE
# # =========================================================

# MEDICINES_FOLDER = r"C:\Users\Satya Sri\Desktop\gemini multi turn\medicines_data"

# API_KEYS_FILE    = "api_keys.txt"

# OUTPUT_FILE      = "multiturn.jsonl"

# MODEL_NAME       = "gemini-2.5-flash-lite"

# PROGRESS_FILE    = "progress.json"

# BATCH_DELAY      = 7   # seconds between calls (quota safety)


# # =========================================================
# # SYSTEM PROMPT (sent via system_instruction, NOT in contents)
# # =========================================================

# SYSTEM_PROMPT = """Role:
# You are an expert homeopathic medical dialogue generator.

# Task:
# Generate realistic multi-turn doctor-patient conversations strictly based on the provided Materia Medica entry.
# The goal is to create medically grounded conversations that help train a model to map patient symptoms to the correct homeopathic remedy.

# Primary Objective:
# Simulate a natural clinical consultation where a patient reports symptoms, the doctor gathers details, identifies the condition, recommends the correct remedy, and provides usage advice.

# Strict Medical Grounding Rules:
# 1. Only use symptoms explicitly mentioned in the provided Materia Medica.
# 2. Do not invent new symptoms.
# 3. Use symptom combinations that match the remedy indications.
# 4. The recommended medicine must match the symptom pattern.
# 5. Diagnosis reasoning must be based on listed symptoms.

# Required Dialogue Structure:
# You MUST generate three ordered phases:

# 1. <diagnosis>
# Doctor gathers symptoms.
# Patient reports complaints.
# Doctor asks symptom-specific questions.
# Requirements:
# - At least 5 symptom-related exchanges.
# - Alternate <user> and <assistant>.
# - Focus on appetite, digestion, pain location, bowel symptoms, and other relevant remedy features.

# 2. <treatment>
# Doctor explains condition.
# Doctor connects symptoms to diagnosis.
# Doctor recommends the correct remedy.
# Requirements:
# - 2-3 assistant responses.
# - Must include remedy name explicitly.

# 3. <consultation>
# Patient asks about medicine usage.
# Doctor gives dosage instructions.
# Doctor suggests diet or lifestyle advice.
# Doctor recommends follow-up.
# Requirements:
# - At least 4 responses.
# - Include dosage guidance.
# - Include precaution advice.

# Conversation Length Requirements:
# - Diagnosis phase: 10-14 turns
# - Treatment phase: 2-3 turns
# - Consultation phase: 4-6 turns
# - Total turns: 16-22

# Output Format (MANDATORY):
# Generate output strictly in this exact JSON format and nothing else:

# {"text": "<medicine>\nMEDICINE_NAME\n</medicine>\n<context>\nSummarize the key symptoms here.\n</context>\n<dialogue>\n<diagnosis>\n<user>\nPatient text.\n</user>\n<assistant>\nDoctor text.\n</assistant>\n</diagnosis>\n<treatment>\n<assistant>\nDoctor explains condition.\n</assistant>\n<assistant>\nDoctor recommends remedy by name.\n</assistant>\n</treatment>\n<consultation>\n<user>\nPatient asks about dosage.\n</user>\n<assistant>\nDoctor gives dosage guidance.\n</assistant>\n<assistant>\nDoctor gives dietary advice.\n</assistant>\n<assistant>\nDoctor recommends follow-up.\n</assistant>\n</consultation>\n</dialogue>\n"}

# Formatting Rules (Strict):
# - Always include <medicine>, <context>, <dialogue>.
# - Always include <diagnosis>, <treatment>, <consultation> inside <dialogue>.
# - Always alternate <user> and <assistant> inside <diagnosis>.
# - Use natural conversational language.
# - Do NOT include annotations or markdown formatting.
# - Do NOT output explanations outside the JSON.
# - Output must be a single valid JSON object with only a "text" field.
# - Do not leave empty sections.

# Language Rules:
# - Use realistic patient speech.
# - Avoid medical jargon unless spoken by the doctor.
# - Keep tone conversational.
# - Avoid repetitive sentences.
# - Ensure symptom flow feels natural."""


# # =========================================================
# # PROMPT BUILDER — only medicine-specific content here
# # =========================================================

# def build_prompt(medicine_name, medicine_text):
#     return (
#         f"Generate a dialogue for this medicine:\n\n"
#         f"MEDICINE NAME: {medicine_name}\n\n"
#         f"MATERIA MEDICA TEXT:\n{medicine_text}"
#     )


# # =========================================================
# # API KEY MANAGEMENT
# # =========================================================

# def load_api_keys():
#     if not os.path.exists(API_KEYS_FILE):
#         raise FileNotFoundError(
#             f"'{API_KEYS_FILE}' not found. "
#             "Create this file with one API key per line."
#         )
#     with open(API_KEYS_FILE, encoding="utf-8") as f:
#         keys = [line.strip() for line in f if line.strip()]
#     if not keys:
#         raise ValueError("No API keys found in api_keys.txt")
#     print(f"Loaded {len(keys)} API key(s)")
#     return keys


# # =========================================================
# # PROGRESS SYSTEM
# # =========================================================

# def load_progress():
#     if not os.path.exists(PROGRESS_FILE):
#         return {"last_index": 0, "api_index": 0}
#     with open(PROGRESS_FILE, encoding="utf-8") as f:
#         return json.load(f)


# def save_progress(medicine_index, api_index):
#     with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
#         json.dump(
#             {"last_index": medicine_index, "api_index": api_index},
#             f, indent=2
#         )


# # =========================================================
# # GEMINI CLIENT WITH KEY ROTATION
# # =========================================================

# class GeminiClient:

#     def __init__(self, api_keys, start_key_index=0):
#         self.api_keys   = api_keys
#         self.key_index  = start_key_index
#         self.call_count = 0
#         self._init_client()

#     def _init_client(self):
#         key = self.api_keys[self.key_index]
#         self.client = genai.Client(api_key=key)
#         print(f"  [KEY] Using API key #{self.key_index + 1}")

#     def _rotate_key(self):
#         self.key_index += 1
#         if self.key_index >= len(self.api_keys):
#             raise RuntimeError(
#                 "ALL API KEYS EXHAUSTED.\n"
#                 "All keys have hit their daily quota.\n"
#                 "Please wait 24 hours and resume tomorrow.\n"
#                 "Progress has been saved — just run the script again tomorrow."
#             )
#         print(f"  [ROTATE] Switching to API key #{self.key_index + 1}")
#         self._init_client()

#     def call(self, prompt):
#         """Call Gemini with system_instruction separated from contents."""
#         for attempt in range(3):
#             try:
#                 self.call_count += 1
#                 response = self.client.models.generate_content(
#                     model=MODEL_NAME,
#                     config=genai.types.GenerateContentConfig(
#                         system_instruction=SYSTEM_PROMPT,
#                         max_output_tokens=8192,
#                     ),
#                     contents=prompt
#                 )
#                 text = response.text
#                 if not text:
#                     raise ValueError("Empty response from API")
#                 return text

#             except Exception as e:
#                 err_msg = str(e).lower()

#                 # Quota / rate limit → rotate key immediately
#                 if any(kw in err_msg for kw in ["quota", "rate limit", "429", "resource exhausted"]):
#                     print(f"  [QUOTA] Key #{self.key_index + 1} exhausted.")
#                     self._rotate_key()
#                     continue

#                 # Other errors → wait and retry
#                 wait = 5 * (attempt + 1)
#                 print(f"  [ERROR] {str(e)[:80]} — retrying in {wait}s (attempt {attempt+1}/3)")
#                 time.sleep(wait)

#         print("  [FAIL] All 3 attempts failed for this medicine.")
#         return None


# # =========================================================
# # PARSING: XML dialogue → messages format
# # =========================================================

# def extract_text_from_response(raw):
#     """Pull the text value out of the JSON response."""
#     # Strip markdown code fences if present
#     raw = re.sub(r'```json\s*', '', raw)
#     raw = re.sub(r'```\s*', '', raw)
#     raw = raw.strip()

#     # Try standard JSON parse first
#     try:
#         data = json.loads(raw)
#         return data.get("text", "")
#     except json.JSONDecodeError:
#         pass

#     # Fallback: extract text value with regex (handles unescaped newlines inside string)
#     match = re.search(r'"text"\s*:\s*"(.*?)"\s*\}', raw, re.DOTALL)
#     if match:
#         return match.group(1)

#     # Last resort: treat whole response as the text
#     return raw


# def parse_to_messages(text):
#     """
#     Convert XML-tagged dialogue text into
#     [{"role": "system", ...}, {"role": "user", ...}, ...] format.
#     """
#     # Extract medicine and context
#     med_match = re.search(r'<medicine>(.*?)</medicine>', text, re.DOTALL)
#     ctx_match  = re.search(r'<context>(.*?)</context>',  text, re.DOTALL)

#     medicine = med_match.group(1).strip() if med_match else ""
#     context  = ctx_match.group(1).strip()  if ctx_match  else ""

#     if not medicine and not context:
#         return None

#     system_content = (
#         f"<medicine>{medicine}</medicine>"
#         f"<context>{context}</context>"
#     )
#     messages = [{"role": "system", "content": system_content}]

#     # Extract dialogue block
#     dial_match = re.search(r'<dialogue>(.*?)</dialogue>', text, re.DOTALL)
#     if not dial_match:
#         return None

#     dialogue_text = dial_match.group(1)

#     # Extract all user/assistant turns in order
#     turns = re.findall(
#         r'<(user|assistant)>(.*?)<\/\1>',
#         dialogue_text,
#         re.DOTALL
#     )

#     if not turns:
#         return None

#     # Merge consecutive same-role messages
#     # (treatment phase has multiple consecutive <assistant> blocks)
#     merged = []
#     for role, content in turns:
#         content = content.strip()
#         if not content:
#             continue
#         if merged and merged[-1]["role"] == role:
#             merged[-1]["content"] += " " + content
#         else:
#             merged.append({"role": role, "content": content})

#     if not merged:
#         return None

#     messages.extend(merged)
#     return messages


# # =========================================================
# # MAIN PIPELINE
# # =========================================================

# def process_all():
#     # Load keys and progress
#     api_keys = load_api_keys()
#     progress = load_progress()
#     start_index     = progress["last_index"]
#     start_key_index = progress["api_index"]

#     # Init Gemini client
#     client = GeminiClient(api_keys, start_key_index)

#     # Gather all medicine files
#     base  = Path(MEDICINES_FOLDER)
#     files = sorted(base.glob("*.txt"))
#     total = len(files)

#     print(f"\nFound {total} medicine files.")
#     print(f"Resuming from medicine #{start_index + 1}\n")

#     converted = 0
#     skipped   = 0

#     # Append mode — safe to resume without losing previous output
#     with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

#         for idx in range(start_index, total):
#             file          = files[idx]
#             medicine_name = file.stem.replace("-", " ").title()

#             print(f"[{idx+1}/{total}] Processing: {medicine_name}")

#             # Read medicine text
#             try:
#                 medicine_text = file.read_text(encoding="utf-8")
#                 medicine_text = re.sub(r'\s+', ' ', medicine_text).strip()
#             except Exception as e:
#                 print(f"  [SKIP] Could not read file: {e}")
#                 skipped += 1
#                 save_progress(idx + 1, client.key_index)
#                 continue

#             if not medicine_text:
#                 print("  [SKIP] Empty file.")
#                 skipped += 1
#                 save_progress(idx + 1, client.key_index)
#                 continue

#             # Build prompt and call API
#             prompt   = build_prompt(medicine_name, medicine_text)
#             response = client.call(prompt)

#             if not response:
#                 print("  [SKIP] No response from API.")
#                 skipped += 1
#                 save_progress(idx + 1, client.key_index)
#                 time.sleep(BATCH_DELAY)
#                 continue

#             # Extract dialogue text from JSON response
#             dialogue_text = extract_text_from_response(response)

#             if not dialogue_text:
#                 print("  [SKIP] Could not extract text from response.")
#                 skipped += 1
#                 save_progress(idx + 1, client.key_index)
#                 time.sleep(BATCH_DELAY)
#                 continue

#             # Parse to messages format
#             messages = parse_to_messages(dialogue_text)

#             if not messages or len(messages) < 4:
#                 print(f"  [SKIP] Too few messages parsed ({len(messages) if messages else 0}).")
#                 skipped += 1
#                 save_progress(idx + 1, client.key_index)
#                 time.sleep(BATCH_DELAY)
#                 continue

#             # Write to output
#             record = {"messages": messages}
#             outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
#             outfile.flush()  # flush after every write — no data loss on crash

#             converted += 1
#             print(f"  [OK] Saved {len(messages)} messages ({len(messages)-1} turns)")

#             # Save progress after every successful medicine
#             save_progress(idx + 1, client.key_index)

#             time.sleep(BATCH_DELAY)

#     # Final summary
#     print("\n" + "=" * 55)
#     print("ALL MEDICINES PROCESSED")
#     print(f"  Converted : {converted}")
#     print(f"  Skipped   : {skipped}")
#     print(f"  Output    : {OUTPUT_FILE}")
#     print(f"  API calls : {client.call_count}")
#     print("=" * 55)


# # =========================================================

# if __name__ == "__main__":
#     try:
#         process_all()
#     except RuntimeError as e:
#         print("\n" + "=" * 55)
#         print(str(e))
#         print("=" * 55)
#     except KeyboardInterrupt:
#         print("\n[STOPPED] Script interrupted by user.")
#         print("Progress has been saved. Run again to resume.")
#!/usr/bin/env python3
"""
Homeopathy Multi-Turn Dialogue Generator
=========================================
- Reads all .txt files from medicines_data folder
- Generates multi-turn doctor-patient conversations using Gemini API
- Converts to {"messages": [...]} format for fine-tuning
- Rotates API keys automatically on quota exhaustion
- Resumes from last processed medicine via progress.json
- Output: multiturn.jsonl (single file, all medicines)
"""

import os
import re
import json
import time
from pathlib import Path
from google import genai
from google.genai import types


# =========================================================
# CONFIG — EDIT THESE
# =========================================================

MEDICINES_FOLDER = r"medicines_data"

API_KEYS_FILE    = "api_keys.txt"

OUTPUT_FILE      = "multiturn.jsonl"

MODEL_NAME       = "gemini-2.5-flash-lite"

PROGRESS_FILE    = "progress.json"

BATCH_DELAY      = 7   # seconds between calls (quota safety)


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """You are an expert homeopathic medical dialogue generator.

Generate a realistic multi-turn doctor-patient conversation strictly based on the provided Materia Medica entry.

Rules:
1. Only use symptoms explicitly mentioned in the Materia Medica text.
2. Do not invent new symptoms.
3. The recommended medicine must match the symptom pattern.
4. Generate three phases: diagnosis, treatment, consultation.
5. Diagnosis: at least 10 alternating user/assistant turns where doctor gathers symptoms.
6. Treatment: doctor explains the condition and explicitly names the remedy.
7. Consultation: patient asks about dosage, doctor gives instructions and lifestyle advice.

Output ONLY a valid JSON object in this exact structure, nothing else, no markdown:

{
  "medicine": "MEDICINE_NAME",
  "context": "brief summary of key symptoms",
  "dialogue": {
    "diagnosis": [
      {"role": "user", "content": "patient text"},
      {"role": "assistant", "content": "doctor text"}
    ],
    "treatment": [
      {"role": "assistant", "content": "doctor explains and names the remedy"}
    ],
    "consultation": [
      {"role": "user", "content": "patient asks about dosage"},
      {"role": "assistant", "content": "doctor gives dosage and advice"}
    ]
  }
}"""


# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(medicine_name, medicine_text):
    return (
        f"Generate a dialogue for this medicine:\n\n"
        f"MEDICINE NAME: {medicine_name}\n\n"
        f"MATERIA MEDICA TEXT:\n{medicine_text}"
    )


# =========================================================
# API KEY MANAGEMENT
# =========================================================

def load_api_keys():
    if not os.path.exists(API_KEYS_FILE):
        raise FileNotFoundError(
            f"'{API_KEYS_FILE}' not found. "
            "Create this file with one API key per line."
        )
    with open(API_KEYS_FILE, encoding="utf-8") as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise ValueError("No API keys found in api_keys.txt")
    print(f"Loaded {len(keys)} API key(s)")
    return keys


# =========================================================
# PROGRESS SYSTEM
# =========================================================

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"last_index": 0, "api_index": 0}
    with open(PROGRESS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_progress(medicine_index, api_index):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"last_index": medicine_index, "api_index": api_index},
            f, indent=2
        )


# =========================================================
# GEMINI CLIENT WITH KEY ROTATION
# =========================================================

class GeminiClient:

    def __init__(self, api_keys, start_key_index=0):
        self.api_keys   = api_keys
        self.key_index  = start_key_index
        self.call_count = 0
        self._init_client()

    def _init_client(self):
        key = self.api_keys[self.key_index]
        self.client = genai.Client(api_key=key)
        print(f"  [KEY] Using API key #{self.key_index + 1}")

    def _rotate_key(self):
        self.key_index += 1
        if self.key_index >= len(self.api_keys):
            raise RuntimeError(
                "ALL API KEYS EXHAUSTED.\n"
                "All keys have hit their daily quota.\n"
                "Please wait 24 hours and resume tomorrow.\n"
                "Progress has been saved — just run the script again tomorrow."
            )
        print(f"  [ROTATE] Switching to API key #{self.key_index + 1}")
        self._init_client()

    def call(self, prompt):
        for attempt in range(3):
            try:
                self.call_count += 1
                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=8192,
                    ),
                    contents=prompt
                )
                text = response.text
                if not text:
                    raise ValueError("Empty response from API")
                return text

            except Exception as e:
                err_msg = str(e).lower()

                if any(kw in err_msg for kw in ["quota", "rate limit", "429", "resource exhausted", "403", "permission denied", "consumer invalid"]):
                    print(f"  [QUOTA] Key #{self.key_index + 1} exhausted.")
                    self._rotate_key()
                    continue

                wait = 5 * (attempt + 1)
                print(f"  [ERROR] {str(e)[:80]} — retrying in {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)

        print("  [FAIL] All 3 attempts failed for this medicine.")
        return None


# =========================================================
# PARSER: nested JSON → messages format
# =========================================================

def clean_json_response(raw):
    """Strip markdown fences and extract JSON object."""
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    # Find the outermost { ... }
    start = raw.find('{')
    end   = raw.rfind('}')
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(raw[start:end+1])
    except json.JSONDecodeError:
        return None


def parse_to_messages(raw_response):
    """
    Convert Gemini's nested JSON response into
    [{"role": ..., "content": ...}, ...] messages format.

    Handles two formats the model may return:
    Format A: {"medicine": ..., "context": ..., "dialogue": {"diagnosis": [...], ...}}
    Format B: {"text": "<medicine>...</medicine><context>...</context><dialogue>..."}
    """
    data = clean_json_response(raw_response)
    if not data:
        return None

    # ---- Format A: nested JSON with lists ----
    if "dialogue" in data and isinstance(data["dialogue"], dict):

        medicine = data.get("medicine", "").strip()
        context  = data.get("context",  "").strip()

        if not medicine:
            return None

        system_content = (
            f"<medicine>{medicine}</medicine>"
            f"<context>{context}</context>"
        )
        messages = [{"role": "system", "content": system_content}]

        dialogue = data["dialogue"]

        # Collect all turns from all phases in order
        all_turns = []
        for phase in ["diagnosis", "treatment", "consultation"]:
            phase_turns = dialogue.get(phase, [])

            # Phase may be a list of {"role":..,"content":..} dicts
            if isinstance(phase_turns, list):
                for turn in phase_turns:
                    if isinstance(turn, dict):
                        role    = turn.get("role", "").strip()
                        content = turn.get("content", "").strip()
                        if role in ("user", "assistant") and content:
                            all_turns.append({"role": role, "content": content})

            # Phase may be a dict with alternating user/assistant keys (less common)
            elif isinstance(phase_turns, dict):
                for key, value in phase_turns.items():
                    if key in ("user", "assistant") and isinstance(value, str):
                        all_turns.append({"role": key, "content": value.strip()})

        if not all_turns:
            return None

        # Merge consecutive same-role turns
        merged = []
        for turn in all_turns:
            if merged and merged[-1]["role"] == turn["role"]:
                merged[-1]["content"] += " " + turn["content"]
            else:
                merged.append({"role": turn["role"], "content": turn["content"]})

        messages.extend(merged)
        return messages

    # ---- Format B: {"text": "...xml..."} ----
    if "text" in data:
        text = data["text"]

        med_match = re.search(r'<medicine>(.*?)</medicine>', text, re.DOTALL)
        ctx_match  = re.search(r'<context>(.*?)</context>',  text, re.DOTALL)
        medicine = med_match.group(1).strip() if med_match else ""
        context  = ctx_match.group(1).strip()  if ctx_match  else ""

        system_content = (
            f"<medicine>{medicine}</medicine>"
            f"<context>{context}</context>"
        )
        messages = [{"role": "system", "content": system_content}]

        dial_match = re.search(r'<dialogue>(.*?)</dialogue>', text, re.DOTALL)
        if not dial_match:
            return None

        turns = re.findall(
            r'<(user|assistant)>(.*?)<\/\1>',
            dial_match.group(1),
            re.DOTALL
        )

        merged = []
        for role, content in turns:
            content = content.strip()
            if not content:
                continue
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += " " + content
            else:
                merged.append({"role": role, "content": content})

        if not merged:
            return None

        messages.extend(merged)
        return messages

    return None


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_all():
    api_keys = load_api_keys()
    progress = load_progress()
    start_index     = progress["last_index"]
    start_key_index = progress["api_index"]

    client = GeminiClient(api_keys, start_key_index)

    base  = Path(MEDICINES_FOLDER)
    files = sorted(base.glob("*.txt"))
    total = len(files)

    print(f"\nFound {total} medicine files.")
    print(f"Resuming from medicine #{start_index + 1}\n")

    converted = 0
    skipped   = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

        for idx in range(start_index, total):
            file          = files[idx]
            medicine_name = file.stem.replace("-", " ").title()

            print(f"[{idx+1}/{total}] Processing: {medicine_name}")

            # Read medicine text
            try:
                medicine_text = file.read_text(encoding="utf-8")
                medicine_text = re.sub(r'\s+', ' ', medicine_text).strip()
            except Exception as e:
                print(f"  [SKIP] Could not read file: {e}")
                skipped += 1
                save_progress(idx + 1, client.key_index)
                continue

            if not medicine_text:
                print("  [SKIP] Empty file.")
                skipped += 1
                save_progress(idx + 1, client.key_index)
                continue

            # Call API
            prompt   = build_prompt(medicine_name, medicine_text)
            response = client.call(prompt)

            if not response:
                print("  [SKIP] No response from API.")
                skipped += 1
                save_progress(idx + 1, client.key_index)
                time.sleep(BATCH_DELAY)
                continue

            # Parse response to messages
            messages = parse_to_messages(response)

            if not messages or len(messages) < 4:
                print(f"  [SKIP] Too few messages ({len(messages) if messages else 0}). Raw snippet: {response[:200]}")
                skipped += 1
                save_progress(idx + 1, client.key_index)
                time.sleep(BATCH_DELAY)
                continue

            # Write to output
            record = {"messages": messages}
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            outfile.flush()

            converted += 1
            print(f"  [OK] Saved {len(messages)} messages ({len(messages)-1} turns)")

            save_progress(idx + 1, client.key_index)
            time.sleep(BATCH_DELAY)

    print("\n" + "=" * 55)
    print("ALL MEDICINES PROCESSED")
    print(f"  Converted : {converted}")
    print(f"  Skipped   : {skipped}")
    print(f"  Output    : {OUTPUT_FILE}")
    print(f"  API calls : {client.call_count}")
    print("=" * 55)


# =========================================================

if __name__ == "__main__":
    try:
        process_all()
    except RuntimeError as e:
        print("\n" + "=" * 55)
        print(str(e))
        print("=" * 55)
    except KeyboardInterrupt:
        print("\n[STOPPED] Script interrupted by user.")
        print("Progress has been saved. Run again to resume.")