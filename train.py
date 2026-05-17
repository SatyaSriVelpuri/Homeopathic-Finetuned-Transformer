"""
Homeopathy model fine-tuning script.
Fine-tunes bharatgenai/Param-1-2.9B-Instruct on homeopathy Q&A data
and pushes the result to mnj-hf/homeopathy on Hugging Face Hub.

Requirements:
    pip install torch transformers datasets accelerate
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "bharatgenai/Param-1-2.9B-Instruct"
HUB_MODEL_ID = "mnj-hf/homeopathy"
MAX_LEN = 512

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for qualified homeopathic medical practitioners.\n"
    "Your role is to help practitioners think through cases, summarize repertory, compare remedy options,\n"
    "and highlight red-flag symptoms that require conventional medical evaluation.\n"
    "Base your reasoning on classical and widely accepted homeopathic sources and general medical knowledge."
)

INPUT_FILES = [
    "groq_train.jsonl",
    "training.jsonl",
    "gemini_train.jsonl",
]

CLEAN_FILES = [
    "train_clean.jsonl",
    "training_clean.jsonl",
    "train3_clean.jsonl",
]


# ---------------------------------------------------------------------------
# Step 1 — Load base model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print("Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if "type" not in config.rope_scaling:
            config.rope_scaling["type"] = "linear"
        if "factor" not in config.rope_scaling:
            config.rope_scaling["factor"] = 1.0

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print("Model loaded ✅")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 2 — Validate & report raw JSONL lengths
# ---------------------------------------------------------------------------

def fix_and_load(filepath, tokenizer):
    good_rows, bad_rows = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                good_rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_rows.append((i + 1, str(e), line[:100]))

    print(f"{filepath}: {len(good_rows)} good, {len(bad_rows)} bad rows")
    for row_num, err, preview in bad_rows:
        print(f"  Row {row_num}: {err} | Preview: {preview}")
    return good_rows


def report_lengths(all_rows, tokenizer):
    lengths = [len(tokenizer(r["text"])["input_ids"]) for r in all_rows]
    print(f"Max:     {max(lengths)}")
    print(f"Mean:    {int(sum(lengths) / len(lengths))}")
    print(f"95th %%: {sorted(lengths)[int(0.95 * len(lengths))]}")
    print(f"99th %%: {sorted(lengths)[int(0.99 * len(lengths))]}")


# ---------------------------------------------------------------------------
# Step 3 — Add special tokens
# ---------------------------------------------------------------------------

def add_special_tokens(model, tokenizer):
    special_tokens = {
        "additional_special_tokens": [
            "<actual response>",
            "</actual response>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Tokenizer vocab size: {len(tokenizer)}")


# ---------------------------------------------------------------------------
# Step 4 — Normalize & validate
# ---------------------------------------------------------------------------

def extract_tag(text, tag):
    try:
        return text.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()
    except IndexError:
        return None


def normalize_and_validate(filepath, output_path):
    good, bad = 0, 0
    with open(filepath, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ❌ Row {i+1}: JSON error — {e}")
                bad += 1
                continue

            text = row.get("text", "")
            medicine      = extract_tag(text, "medicine")
            context       = extract_tag(text, "context")
            user_msg      = extract_tag(text, "user")
            assistant_msg = extract_tag(text, "actual response") or extract_tag(text, "assistant")

            if not all([context, user_msg, assistant_msg]):
                missing = []
                if not context:       missing.append("context")
                if not user_msg:      missing.append("user")
                if not assistant_msg: missing.append("assistant")
                print(f"  ⚠️  Row {i+1}: Missing {missing} — skipping")
                bad += 1
                continue

            context       = re.sub(r'\n{2,}', '\n', context).strip()
            user_msg      = re.sub(r'\n{2,}', '\n', user_msg).strip()
            assistant_msg = re.sub(r'\n{2,}', '\n', assistant_msg).strip()
            if medicine:
                medicine  = re.sub(r'\n{2,}', '\n', medicine).strip()

            user_content = ""
            if medicine:
                user_content += f"{medicine}\n"
            user_content += f"<context>\n{context}\n</context>\n\n{user_msg}"

            row["_medicine"]     = medicine or ""
            row["_context"]      = context
            row["_user"]         = user_msg
            row["_assistant"]    = assistant_msg
            row["_user_content"] = user_content

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            good += 1

    print(f"{filepath} → {output_path}: ✅ {good} kept, ❌ {bad} dropped")


# ---------------------------------------------------------------------------
# Step 5 — Load cleaned files into a Dataset
# ---------------------------------------------------------------------------

def load_clean(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"{filepath}: {len(rows)} rows")
    return rows


# ---------------------------------------------------------------------------
# Step 6 — Tokenize
# ---------------------------------------------------------------------------

def make_preprocess_fn(tokenizer):
    def preprocess(example):
        conversation = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": example["_user_content"]},
            {"role": "assistant", "content": f"<actual response>\n{example['_assistant']}\n</actual response>"},
        ]

        full_text   = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(conversation[:-1], tokenize=False, add_generation_prompt=True)

        full_enc   = tokenizer(full_text,   truncation=True, max_length=MAX_LEN, padding=False)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_LEN, padding=False)

        full_ids   = full_enc["input_ids"]
        prompt_len = len(prompt_enc["input_ids"])

        labels = full_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids":      full_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels":         labels,
        }
    return preprocess


# ---------------------------------------------------------------------------
# Step 7 — Collator
# ---------------------------------------------------------------------------

@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features):
        input_ids      = [f["input_ids"]     for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels         = [f["labels"]         for f in features]

        max_len = max(len(x) for x in input_ids)

        def pad(seq, val):
            return seq + [val] * (max_len - len(seq))

        return {
            "input_ids":      torch.tensor([pad(x, self.tokenizer.pad_token_id) for x in input_ids]),
            "attention_mask": torch.tensor([pad(x, 0)    for x in attention_mask]),
            "labels":         torch.tensor([pad(x, -100) for x in labels]),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load model
    model, tokenizer = load_model_and_tokenizer()

    # 2. Inspect raw files
    all_raw = []
    for fp in INPUT_FILES:
        all_raw.extend(fix_and_load(fp, tokenizer))
    print(f"\nTotal usable raw rows: {len(all_raw)}")
    report_lengths(all_raw, tokenizer)

    # 3. Add special tokens
    add_special_tokens(model, tokenizer)

    # 4. Normalize & validate
    for src, dst in zip(INPUT_FILES, CLEAN_FILES):
        normalize_and_validate(src, dst)

    # 5. Build dataset
    all_rows = []
    for fp in CLEAN_FILES:
        all_rows.extend(load_clean(fp))
    print(f"Total rows: {len(all_rows)}")
    dataset = Dataset.from_list(all_rows)
    print("Dataset ready ✅")

    # 6. Tokenize
    tokenized = dataset.map(
        make_preprocess_fn(tokenizer),
        remove_columns=dataset.column_names,
        num_proc=1,
    )
    print("Tokenization done ✅")
    print(f"Sample input_ids length: {len(tokenized[0]['input_ids'])}")

    # 7. Train
    training_args = TrainingArguments(
        output_dir="./hft_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=440,
        bf16=True,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="checkpoint",
        hub_private_repo=True,
        logging_steps=50,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CausalLMCollator(tokenizer=tokenizer),
    )

    trainer.train()

    # 8. Push final model & tokenizer
    trainer.push_to_hub(commit_message="HFT full finetune - homeopathy 100k")
    tokenizer.push_to_hub(HUB_MODEL_ID)
    print("Done ✅")


if __name__ == "__main__":
    main()