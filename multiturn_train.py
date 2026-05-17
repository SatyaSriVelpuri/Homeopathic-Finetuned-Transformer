"""
Multi-turn dialogue fine-tuning script.
Continues training from mnj-hf/homeopathy (already fine-tuned on Q&A pairs)
using multi-turn dialogue data to teach conversational behaviour.

Requirements:
    pip install torch transformers datasets accelerate
"""

import json
import os
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

# Load from your already fine-tuned model, not the base model
MODEL_ID     = "mnj-hf/homeopathy"
HUB_MODEL_ID = "mnj-hf/homeopathy-chat"   # push to a new repo to keep the QA model safe
DIALOGUE_FILE = "final_multiturn_dialogues.jsonl"
MAX_LEN = 2048   # dialogues are longer than single QA pairs


# ---------------------------------------------------------------------------
# Step 1 — Load fine-tuned model & tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print("Loading fine-tuned model from mnj-hf/homeopathy ...")

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
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Re-add the special tokens that were added during QA fine-tuning
    special_tokens = {
        "additional_special_tokens": [
            "<actual response>",
            "</actual response>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Model loaded ✅  |  Vocab size: {len(tokenizer)}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Step 2 — Load & validate dialogue file
# ---------------------------------------------------------------------------

def load_dialogues(filepath):
    rows, bad = [], 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                # Must have at least system + 1 user + 1 assistant
                if len(msgs) < 3:
                    print(f"  ⚠️  Row {i+1}: too few messages ({len(msgs)}) — skipping")
                    bad += 1
                    continue
                rows.append(obj)
            except json.JSONDecodeError as e:
                print(f"  ❌ Row {i+1}: JSON error — {e}")
                bad += 1

    print(f"{filepath}: ✅ {len(rows)} kept, ❌ {bad} dropped")
    return rows


# ---------------------------------------------------------------------------
# Step 3 — Tokenize with per-turn loss masking
#
# Strategy: build the full conversation with apply_chat_template, then
# for each assistant turn find its token span and unmask only those tokens.
# Everything else (system, user turns) stays masked as -100.
# ---------------------------------------------------------------------------

def make_multiturn_preprocess_fn(tokenizer):
    def preprocess(example):
        messages = example["messages"]

        # Full conversation tokenized
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )["input_ids"]

        # Start with all labels masked
        labels = [-100] * len(full_ids)

        # For each assistant turn, find its span by comparing prefix lengths
        # We walk turn-by-turn, building up the prefix to find where each
        # assistant reply starts and ends.
        for turn_idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Prefix = everything up to (but not including) this assistant turn
            prefix_msgs = messages[:turn_idx]
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs,
                tokenize=False,
                add_generation_prompt=True,   # adds the cue that starts assistant reply
            )
            prefix_ids = tokenizer(
                prefix_text,
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )["input_ids"]

            # Prefix up to and including this assistant turn
            suffix_msgs = messages[:turn_idx + 1]
            suffix_text = tokenizer.apply_chat_template(
                suffix_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            suffix_ids = tokenizer(
                suffix_text,
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )["input_ids"]

            start = len(prefix_ids)
            end   = len(suffix_ids)

            # Unmask the assistant tokens within the full sequence bounds
            for pos in range(start, min(end, len(labels))):
                labels[pos] = full_ids[pos]

        # Skip examples where every label is masked (truncation ate all assistant content)
        if all(l == -100 for l in labels):
            # Return a dummy that the collator will discard via -100 labels
            return {
                "input_ids":      full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels":         labels,
            }

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }

    return preprocess


# ---------------------------------------------------------------------------
# Step 4 — Collator
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — cannot run on device 0.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)} (cuda:0)")

    # 1. Load model
    model, tokenizer = load_model_and_tokenizer()

    # 2. Load dialogues
    rows = load_dialogues(DIALOGUE_FILE)
    print(f"Total dialogues: {len(rows)}")

    dataset = Dataset.from_list(rows)

    # 3. Tokenize with multi-turn masking
    tokenized = dataset.map(
        make_multiturn_preprocess_fn(tokenizer),
        remove_columns=dataset.column_names,
        num_proc=1,   # keep at 1 — tokenizer is not fork-safe
    )
    print("Tokenization done ✅")
    print(f"Sample input_ids length: {len(tokenized[0]['input_ids'])}")

    # 4. Train
    # Lower LR (5e-6 vs 2e-5) and fewer epochs (2) because:
    #   - we're continuing from an already fine-tuned model
    #   - dataset is small (204 samples) — risk of overfitting with more
    training_args = TrainingArguments(
        output_dir="./chat_finetuned",
        num_train_epochs=2,
        per_device_train_batch_size=4,       # smaller batch — dialogues are longer
        gradient_accumulation_steps=8,        # effective batch = 32
        learning_rate=5e-6,                   # lower LR — continuing fine-tune
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="end",                   # only push at the end — 204 samples is small
        hub_private_repo=True,
        logging_steps=10,
        dataloader_num_workers=2,
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

    # 5. Push final model & tokenizer
    trainer.push_to_hub(commit_message="Multi-turn dialogue fine-tune on homeopathy chat data")
    tokenizer.push_to_hub(HUB_MODEL_ID)
    print("Done ✅")
    print(f"Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")


if __name__ == "__main__":
    main()