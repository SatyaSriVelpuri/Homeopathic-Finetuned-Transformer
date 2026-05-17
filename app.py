


"""
Homeopathic Consultation Chat — app.py
Model: mnj-hf/homeopathy-chat (HuggingFace)

Requirements:
    pip install streamlit torch transformers accelerate huggingface_hub sentencepiece tokenizers
"""

import json
import os
import re
import streamlit as st
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Homeo Consult",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Nunito:wght@300;400;600&display=swap');

:root {
    --bg:       #F4F0E8;
    --surface:  #FFFDF8;
    --border:   #D9D3C3;
    --sage:     #5C7A52;
    --moss:     #3D5C35;
    --bark:     #7C5C3E;
    --gold:     #B8922A;
    --text:     #2C2A25;
    --muted:    #7A7060;
    --user-bg:  #EAF0E6;
    --bot-bg:   #FFFDF8;
    --shadow:   rgba(60,80,50,0.10);
}

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #2E4A27 0%, #1C3018 100%) !important;
    border-right: 1px solid #3D5C35;
}
[data-testid="stSidebar"] * { color: #DCE8D8 !important; }
[data-testid="stSidebar"] hr { border-color: #3D5C35 !important; opacity: 0.5; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid #4A6741 !important;
    color: #DCE8D8 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] label {
    color: #9CB896 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.app-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 0.2rem; }
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; font-weight: 700;
    color: var(--moss); line-height: 1.1;
}
.app-subtitle {
    font-family: 'Playfair Display', serif;
    font-style: italic; font-size: 1rem;
    color: var(--bark); margin-bottom: 1.4rem;
}

.chat-wrap { display: flex; flex-direction: column; gap: 10px; margin-bottom: 1rem; }
.bubble-row { display: flex; align-items: flex-end; gap: 10px; }
.bubble-row.user { flex-direction: row-reverse; }
.bubble-row.bot  { flex-direction: row; }
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar.user { background: var(--sage); color: white; }
.avatar.bot  { background: var(--bark); color: white; }
.bubble {
    max-width: 72%; padding: 0.75rem 1.1rem;
    border-radius: 16px; font-size: 0.94rem;
    line-height: 1.6; white-space: pre-wrap;
    box-shadow: 0 2px 8px var(--shadow);
    animation: popIn 0.25s ease both;
}
.bubble.user {
    background: var(--user-bg); border: 1px solid #C0D4BA;
    border-bottom-right-radius: 4px; color: var(--text);
}
.bubble.bot {
    background: var(--bot-bg); border: 1px solid var(--border);
    border-bottom-left-radius: 4px; color: var(--text);
}
@keyframes popIn {
    from { opacity: 0; transform: translateY(8px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.stTextInput input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
}
.stTextInput input:focus {
    border-color: var(--sage) !important;
    box-shadow: 0 0 0 3px rgba(92,122,82,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--moss), var(--sage)) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.5rem 1.4rem !important; transition: all 0.2s !important;
    box-shadow: 0 3px 10px rgba(60,90,50,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(60,90,50,0.3) !important;
}

.pill {
    display: inline-block; padding: 2px 12px;
    border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; letter-spacing: 0.04em;
}
.pill-green { background: #D4EAD0; color: #2E5A28; border: 1px solid #A8CFA2; }
.pill-red   { background: #FAE0DC; color: #8B2A20; border: 1px solid #E8B0A8; }
.pill-amber { background: #FDF0D0; color: #7A5010; border: 1px solid #E8D090; }

.disclaimer {
    background: #FFF8EC; border: 1px solid #E8C97A;
    border-radius: 10px; padding: 0.75rem 1.1rem;
    font-size: 0.78rem; color: #7A6020;
    margin-top: 1rem; line-height: 1.5;
}
.section-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.3rem; margin-top: 0.8rem;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
HF_REPO_ID     = "mnj-hf/homeopathy-chat"
LOCAL_DIR      = Path("./homeopathy_model")
MAX_NEW_TOKENS = 220
MEMORY_TURNS   = 4

# All files we need — INCLUDING tokenizer.json and tokenizer_config.json
# We will PATCH tokenizer.json after download to fix the ModelWrapper error.
MODEL_FILES = [
    "config.json",
    "config_parambharatgen.py",
    "modeling_parambharatgen.py",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]

SYSTEM_PROMPT_BASE = (
    "You are a clinical decision-support assistant for qualified homeopathic medical practitioners.\n"
    "Your role is to help practitioners think through cases by asking focused clarifying questions "
    "one at a time, summarize repertory, compare remedy options, and highlight red-flag symptoms "
    "that require conventional medical evaluation.\n"
    "Base your reasoning on classical and widely accepted homeopathic sources and general medical knowledge."
)

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER.JSON PATCHER
# Root cause: parambharatgen saves tokenizer.json with a custom "type" value
# in the "model" block (e.g. "ParamBPE", "CustomBPE", or a namespaced string).
# The `tokenizers` Rust library only accepts: BPE | WordPiece | WordLevel | Unigram.
# Fix: read tokenizer.json, rename the unknown model type to "BPE", write back.
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_MODEL_TYPES = {"BPE", "WordPiece", "WordLevel", "Unigram"}

def patch_tokenizer_json(tok_path: Path) -> str:
    """
    Patch tokenizer.json so the 'model.type' field is one the tokenizers
    library recognises.  Writes the result back in-place so the fix persists
    across Streamlit cache hits and server restarts.
    Returns a status string for logging.
    """
    if not tok_path.exists():
        return "tokenizer.json not found — skipping patch."

    raw = tok_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    model_block = data.get("model", {})
    current_type = model_block.get("type", "")

    if current_type in KNOWN_MODEL_TYPES:
        return f"tokenizer.json already has known type '{current_type}' — no patch needed."

    # parambharatgen is LLaMA-style BPE: 'merges' list is the tell.
    if "merges" in model_block:
        new_type = "BPE"
    elif "vocab" in model_block and "unk_token" in model_block:
        new_type = "WordLevel"
    elif "vocab" in model_block:
        new_type = "BPE"   # safest fallback for LLaMA descendants
    else:
        new_type = "BPE"

    model_block["type"] = new_type
    data["model"] = model_block

    # Write back in-place — this survives Streamlit cache reuse and restarts
    tok_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    return f"Patched tokenizer.json in-place: '{current_type}' → '{new_type}'"


# ── Eagerly patch on every startup (cheap if already patched) ─────────────────
# Must run BEFORE load_model() is ever called, because @st.cache_resource means
# load_model body only executes once — the patch would be skipped on all
# subsequent Streamlit reruns if placed only inside load_model().
# _tok_json_path = LOCAL_DIR / "tokenizer.json"
# _patch_log = patch_tokenizer_json(_tok_json_path)
# print("[tokenizer patch]", _patch_log)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_model_if_needed():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in MODEL_FILES if not (LOCAL_DIR / f).exists()]
    if not missing:
        return True, "All files already present."

    hf_token = os.environ.get("HF_TOKEN", None)

    progress = st.sidebar.progress(0, text="Downloading model files…")
    try:
        for i, fname in enumerate(missing):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                local_dir=str(LOCAL_DIR),
                token=hf_token,
            )
            progress.progress((i + 1) / len(missing), text=f"↓ {fname}")
        progress.empty()
        return True, "Download complete."
    except Exception as e:
        progress.empty()
        return False, f"Download failed: {e}"


# ## ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    ok, msg = download_model_if_needed()
    if not ok:
        return None, None, msg

    try:
        # ─────────────────────────────────────────────────────
        # PATCH tokenizer.json AFTER download
        # ─────────────────────────────────────────────────────
        tok_path = LOCAL_DIR / "tokenizer.json"

        patch_msg = patch_tokenizer_json(tok_path)
        print("[tokenizer patch]", patch_msg)

        # ─────────────────────────────────────────────────────
        # LOAD TOKENIZER
        # ─────────────────────────────────────────────────────
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(LOCAL_DIR),
            trust_remote_code=True,
            use_fast=False
        )   

        # Apply overrides from tokenizer_config.json
        tok_config_path = LOCAL_DIR / "tokenizer_config.json"

        if tok_config_path.exists():
            tok_cfg = json.loads(
                tok_config_path.read_text(encoding="utf-8")
            )

            # Chat template
            chat_template_path = LOCAL_DIR / "chat_template.jinja"

            if chat_template_path.exists():
                tokenizer.chat_template = (
                    chat_template_path.read_text(encoding="utf-8")
                )

            elif "chat_template" in tok_cfg:
                tokenizer.chat_template = tok_cfg["chat_template"]

            # Special tokens
            special_map = {}

            for key in (
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
                "sep_token",
                "cls_token",
                "mask_token",
            ):
                val = tok_cfg.get(key)

                if val:

                    if isinstance(val, dict):
                        val = val.get("content", "")

                    if val:
                        special_map[key] = val

            if special_map:
                tokenizer.add_special_tokens(special_map)

        # Add extra tokens
        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<actual response>",
                "</actual response>",
            ]
        })

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ─────────────────────────────────────────────────────
        # LOAD MODEL
        # ─────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_DIR),
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        model.to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        return model, tokenizer, "ok"

    except Exception as e:
        import traceback
        traceback.print_exc()

        return None, None, f"error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
def build_system_prompt(medicine: str) -> str:
    if medicine.strip():
        return f"<medicine>\n{medicine.strip()}\n</medicine>\n\n{SYSTEM_PROMPT_BASE}"
    return SYSTEM_PROMPT_BASE


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────────────────────
def generate_reply(model, tokenizer, chat_history: list, medicine: str) -> str:
    system_msg = {"role": "system", "content": build_system_prompt(medicine)}
    window = chat_history[-(MEMORY_TURNS * 2):]
    messages = [system_msg] + window

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    for tag in ["</actual response>", "</assistant>", "<|endoftext|>"]:
        if tag in text:
            text = text[:text.index(tag)]

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "medicine" not in st.session_state:
    st.session_state.medicine = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Session Setup")
    st.markdown("---")

    st.markdown('<div class="section-label">Medicine (optional)</div>', unsafe_allow_html=True)
    medicine_input = st.text_input(
        label="medicine",
        placeholder="e.g. Iberis Amara",
        value=st.session_state.medicine,
        label_visibility="collapsed",
    )
    st.caption("Leave blank if you want the model to determine the remedy through questioning.")

    if medicine_input != st.session_state.medicine:
        st.session_state.medicine = medicine_input

    st.markdown("---")

    st.markdown("**Model Status**")
    with st.spinner("Loading model…"):
        model, tokenizer, status = load_model()

    if status == "ok":
        st.markdown('<span class="pill pill-green">✅ Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-red">❌ Not loaded</span>', unsafe_allow_html=True)
        st.caption(status)

    st.markdown("---")

    turns_in_memory = len(st.session_state.chat_history) // 2
    active_turns = min(turns_in_memory, MEMORY_TURNS)
    st.markdown(f"**Memory:** {active_turns}/{MEMORY_TURNS} turns active")
    if turns_in_memory > MEMORY_TURNS:
        st.caption(f"Oldest {turns_in_memory - MEMORY_TURNS} turn(s) dropped from context.")

    st.markdown("---")

    col_new, col_clear = st.sidebar.columns(2)
    with col_new:
        if st.button("New Case", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.medicine = ""
            st.session_state.input_key += 1
            st.rerun()
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.input_key += 1
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚕️ For qualified homeopathic practitioners only. '
        'Always exercise clinical judgement. Not a substitute for professional medical advice.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header"><div class="app-title">🌿 Homeo Consult</div></div>'
    '<div class="app-subtitle">Clinical decision support for homeopathic practitioners</div>',
    unsafe_allow_html=True,
)

if st.session_state.medicine.strip():
    st.markdown(
        f'<span class="pill pill-amber">💊 Medicine context: {st.session_state.medicine}</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")

if not st.session_state.chat_history:
    st.markdown(
        '<div style="text-align:center; color:#9A9080; padding: 2rem 0; '
        'font-style: italic; font-size: 0.95rem;">'
        'Start by describing the patient\'s chief complaint below.<br>'
        'The model will ask clarifying questions to reach a remedy.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        avatar = "👤" if role == "user" else "🌿"
        st.markdown(
            f'<div class="bubble-row {role}">'
            f'  <div class="avatar {role}">{avatar}</div>'
            f'  <div class="bubble {role}">{content}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
col_input, col_send = st.columns([5, 1])

with col_input:
    user_text = st.text_input(
        label="message",
        placeholder="Describe symptoms or reply to the model's question…",
        label_visibility="collapsed",
        key=f"user_input_{st.session_state.input_key}",
    )

with col_send:
    send_clicked = st.button("Send →", use_container_width=True)

if send_clicked and user_text.strip():
    if status != "ok":
        st.error("Model is not loaded. Check the sidebar for details.")
    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_text.strip()}
        )

        with st.spinner("Thinking…"):
            reply = generate_reply(
                model,
                tokenizer,
                st.session_state.chat_history,
                st.session_state.medicine,
            )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": reply}
        )

        st.session_state.input_key += 1
        st.rerun()


