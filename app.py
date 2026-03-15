import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import random
from datetime import date, datetime, timedelta
import google.generativeai as genai
import json
import re
import time
import os
import tempfile
import hashlib
import concurrent.futures
import math
from typing import TypedDict, List as TList

# ── Tenacity (Enhancement #5) ─────────────────────────────────────────────────
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# ── Audio & Anki ──────────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

# --- CSS VARIABLES ---
THEME_COLOR = "#00ff41"
THEME_GLOW  = "rgba(0, 255, 65, 0.4)"
BG_COLOR    = "#111111"
BG_STRIPE   = "#181818"
TEXT_COLOR  = "#aaffaa"

# ========================== MOBILE KEYBOARD FIX ==========================
js_hide_keyboard = """
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
        setTimeout(() => { e.target.blur(); }, 50);
    }
}, true);
</script>
"""
st.components.v1.html(js_hide_keyboard, height=0)

# ========================== RESPONSE SCHEMA (Enhancement #17) ==========================
# TypedDicts for structured Gemini output — with graceful fallback if SDK too old.
class _SynAnt(TypedDict):
    synonyms: TList[str]
    antonyms: TList[str]

class _CardData(TypedDict):
    vocab: str
    translation: str
    part_of_speech: str
    pronunciation_ipa: str
    definition_english: str
    example_sentences: TList[str]
    synonyms_antonyms: _SynAnt
    etymology: str

try:
    CARD_RESPONSE_SCHEMA = list[_CardData]
    SCHEMA_AVAILABLE = True
except Exception:
    CARD_RESPONSE_SCHEMA = None
    SCHEMA_AVAILABLE = False

# ========================== SYSTEM INSTRUCTION (Enhancement #16) ==========================
# Static rules + few-shots live here → ~40% fewer tokens per user-turn request.
SYSTEM_INSTRUCTION = """You are an expert educational lexicographer creating Anki flashcard content.

Think step-by-step:
1. Identify the primary sense of the vocabulary word from the phrase/context provided.
2. Generate accurate, educational linguistic data for every field.
3. Ensure your response is a valid JSON array containing EXACTLY the number of items requested.

OUTPUT FORMAT: Respond ONLY with a valid JSON array. No preamble, no commentary, no markdown fences.

SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.

FIELD RULES:
1. Copy ALL input fields exactly as provided.
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT for the word's meaning/usage.
3. IF 'phrase' is normal text: Use it ONLY to identify which meaning/nuance of 'vocab' applies.
4. IF 'phrase' is empty: Generate ONE simple example sentence (max 12 words) for the vocab.
5. The EXACT 'vocab' field value must remain unchanged in your output.
6. 'part_of_speech' MUST be one of: Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection, Phrase.
7. 'example_sentences': provide up to 3 varied example sentences.
8. 'synonyms_antonyms': provide up to 5 synonyms and 5 antonyms.
9. 'etymology': brief origin note (1–2 sentences max).

FEW-SHOT EXAMPLES:
[
  {"vocab": "serendipity", "phrase": "We found the perfect cafe by pure serendipity.", "translation": "kebetulan", "part_of_speech": "Noun", "pronunciation_ipa": "/\\u02ccs\\u025br\\u0259n\\u02c8d\\u026ap\\u026ati/", "definition_english": "The occurrence and development of events by chance in a happy or beneficial way.", "example_sentences": ["It was pure serendipity that we met.", "His discovery was a happy serendipity.", "Serendipity led her to the perfect job."], "synonyms_antonyms": {"synonyms": ["chance", "luck", "fortune", "fluke", "coincidence"], "antonyms": ["misfortune", "design", "intention", "plan", "deliberation"]}, "etymology": "Coined by Horace Walpole in 1754, from the Persian fairy tale 'The Three Princes of Serendip'."},
  {"vocab": "ephemeral", "phrase": "", "translation": "sementara", "part_of_speech": "Adjective", "pronunciation_ipa": "/\\u026a\\u02c8f\\u025bm\\u0259r\\u0259l/", "definition_english": "Lasting for a very short time.", "example_sentences": ["The ephemeral beauty of the sunset faded quickly.", "Social media trends are often ephemeral.", "Morning dew is an ephemeral phenomenon."], "synonyms_antonyms": {"synonyms": ["transient", "fleeting", "temporary", "brief", "momentary"], "antonyms": ["permanent", "lasting", "eternal", "enduring", "perpetual"]}, "etymology": "From Greek ephemeros, literally meaning 'lasting only a day', from epi- (on) + hemera (day)."},
  {"vocab": "run", "phrase": "*He decided to run for office", "translation": "mencalonkan diri", "part_of_speech": "Verb", "pronunciation_ipa": "/r\\u028cn/", "definition_english": "To compete as a candidate in an election or for a position.", "example_sentences": ["She decided to run for city council.", "He announced he would run for president.", "Three candidates are running for the seat."], "synonyms_antonyms": {"synonyms": ["campaign", "stand", "contest", "seek election", "bid"], "antonyms": ["withdraw", "abstain", "concede", "step down", "retire"]}, "etymology": "Old English rinnan, of Germanic origin; the political sense emerged in 19th-century American English."},
  {"vocab": "placebo", "phrase": "The placebo effect was strong in the trial.", "translation": "plasebo", "part_of_speech": "Noun", "pronunciation_ipa": "/pl\\u0259\\u02c8si\\u02d0bo\\u028a/", "definition_english": "A substance with no therapeutic effect used as a control in clinical trials, or to satisfy a patient who expects treatment.", "example_sentences": ["Patients in the control group received a placebo.", "The placebo effect can produce measurable physical responses.", "Researchers use placebos to eliminate bias in drug studies."], "synonyms_antonyms": {"synonyms": ["dummy", "sugar pill", "control", "sham treatment", "inert substance"], "antonyms": []}, "etymology": "From Latin placebo meaning 'I shall please'; first used in a medical context in the 18th century."}
]"""

# ========================== SECRETS ==========================
try:
    token           = st.secrets["GITHUB_TOKEN"]
    repo_name       = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

# ========================== GITHUB CONNECT ==========================
@st.cache_resource
def get_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except GithubException as e:
        st.error(f"❌ GitHub connection failed: {e}")
        st.stop()

repo = get_repo()

# ========================== PERSISTENT API QUOTA TRACKING ==========================
def load_usage():
    try:
        file = repo.get_contents("usage.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        if data.get("date") == str(date.today()):
            return data.get("rpd_count", 0)
        return 0
    except:
        return 0

def save_usage(count):
    data = json.dumps({"date": str(date.today()), "rpd_count": count})
    try:
        file = repo.get_contents("usage.json")
        repo.update_file(file.path, "Update API usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage.json", "Init API usage", data)

def load_minute_usage():
    try:
        file = repo.get_contents("usage_minute.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return [datetime.fromisoformat(ts) for ts in data.get("timestamps", [])]
    except:
        return []

def save_minute_usage(timestamps):
    ts_list = [ts.isoformat() for ts in timestamps]
    data = json.dumps({"timestamps": ts_list})
    try:
        file = repo.get_contents("usage_minute.json")
        repo.update_file(file.path, "Update minute usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage_minute.json", "Init minute usage", data)

if "rpd_count"        not in st.session_state: st.session_state.rpd_count        = load_usage()
if "rpm_timestamps"   not in st.session_state: st.session_state.rpm_timestamps   = load_minute_usage()
if "rpm_date"         not in st.session_state: st.session_state.rpm_date         = str(date.today())

def enforce_rpm():
    """Rate-limit guard: 5 RPM + midnight RPD reset. NO GitHub I/O (Enhancement #4/#6)."""
    # Enhancement #6: Midnight RPD auto-reset
    today_str = str(date.today())
    if st.session_state.rpm_date != today_str:
        st.session_state.rpd_count     = 0
        st.session_state.rpm_timestamps = []
        st.session_state.rpm_date      = today_str
        save_usage(0)

    now = datetime.now()
    st.session_state.rpm_timestamps = [
        ts for ts in st.session_state.rpm_timestamps
        if (now - ts).total_seconds() < 60
    ]
    if len(st.session_state.rpm_timestamps) >= 5:
        for remaining in range(12, 0, -1):
            st.warning(f"⏳ RPM limit reached (5/min). Waiting {remaining}s...")
            time.sleep(1)
    st.session_state.rpm_timestamps.append(datetime.now())
    # ✅ No save_minute_usage() here — called once per batch run to prevent GitHub I/O storm.

# ========================== GEMINI MODEL (system_instruction + response_schema) ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    """Enhancements #16 (system_instruction) + #17 (response_schema with fallback)."""
    try:
        genai.configure(api_key=api_key)
        # Try with structured response_schema first
        try:
            return genai.GenerativeModel(
                model_name,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=CARD_RESPONSE_SCHEMA,
                    temperature=0.1
                )
            )
        except Exception:
            # Graceful fallback: no schema, but keep system_instruction
            return genai.GenerativeModel(
                model_name,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={"response_mime_type": "application/json", "temperature": 0.1}
            )
    except Exception as e:
        st.error(f"❌ Gemini key error: {e}")
        return None

# ========================== API KEY VALIDATOR (Enhancement #13) ==========================
def validate_api_key(api_key: str, model_name: str) -> bool:
    """Run once per session/key change. Uses count_tokens (no quota cost)."""
    cache_key = f"{api_key}_{model_name}"
    if st.session_state.get("api_key_validated_key") == cache_key:
        return st.session_state.get("api_key_valid", False)
    try:
        genai.configure(api_key=api_key)
        probe = genai.GenerativeModel(model_name)
        probe.count_tokens("validate")
        st.session_state.api_key_validated_key = cache_key
        st.session_state.api_key_valid         = True
        return True
    except Exception as e:
        st.session_state.api_key_valid = False
        st.session_state.api_key_validated_key = cache_key
        return False

# ========================== TENACITY RETRY WRAPPER (Enhancement #5) ==========================
def _is_rate_limit(exc: BaseException) -> bool:
    return "429" in str(exc)

# Hard wall-clock timeout — free tier can hang indefinitely without this.
GEMINI_TIMEOUT_SEC = 45

def _raw_gemini_call(model, prompt: str):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(model.generate_content, prompt)
        try:
            return future.result(timeout=GEMINI_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Gemini did not respond within {GEMINI_TIMEOUT_SEC}s — "
                "free-tier queue may be overloaded. Please retry."
            )

if TENACITY_AVAILABLE:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=20, max=60),
        retry=retry_if_exception(_is_rate_limit),   # ONLY retries on 429, not timeouts
        reraise=True
    )
    def _call_gemini(model, prompt: str):
        return _raw_gemini_call(model, prompt)
else:
    def _call_gemini(model, prompt: str):
        return _raw_gemini_call(model, prompt)

# ========================== CLEANING FUNCTIONS ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if s and s[-1] in ".!?" else (s + "." if s else "")

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() if text else ""

def clean_grammar(text: str) -> str:
    if not isinstance(text, str): return text
    rules = [
        (r"\bto doing\b", "to do"), (r"\bfor helps\b", "to help"),
        (r"\bis use to\b", "is used to"), (r"\bhelp for to\b", "help to"),
        (r"\bfor to\b", "to"), (r"\bcan able to\b", "can"),
        (r"\bI am agree\b", "I agree"), (r"\bdiscuss about\b", "discuss"),
        (r"\breturn back\b", "return")
    ]
    for pat, repl in rules:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def cap_each_sentence(text: str) -> str:
    if not isinstance(text, str): return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join([cap_first(s) for s in sentences if s.strip()])

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    pattern = r'\b' + re.escape(vocab) + r'\b'
    return re.sub(pattern, f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    pattern = r'\b' + re.escape(vocab.lower()) + r'\b'
    return re.sub(pattern, vocab, phrase, flags=re.IGNORECASE)

def robust_json_parse(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try: return json.loads(text)
    except: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== SINGLE-ITEM FALLBACK GENERATOR (Enhancement #7) ==========================
def _generate_single_item(model, vocab: str, phrase: str, target_lang: str) -> dict | None:
    """Fallback: generate data for one item when a full batch exhausts all retries."""
    prompt = (
        f"MANDATORY: 'translation' field must contain ONLY the {target_lang} translation "
        f"of the vocab word. NEVER translate the full sentence.\n\n"
        f"Output EXACTLY 1 item as a JSON array.\n\n"
        f"BATCH INPUT: {json.dumps([{'vocab': vocab, 'phrase': phrase}], ensure_ascii=False)}"
    )
    try:
        response = _call_gemini(model, prompt)
        parsed = robust_json_parse(response.text)
        if isinstance(parsed, list) and len(parsed) >= 1:
            return parsed[0]
    except Exception:
        pass
    return None

# ========================== ASYNC BATCH GENERATOR ==========================
# Time breakdown per batch (realistic):
#   ~10–15s  Gemini API latency
#   ~3–5s    gTTS audio (threaded, after this function)
#   ~3–4s    ONE GitHub write (save_usage) at the END — not inside loop
#   ─────────────────────────────────────
#   ~15–20s  per batch total (12s is just the RPM floor for burst-of-5)
_SEC_PER_BATCH = 20   # realistic ETA constant used in display

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6, dry_run=False, target_lang="Indonesian"):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []

    all_card_data     = []
    batches           = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    # Track how many new RPD calls we make; write to GitHub ONCE at the end.
    rpd_delta         = 0

    with st.status("🤖 Processing AI Batches...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        eta_slot     = st.empty()

        # Realistic ETA: Gemini latency dominates, not the RPM floor
        est_total_sec = len(batches) * _SEC_PER_BATCH
        eta_slot.info(
            f"⏱️ Est. time: ~{est_total_sec}s "
            f"({len(batches)} batch{'es' if len(batches) != 1 else ''} × ~{_SEC_PER_BATCH}s each)"
        )

        t_start = time.time()

        for idx, batch in enumerate(batches):
            if st.session_state.rpd_count + rpd_delta >= 20:
                st.warning("🛑 Daily AI Limit (20 requests) reached. Please try again tomorrow.")
                break

            enforce_rpm()   # in-memory only — no GitHub write here

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            vocab_words = [v[0] for v in batch]

            # Enhancement #16: Lean user-turn prompt (static rules live in system_instruction)
            prompt = (
                f"MANDATORY: 'translation' field must contain ONLY the {target_lang} translation "
                f"of the vocab word. NEVER translate the full sentence.\n\n"
                f"Output EXACTLY {len(batch_dicts)} items.\n\n"
                f"BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"
            )

            success   = False
            step_slot = st.empty()   # live step label — tells user exactly which phase is running

            if dry_run:
                step_slot.info(f"🔬 Dry-run: {', '.join(vocab_words)}")
                mock_data = [
                    {"vocab": v[0], "phrase": v[1], "translation": "mock-" + v[0],
                     "part_of_speech": "Noun", "pronunciation_ipa": "/mock/",
                     "definition_english": "Simulated definition.",
                     "example_sentences": ["Mock example sentence."],
                     "synonyms_antonyms": {"synonyms": ["mock"], "antonyms": []},
                     "etymology": "Simulated."}
                    for v in batch
                ]
                all_card_data.extend(mock_data)
                success = True
            else:
                try:
                    t_call = time.time()
                    step_slot.info(f"🌐 Batch {idx+1}/{len(batches)}: calling Gemini API… (max {GEMINI_TIMEOUT_SEC}s)")
                    response = _call_gemini(model, prompt)   # hard timeout inside
                    call_ms  = int((time.time() - t_call) * 1000)
                    rpd_delta += 1                           # count locally — one GitHub write at end
                    step_slot.info(f"🔍 Parsing response… (API took {call_ms}ms)")
                    parsed = robust_json_parse(response.text)
                    if isinstance(parsed, list) and len(parsed) == len(batch_dicts):
                        all_card_data.extend(parsed)
                        step_slot.success(f"✅ **Processed**: `{', '.join(vocab_words)}` ({call_ms}ms)")
                        success = True
                    elif isinstance(parsed, list) and parsed:
                        all_card_data.extend(parsed)
                        step_slot.warning(f"⚠️ Got {len(parsed)}/{len(batch_dicts)} items — accepting partial.")
                        success = True
                except TimeoutError as e:
                    step_slot.error(f"⏰ Timeout: {e}")
                except Exception as e:
                    step_slot.warning(f"⚠️ Batch failed: {e}")

                # Enhancement #7: Single-item fallback when full batch fails
                if not success:
                    st.warning(f"🔁 Retrying {len(batch)} items individually...")
                    for single in batch:
                        if st.session_state.rpd_count + rpd_delta >= 20: break
                        enforce_rpm()
                        result = _generate_single_item(model, single[0], single[1], target_lang)
                        if result:
                            all_card_data.append(result)
                            rpd_delta += 1                   # ← still local, no GitHub write
                            st.markdown(f"  ↳ ✅ Recovered: `{single[0]}`")
                        else:
                            st.error(f"  ↳ ❌ Could not recover `{single[0]}`")

            # Enhancement #12: Live ETA using actual elapsed time
            elapsed   = time.time() - t_start
            done      = idx + 1
            remaining = len(batches) - done
            if remaining > 0 and done > 0:
                avg_sec   = elapsed / done
                eta_sec   = int(avg_sec * remaining)
                eta_slot.info(f"⏱️ ~{eta_sec}s remaining | Batch {done}/{len(batches)} done")
            elif remaining == 0:
                eta_slot.success(f"✅ All {len(batches)} batches done in {int(elapsed)}s.")

            progress_bar.progress(done / len(batches))

        # ── ONE GitHub write for the entire run ──────────────────────────────
        if rpd_delta > 0 and not dry_run:
            st.session_state.rpd_count += rpd_delta
            save_usage(st.session_state.rpd_count)          # single GitHub write

        # ── ONE GitHub write for minute timestamps ────────────────────────────
        save_minute_usage(st.session_state.rpm_timestamps)  # single GitHub write

        status_log.update(
            label=f"✅ AI Generation Complete! ({len(all_card_data)} items)",
            state="complete", expanded=False
        )

    return all_card_data

# ========================== PROCESS ANKI DATA ==========================
def process_anki_data(df_subset, batch_size=6, dry_run=False, target_lang="Indonesian"):
    # D15: 5-minute cache check
    cache_key = hashlib.sha256(str(df_subset.to_dict()).encode()).hexdigest()
    cached = st.session_state.get("processed_cache", {})
    if (cached.get("key") == cache_key and
            (datetime.now() - cached.get("time", datetime.min)).total_seconds() < 300):
        st.info("♻️ Using cached processed notes (no re-generation needed).")
        return cached["notes"]

    df_subset = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(
        vocab_phrase_list, batch_size=batch_size, dry_run=dry_run, target_lang=target_lang
    )
    processed_notes = []

    for card_data in all_card_data:
        required = ["vocab", "translation", "part_of_speech"]
        if not all(k in card_data and card_data[k] for k in required):
            st.error(f"⚠️ Missing required fields for '{card_data.get('vocab','?')}' — skipping.")
            continue

        vocab_raw  = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue
        vocab_cap  = cap_first(vocab_raw)

        phrase     = normalize_spaces(card_data.get("phrase", ""))
        phrase     = clean_grammar(phrase)
        phrase     = cap_each_sentence(phrase)
        phrase     = ensure_trailing_dot(phrase)
        phrase     = fix_vocab_casing(phrase, vocab_raw)
        fmt_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""

        translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("translation", "?"))))
        pos         = str(card_data.get("part_of_speech", "")).title()
        ipa         = card_data.get("pronunciation_ipa", "")
        eng_def     = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))

        examples    = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e))))
                       for e in (card_data.get("example_sentences", []) or [])[:3]]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""

        syn_ant    = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ensure_trailing_dot(", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]]))
        antonyms_field = ensure_trailing_dot(", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])[:5]]))
        etymology  = normalize_spaces(card_data.get("etymology", ""))

        text_field = (
            f"{fmt_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
            if fmt_phrase else
            f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        )
        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"

        # Enhancement #9: Auto-tags from POS + target language
        tags = []
        if pos:
            tags.append(pos.replace(" ", "_"))
        if target_lang:
            tags.append(target_lang.replace(" ", "_").replace("(", "").replace(")", ""))

        processed_notes.append({
            "VocabRaw":     vocab_raw,
            "Text":         text_field,
            "Pronunciation": pronunciation_field,
            "Definition":   eng_def,
            "Examples":     examples_field,
            "Synonyms":     synonyms_field,
            "Antonyms":     antonyms_field,
            "Etymology":    etymology,
            "Tags":         tags,
        })

    st.session_state.processed_cache = {"key": cache_key, "notes": processed_notes, "time": datetime.now()}
    return processed_notes

# ========================== LIVE CARD PREVIEW (Enhancement #8) ==========================
# CSS must be defined before this function is called (it's referenced below).
CYBERPUNK_CSS = f"""
.card {{ font-family: 'Roboto Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; color: {THEME_COLOR}; background-color: {BG_COLOR}; background-image: repeating-linear-gradient(0deg, {BG_STRIPE}, {BG_STRIPE} 1px, {BG_COLOR} 1px, {BG_COLOR} 20px); padding: 30px 20px; text-align: left; }}
.vellum-focus-container {{ background: #0d0d0d; padding: 30px 20px; margin: 0 auto 20px; border: 2px solid {THEME_COLOR}; box-shadow: 0 0 5px {THEME_COLOR}, 0 0 15px {THEME_GLOW}; text-align: center; }}
.prompt-text {{ font-family: 'Electrolize', sans-serif; font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }}
.cloze {{ color: {BG_COLOR}; background-color: {THEME_COLOR}; padding: 2px 4px; }}
.solved-text .cloze {{ color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }}
.vellum-section {{ margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed {THEME_COLOR}; }}
.section-header {{ font-weight: 600; color: #00ffff; border-left: 3px solid {THEME_COLOR}; padding-left: 10px; }}
.content {{ color: {TEXT_COLOR}; padding-left: 13px; }}
.vellum-detail-container {{ padding: 0 20px; }}
.preview-label {{ font-family: monospace; font-size: 0.7em; color: #444; padding: 4px 20px; letter-spacing: 2px; }}
@media (max-width: 480px) {{ .card {{ font-size: 16px; padding: 15px; }} .vellum-focus-container {{ padding: 15px; }} }}
"""

def render_card_preview(note: dict) -> str:
    """Render a single Anki card as HTML using CYBERPUNK_CSS for in-app preview."""
    text = note.get("Text", "")
    front = re.sub(r'\{\{c\d+::(.*?)(?::.*?)?\}\}',
                   '<span class="cloze">[?]</span>', text)
    back  = re.sub(r'\{\{c\d+::(.*?)(?::.*?)?\}\}',
                   r'<span style="color:#ff00ff;text-shadow:0 0 8px #ff00ff;'
                   r'border-bottom:2px solid #00ffff;">\1</span>', text)

    def _section(icon, label, value):
        if not value: return ""
        return (f'<div class="vellum-section">'
                f'<div class="section-header">{icon} {label}</div>'
                f'<div class="content">{value}</div></div>')

    details = (
        _section("📜", "DEFINITION",   note.get("Definition", ""))  +
        _section("🗣️", "PRONUNCIATION", note.get("Pronunciation", "")) +
        _section("➕", "SYNONYMS",     note.get("Synonyms", ""))    +
        _section("➖", "ANTONYMS",     note.get("Antonyms", ""))
    )

    html = f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&family=Electrolize:wght@400;700&display=swap" rel="stylesheet">
<style>body{{margin:0;padding:0;}}{CYBERPUNK_CSS}</style>
</head><body class="card">
<div class="preview-label">▌ FRONT — Question</div>
<div class="vellum-focus-container"><div class="prompt-text">{front}</div></div>
<div class="preview-label">▌ BACK — Answer</div>
<div class="vellum-focus-container"><div class="prompt-text">{back}</div></div>
<div class="vellum-detail-container">{details}</div>
</body></html>"""
    return html

# ========================== AUDIO HELPER ==========================
def generate_audio_file(vocab, temp_dir):
    """gTTS with a 10s hard timeout — a stalled TTS request can no longer hang the thread."""
    try:
        clean_vocab    = re.sub(r"[^a-zA-Z0-9 \-']", '', vocab).strip()
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', clean_vocab) + ".mp3"
        file_path      = os.path.join(temp_dir, clean_filename)
        if clean_vocab:
            def _do_tts():
                tts = gTTS(text=clean_vocab, lang='en', slow=False)
                tts.save(file_path)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_tts)
                fut.result(timeout=10)
            return vocab, clean_filename, file_path
    except concurrent.futures.TimeoutError:
        print(f"gTTS timeout for '{vocab}' — skipping audio.")
    except Exception as e:
        print(f"Audio error for {vocab}: {e}")
    return vocab, None, None


# ========================== GENANKI LOGIC ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True,
                        deck_id=2059400110, include_antonyms=True):
    # Bug fix #2: {{Audio}} appears only ONCE (inside the hidden div)
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html  = (
        """<div class="vellum-focus-container back">"""
        """<div class="prompt-text solved-text">{{cloze:Text}}</div></div>"""
        """<div class="vellum-detail-container">"""
        """{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}"""
        """{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}"""
        """{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}"""
        """{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}"""
    )
    if include_antonyms:
        back_html += """{{#Antonyms}}<div class="vellum-section"><div class="section-header">➖ ANTONYMS</div><div class="content">{{Antonyms}}</div></div>{{/Antonyms}}"""
    back_html += (
        """{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}"""
        # ✅ Audio rendered once inside hidden div — no bare {{Audio}} after this
        """<div style='display:none'>{{Audio}}</div>"""
        """</div>{{Audio}}"""
    )

    # B6: Session-state model ID
    model_id = st.session_state.get("model_id", 1607392319)
    my_model = genanki.Model(
        model_id,
        'Cyberpunk Vocab Model',
        fields=[
            {'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'},
            {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'},
            {'name': 'Etymology'}, {'name': 'Audio'}
        ],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=CYBERPUNK_CSS,
        model_type=genanki.Model.CLOZE
    )
    my_deck    = genanki.Deck(deck_id, deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(generate_audio_file, v, temp_dir): v for v in unique_vocabs}
                for future in concurrent.futures.as_completed(futures):
                    vk, fn, fp = future.result()
                    if fn:
                        media_files.append(fp)
                        audio_map[vk] = f"[sound:{fn}]"

        for note_data in notes_data:
            # B9: Deterministic GUID scoped to deck
            guid_input = note_data['VocabRaw'] + deck_name
            vocab_hash = str(int(hashlib.sha256(guid_input.encode('utf-8')).hexdigest(), 16) % (10 ** 10))
            my_deck.add_note(genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'], note_data['Pronunciation'], note_data['Definition'],
                    note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'],
                    note_data['Etymology'], audio_map.get(note_data['VocabRaw'], "")
                ],
                tags=note_data['Tags'],
                guid=vocab_hash
            ))

        my_package            = genanki.Package(my_deck)
        my_package.media_files = media_files
        output_path           = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(output_path)
        buffer = io.BytesIO()
        with open(output_path, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
    return buffer

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str)
        df['phrase']     = df['phrase'].fillna("")
        df['status']     = df.get('status', pd.Series(['New'] * len(df))).fillna('New')
        df['tags']       = df.get('tags',   pd.Series(['']   * len(df))).fillna('')
        # Enhancement #10: date_added column
        df['date_added'] = df.get('date_added', pd.Series([''] * len(df))).fillna('')
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags', 'date_added'])
        st.stop()
    except:
        st.stop()

def save_to_github(dataframe):
    dataframe = (
        dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
        .drop_duplicates(subset=['vocab'], keep='last')
    )
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear()
    return True

# ========================== SESSION STATE INIT ==========================
if "vocab_df"          not in st.session_state: st.session_state.vocab_df          = load_data().copy()
if "deck_id"           not in st.session_state: st.session_state.deck_id           = 2059400110
if "bulk_preview_df"   not in st.session_state: st.session_state.bulk_preview_df   = None
if "apkg_buffer"       not in st.session_state: st.session_state.apkg_buffer       = None
if "processed_vocabs"  not in st.session_state: st.session_state.processed_vocabs  = []
if "preview_notes"     not in st.session_state: st.session_state.preview_notes     = []
if "model_id"          not in st.session_state: st.session_state.model_id          = 1607392319
if "include_antonyms"  not in st.session_state: st.session_state.include_antonyms  = True
if "dry_run"           not in st.session_state: st.session_state.dry_run           = False
if "processed_cache"   not in st.session_state: st.session_state.processed_cache   = {}
if "input_phrase"      not in st.session_state: st.session_state.input_phrase      = ""
if "input_vocab"       not in st.session_state: st.session_state.input_vocab       = ""
if "api_key_valid"     not in st.session_state: st.session_state.api_key_valid     = False

# ========================== CALLBACKS ==========================
def mark_as_done_callback():
    if st.session_state.processed_vocabs:
        st.session_state.vocab_df.loc[
            st.session_state.vocab_df['vocab'].isin(st.session_state.processed_vocabs), 'status'
        ] = 'Done'
        save_to_github(st.session_state.vocab_df)
    st.session_state.apkg_buffer      = None
    st.session_state.processed_vocabs = []
    st.session_state.preview_notes    = []

def save_single_word_callback():
    v     = st.session_state.input_vocab.lower().strip()
    p_raw = st.session_state.input_phrase
    if v:
        p = p_raw.strip()
        if p and p != "1" and not p.startswith("*"):
            if p.endswith(","): p = p[:-1] + "."
            elif not p.endswith((".", "!", "?")): p += "."
            p = p.capitalize()
        mask = st.session_state.vocab_df['vocab'] == v
        if not st.session_state.vocab_df.empty and mask.any():
            st.session_state.vocab_df.loc[mask, ['phrase', 'status']] = [p, 'New']
        else:
            new_row = pd.DataFrame([{
                "vocab": v, "phrase": p, "status": "New",
                "tags": "", "date_added": str(date.today())   # Enhancement #10
            }])
            st.session_state.vocab_df = pd.concat(
                [st.session_state.vocab_df, new_row], ignore_index=True
            )
        save_to_github(st.session_state.vocab_df)
        st.session_state.input_phrase = ""
        st.session_state.input_vocab  = ""
        st.toast(f"✅ Saved '{v}'!", icon="🚀")
    else:
        st.error("⚠️ Enter a vocabulary word.")

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("⚙️ Settings")

    total_words = len(st.session_state.vocab_df)
    new_words   = len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'])

    col1, col2 = st.columns(2)
    col1.metric("📖 Total", total_words)
    col2.metric("✨ New",   new_words)
    st.metric("🤖 Daily AI Usage", f"{st.session_state.rpd_count}/20 Requests")

    # A2: Visual quota dashboard
    rpm_current = len([ts for ts in st.session_state.rpm_timestamps
                       if (datetime.now() - ts).total_seconds() < 60])
    st.progress(rpm_current / 5, text=f"RPM Live: {rpm_current}/5 (last 60s)")
    st.progress(st.session_state.rpd_count / 20, text=f"RPD: {st.session_state.rpd_count}/20")

    # Enhancement #15: Status bar chart
    if not st.session_state.vocab_df.empty:
        status_counts = st.session_state.vocab_df['status'].value_counts()
        st.bar_chart(status_counts, height=140)

    st.divider()
    TARGET_LANG  = st.selectbox("🎯 Translation Language",
                                ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"],
                                index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model",
                                ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"],
                                index=0)

    # Enhancement #13: API key live validator
    key_valid = validate_api_key(st.session_state.gemini_key, GEMINI_MODEL)
    if key_valid:
        st.success("🔑 API Key: Valid ✓", icon="✅")
    else:
        st.error("🔑 API Key: Invalid ✗")

    st.divider()
    # B6: Model ID control
    if st.button("🔄 Regenerate Note Type Model ID"):
        st.session_state.model_id = random.randrange(1 << 30, 1 << 31)
        st.success(f"New Model ID: {st.session_state.model_id}")
    st.caption(f"Current Model ID: {st.session_state.model_id}")

    if not st.session_state.vocab_df.empty:
        st.download_button(
            "💾 Backup Database (CSV)",
            st.session_state.vocab_df.to_csv(index=False).encode('utf-8'),
            f"vocab_backup_{date.today()}.csv", "text/csv"
        )

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

# ─────────────────────────── TAB 1: ADD ───────────────────────────────────────
with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")

    if add_mode == "Single":
        p_raw      = st.text_input("🔤 Phrase", placeholder="Paste your sentence here...", key="input_phrase")
        v_selected = ""
        if p_raw and p_raw not in ["1", "*"]:
            clean_text   = re.sub(r'[^\w\s\-\']', '', p_raw)
            unique_words = list(dict.fromkeys([w.lower() for w in clean_text.split() if w.strip()]))
            if unique_words:
                st.caption("Click words below to extract them as vocabulary:")
                try:
                    selected_pills = st.pills("Select Vocab", unique_words,
                                              selection_mode="multi", label_visibility="collapsed")
                    v_selected = " ".join(selected_pills) if selected_pills else ""
                except:
                    selected_words = []
                    cols = st.columns(6)
                    for i, w in enumerate(unique_words):
                        if cols[i % 6].checkbox(w, key=f"chk_{w}"):
                            selected_words.append(w)
                    v_selected = " ".join(selected_words)

        if v_selected and v_selected != st.session_state.input_vocab:
            st.session_state.input_vocab = v_selected

        st.text_input("📝 Vocab", placeholder="e.g. serendipity", key="input_vocab")

        v_check = st.session_state.input_vocab.lower().strip()
        if v_check and not st.session_state.vocab_df.empty and (st.session_state.vocab_df['vocab'] == v_check).any():
            st.warning(f"⚠️ '{v_check}' already exists. Saving will overwrite its phrase and reset to 'New'.")

        st.button("💾 Save to Cloud", type="primary", use_container_width=True,
                  on_click=save_single_word_callback)

    else:
        bulk_text = st.text_area("Paste List (word, phrase)", height=150, key="bulk_input")
        if st.button("Preview Bulk Import"):
            lines    = [l.strip() for l in bulk_text.split('\n') if l.strip()]
            new_rows = []
            for line in lines:
                parts = line.split(',', 1)
                bv    = parts[0].strip().lower()
                bp    = parts[1].strip() if len(parts) > 1 else ""
                if bp and bp != "1" and not bp.startswith("*"):
                    if bp.endswith(","): bp = bp[:-1] + "."
                    elif not bp.endswith((".", "!", "?")): bp += "."
                    bp = bp.capitalize()
                if bv:
                    new_rows.append({
                        "vocab": bv, "phrase": bp, "status": "New",
                        "tags": "", "date_added": str(date.today())  # Enhancement #10
                    })
            if new_rows:
                st.session_state.bulk_preview_df = pd.DataFrame(new_rows)

        if st.session_state.bulk_preview_df is not None:
            st.write("### Preview:")
            st.dataframe(st.session_state.bulk_preview_df, hide_index=True)
            if st.button("💾 Confirm & Process Bulk", type="primary"):
                st.session_state.vocab_df = pd.concat(
                    [st.session_state.vocab_df, st.session_state.bulk_preview_df]
                ).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(st.session_state.vocab_df)
                st.success(f"✅ Added {len(st.session_state.bulk_preview_df)} words!")
                st.session_state.bulk_preview_df = None
                time.sleep(0.5)
                st.rerun()

# ─────────────────────────── TAB 2: EDIT ──────────────────────────────────────
with tab2:
    if st.session_state.vocab_df.empty:
        st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        search     = st.text_input("🔎 Search...", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()   # preserves original indices
        if search:
            display_df = display_df[display_df['vocab'].str.contains(search, case=False)]

        # D16: Pagination
        page_size = 50
        page      = st.number_input("Page", min_value=1, value=1, step=1)
        start     = (page - 1) * page_size
        paginated_df = display_df.iloc[start:start + page_size]

        edited = st.data_editor(
            paginated_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "status":     st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True),
                "date_added": st.column_config.TextColumn("Added", disabled=True),  # Enhancement #10
            }
        )

        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            # Bug fix #3: Merge only the edited rows back by index — no silent deletion.
            existing_mask  = edited.index.isin(st.session_state.vocab_df.index)
            existing_edits = edited[existing_mask]
            new_rows       = edited[~existing_mask]

            st.session_state.vocab_df.update(existing_edits)   # in-place index-aligned merge

            if not new_rows.empty:
                if 'date_added' not in new_rows.columns:
                    new_rows = new_rows.copy()
                    new_rows['date_added'] = str(date.today())
                st.session_state.vocab_df = pd.concat(
                    [st.session_state.vocab_df, new_rows], ignore_index=True
                )

            save_to_github(st.session_state.vocab_df)
            st.toast("✅ Cloud updated!")
            time.sleep(0.5)
            st.rerun()

        # Enhancement #14: Reset Done → New
        with st.expander("🔄 Reset 'Done' words back to 'New'"):
            done_count = len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'Done'])
            if done_count == 0:
                st.info("No 'Done' words to reset.")
            else:
                st.warning(f"This will reset all **{done_count}** 'Done' words back to 'New' for re-export.")
                if st.button("🔄 Confirm Reset All to 'New'", type="primary"):
                    st.session_state.vocab_df.loc[
                        st.session_state.vocab_df['status'] == 'Done', 'status'
                    ] = 'New'
                    save_to_github(st.session_state.vocab_df)
                    st.toast(f"✅ {done_count} words reset to 'New'!")
                    time.sleep(0.5)
                    st.rerun()

# ─────────────────────────── TAB 3: GENERATE ANKI ─────────────────────────────
with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")

    # ── Download / preview section (shown after generation) ──────────────────
    if st.session_state.apkg_buffer is not None:
        st.success("✅ Deck generated! Preview below, then download to sync your database.")

        # Enhancement #8: Live card preview for first 3 notes
        if st.session_state.preview_notes:
            with st.expander("🎴 Card Preview (first 3 cards)", expanded=True):
                for i, note in enumerate(st.session_state.preview_notes[:3]):
                    st.caption(f"Card {i+1}: `{note['VocabRaw']}`")
                    st.components.v1.html(render_card_preview(note), height=420, scrolling=True)

        st.download_button(
            "📥 Download .apkg",
            data=st.session_state.apkg_buffer,
            file_name=f"AnkiDeck_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg",
            mime="application/octet-stream",
            use_container_width=True,
            on_click=mark_as_done_callback
        )
        if st.button("❌ Cancel / Clear"):
            st.session_state.apkg_buffer      = None
            st.session_state.processed_vocabs = []
            st.session_state.preview_notes    = []
            st.rerun()

    # ── Generation form ───────────────────────────────────────────────────────
    else:
        if st.session_state.vocab_df.empty:
            st.info("Add words first!")
        else:
            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
            if subset.empty:
                st.warning("⚠️ No 'New' words to export!")
            else:
                # Enhancement #18: Search filter above the export editor
                search_export = st.text_input(
                    "🔎 Filter words to export...", "",
                    placeholder="Type to narrow down..."
                ).lower().strip()
                if search_export:
                    subset = subset[
                        subset['vocab'].str.contains(search_export, case=False) |
                        subset['phrase'].str.contains(search_export, case=False, na=False)
                    ]

                deck_col1, deck_col2 = st.columns([3, 1])
                deck_name_input = deck_col1.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
                if deck_col2.button("🎲 New Deck ID"):
                    st.session_state.deck_id = random.randrange(1 << 30, 1 << 31)
                deck_col2.caption(f"ID: {st.session_state.deck_id}")

                batch_size    = st.slider("⚡ Batch Size (Words per Request)", 1, 15, 10)
                requests_left = max(0, 20 - st.session_state.rpd_count)
                # A3: Auto-adjust batch size to fit quota
                max_safe_batch = max(1, math.ceil(len(subset) / max(1, requests_left))) if requests_left > 0 else 1
                batch_size     = min(batch_size, max_safe_batch)
                st.caption(f"✅ Auto-adjusted effective batch size: **{batch_size}** (based on remaining quota)")

                include_audio                  = st.checkbox("🔊 Generate Audio Files", value=True)
                st.session_state.include_antonyms = st.checkbox("➖ Include Antonyms in Card Back",  value=st.session_state.include_antonyms)
                st.session_state.dry_run          = st.checkbox("🔬 Dry Run (simulate AI, no quota)", value=st.session_state.dry_run)

                st.write("**Select words to export:**")
                subset_export     = subset.copy()
                subset_export['Export'] = True
                edited_export = st.data_editor(
                    subset_export,
                    column_config={"Export": st.column_config.CheckboxColumn("Export?", required=True)},
                    hide_index=True,
                    disabled=["vocab", "phrase", "status", "tags", "date_added"]
                )
                final_export_subset = edited_export[edited_export['Export'] == True]

                # D17: Export preview + Enhancement #20: refined size estimator
                if not final_export_subset.empty:
                    st.write("### Export Preview")
                    st.dataframe(final_export_subset[['vocab', 'phrase']], hide_index=True)
                    card_count = len(final_export_subset)
                    # Refined formula: 0.5 KB base per card + 15 KB/card for audio
                    est_size_kb = card_count * (15.5 if include_audio else 0.5)
                    audio_note  = "(incl. ~15 KB/word audio)" if include_audio else "(text only)"
                    st.info(f"📊 **{card_count} cards** • Est. .apkg size: **{est_size_kb:.0f} KB** {audio_note}")

                required_requests = math.ceil(len(final_export_subset) / batch_size) if not final_export_subset.empty else 0
                st.info(f"💡 **{requests_left}** API requests left today — this batch needs **{required_requests}**.")

                if final_export_subset.empty:
                    st.warning("Select at least one word to export.")
                elif required_requests > requests_left and not st.session_state.dry_run:
                    st.error("🛑 Exceeds Daily Limit! Reduce selection or increase batch size.")
                else:
                    if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
                        # Bug fix #1: raw_notes initialised before try — prevents NameError in except
                        raw_notes = []
                        try:
                            raw_notes = process_anki_data(
                                final_export_subset,
                                batch_size=batch_size,
                                dry_run=st.session_state.dry_run,
                                target_lang=TARGET_LANG
                            )
                            if raw_notes:
                                apkg_buffer = create_anki_package(
                                    raw_notes, deck_name_input,
                                    generate_audio=include_audio,
                                    deck_id=st.session_state.deck_id,
                                    include_antonyms=st.session_state.include_antonyms
                                )
                                st.session_state.apkg_buffer      = apkg_buffer.getvalue()
                                st.session_state.processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                                st.session_state.preview_notes    = raw_notes  # Enhancement #8
                                st.rerun()
                            else:
                                st.warning("⚠️ No notes were generated. Check your quota and try again.")
                        except Exception as e:
                            st.error(f"❌ Generation error: {e}. Status rolled back to 'New'.")
                            # Bug fix #1: raw_notes is always defined here — no NameError
                            if raw_notes:
                                st.session_state.vocab_df.loc[
                                    st.session_state.vocab_df['vocab'].isin(
                                        [n.get('VocabRaw', '') for n in raw_notes]
                                    ), 'status'
                                ] = 'New'
                                save_to_github(st.session_state.vocab_df)
                            if st.button("🔁 Retry with smaller batch"):
                                st.rerun()
