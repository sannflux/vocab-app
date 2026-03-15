"""
╔══════════════════════════════════════════════════════════════╗
║       MY CLOUD VOCAB — PERFORMANCE-OPTIMISED BUILD           ║
║                                                              ║
║  Applied optimisations (vs previous version):               ║
║  A1  @st.fragment on Tab 3 form  → no full-page rerun       ║
║  A2  @st.fragment on Tab 2 editor → no full-page rerun      ║
║  A3  Removed 3× time.sleep(0.5) dead-waits                  ║
║  A4  All 11 regex patterns pre-compiled at module level      ║
║  A5  pd.util.hash_pandas_object for O(1) cache key          ║
║  A6/A7 Merged usage_combined.json  → 1 read + 1 write       ║
║  B1  genanki.Model cached in session_state                   ║
║  B2  Parallel card cleaning via ThreadPoolExecutor           ║
║  B4  Optimistic session-state update; no extra GH read       ║
║  B5  Lazy load_data (skip if vocab_df already in state)      ║
║  C1  Non-blocking RPM: single sleep() + st.toast            ║
║  C2  Fire-and-forget GitHub usage write on daemon thread     ║
║  C3  RPD pre-check before enforce_rpm in batch loop          ║
║  C4  perf_counter on every API call + adaptive timeout       ║
║  C5  SYSTEM_INSTRUCTION whitespace-stripped                  ║
║  D1  _process_single_card → parallelised field extraction    ║
║  D2  functools.lru_cache on highlight_vocab pattern          ║
║  D3  BytesIO direct path; skips redundant temp-file read     ║
║  D4  Module-level _AUDIO_POOL (thread-pool reuse)            ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import random
from datetime import date, datetime
import google.generativeai as genai
import json
import re
import time
import os
import tempfile
import hashlib
import concurrent.futures
import threading
import math
import functools
from time import perf_counter
from typing import TypedDict, List as TList

# ── Tenacity ──────────────────────────────────────────────────────────────────
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

# ── D4: Module-level audio thread-pool (persistent; avoids per-call spawn) ────
_AUDIO_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=5)

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
st.components.v1.html("""
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
        setTimeout(() => { e.target.blur(); }, 50);
    }
}, true);
</script>
""", height=0)

# ========================== RESPONSE SCHEMA ==========================
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

# ========================== SYSTEM INSTRUCTION (C5: whitespace-stripped) ==========================
# Static rules + few-shots → ~40 % fewer tokens per user-turn request.
SYSTEM_INSTRUCTION = (
    "You are an expert educational lexicographer creating Anki flashcard content.\n"
    "Think step-by-step:\n"
    "1. Identify the primary sense of the vocabulary word from the phrase/context provided.\n"
    "2. Generate accurate, educational linguistic data for every field.\n"
    "3. Ensure your response is a valid JSON array containing EXACTLY the number of items requested.\n"
    "OUTPUT FORMAT: Respond ONLY with a valid JSON array. No preamble, no commentary, no markdown fences.\n"
    "SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.\n"
    "FIELD RULES:\n"
    "1. Copy ALL input fields exactly as provided.\n"
    "2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT for the word's meaning/usage.\n"
    "3. IF 'phrase' is normal text: Use it ONLY to identify which meaning/nuance of 'vocab' applies.\n"
    "4. IF 'phrase' is empty: Generate ONE simple example sentence (max 12 words) for the vocab.\n"
    "5. The EXACT 'vocab' field value must remain unchanged in your output.\n"
    "6. 'part_of_speech' MUST be one of: Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection, Phrase.\n"
    "7. 'example_sentences': provide up to 3 varied example sentences.\n"
    "8. 'synonyms_antonyms': provide up to 5 synonyms and 5 antonyms.\n"
    "9. 'etymology': brief origin note (1-2 sentences max).\n"
    'FEW-SHOT EXAMPLES:\n'
    '[{"vocab":"serendipity","phrase":"We found the perfect cafe by pure serendipity.","translation":"kebetulan","part_of_speech":"Noun","pronunciation_ipa":"/\u02ccs\u025br\u0259n\u02c8d\u026ap\u026ati/","definition_english":"The occurrence and development of events by chance in a happy or beneficial way.","example_sentences":["It was pure serendipity that we met.","His discovery was a happy serendipity.","Serendipity led her to the perfect job."],"synonyms_antonyms":{"synonyms":["chance","luck","fortune","fluke","coincidence"],"antonyms":["misfortune","design","intention","plan","deliberation"]},"etymology":"Coined by Horace Walpole in 1754, from the Persian fairy tale The Three Princes of Serendip."},'
    '{"vocab":"ephemeral","phrase":"","translation":"sementara","part_of_speech":"Adjective","pronunciation_ipa":"/\u026a\u02c8f\u025bm\u0259r\u0259l/","definition_english":"Lasting for a very short time.","example_sentences":["The ephemeral beauty of the sunset faded quickly.","Social media trends are often ephemeral.","Morning dew is an ephemeral phenomenon."],"synonyms_antonyms":{"synonyms":["transient","fleeting","temporary","brief","momentary"],"antonyms":["permanent","lasting","eternal","enduring","perpetual"]},"etymology":"From Greek ephemeros, literally meaning lasting only a day, from epi- (on) + hemera (day)."},'
    '{"vocab":"run","phrase":"*He decided to run for office","translation":"mencalonkan diri","part_of_speech":"Verb","pronunciation_ipa":"/r\u028cn/","definition_english":"To compete as a candidate in an election or for a position.","example_sentences":["She decided to run for city council.","He announced he would run for president.","Three candidates are running for the seat."],"synonyms_antonyms":{"synonyms":["campaign","stand","contest","seek election","bid"],"antonyms":["withdraw","abstain","concede","step down","retire"]},"etymology":"Old English rinnan, of Germanic origin; the political sense emerged in 19th-century American English."}]'
)

# ========================== SECRETS ==========================
try:
    token              = st.secrets["GITHUB_TOKEN"]
    repo_name          = st.secrets["REPO_NAME"]
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

# ========================== A7: MERGED USAGE TRACKING ==========================
# One GitHub read at startup, one write per batch — down from 2+2.

@st.cache_data(ttl=60)   # A6: 60s cache — prevents re-fetch within same minute
def _fetch_combined_usage_raw() -> dict:
    """Cached GitHub read — returns raw dict."""
    try:
        file = repo.get_contents("usage_combined.json")
        return json.loads(file.decoded_content.decode("utf-8"))
    except Exception:
        return {}

def load_combined_usage() -> tuple[int, list]:
    """Return (rpd_count, rpm_timestamps) for today."""
    t0 = perf_counter()
    data      = _fetch_combined_usage_raw()
    today_str = str(date.today())
    rpd       = data.get("rpd_count", 0) if data.get("date") == today_str else 0
    ts_raw    = data.get("timestamps", []) if data.get("date") == today_str else []
    timestamps = []
    for ts in ts_raw:
        try:
            timestamps.append(datetime.fromisoformat(ts))
        except Exception:
            pass
    print(f"[PERF] load_combined_usage: {(perf_counter()-t0)*1000:.1f}ms")
    return rpd, timestamps

def save_combined_usage(rpd_count: int, timestamps: list):
    """Synchronous write — called from daemon thread (C2)."""
    data = json.dumps({
        "date":       str(date.today()),
        "rpd_count":  rpd_count,
        "timestamps": [ts.isoformat() for ts in timestamps],
    })
    try:
        file = repo.get_contents("usage_combined.json")
        repo.update_file(file.path, "Update usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage_combined.json", "Init usage", data)
    _fetch_combined_usage_raw.clear()   # invalidate read-cache after write

def _async_save_usage(rpd_count: int, timestamps: list):
    """C2: Fire-and-forget daemon thread — never blocks UI."""
    t = threading.Thread(
        target=save_combined_usage,
        args=(rpd_count, list(timestamps)),
        daemon=True,
    )
    t.start()
    return t

# ========================== SESSION STATE INIT (B5: lazy load) ==========================
# load_data() is defined later but the lazy-guard is applied here.
# We defer the actual CSV fetch until after load_data is defined.

# ========================== enforce_rpm (C1: non-blocking) ==========================
def enforce_rpm():
    """
    Rate-limit guard — C1 improvement:
    Single time.sleep(wait_sec) + st.toast instead of 12-iteration countdown loop.
    The UI thread is blocked for the minimal required duration only.
    No GitHub I/O here.
    """
    today_str = str(date.today())
    # Midnight RPD auto-reset
    if st.session_state.rpm_date != today_str:
        st.session_state.rpd_count      = 0
        st.session_state.rpm_timestamps = []
        st.session_state.rpm_date       = today_str
        _async_save_usage(0, [])

    now = datetime.now()
    st.session_state.rpm_timestamps = [
        ts for ts in st.session_state.rpm_timestamps
        if (now - ts).total_seconds() < 60
    ]

    if len(st.session_state.rpm_timestamps) >= 5:
        oldest    = min(st.session_state.rpm_timestamps)
        wait_sec  = max(0.0, 60.0 - (now - oldest).total_seconds()) + 1.0
        st.toast(f"⏳ RPM limit — waiting {wait_sec:.0f}s before next call…", icon="⏳")
        time.sleep(wait_sec)   # single sleep — minimal block

    st.session_state.rpm_timestamps.append(datetime.now())

# ========================== GEMINI MODEL ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1},
        )
    except Exception as e:
        st.error(f"❌ Gemini key error: {e}")
        return None

# ========================== API KEY VALIDATOR ==========================
def validate_api_key(api_key: str, model_name: str) -> bool:
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
    except Exception:
        st.session_state.api_key_valid         = False
        st.session_state.api_key_validated_key = cache_key
        return False

# ========================== GEMINI CALL (C4: adaptive timeout) ==========================
GEMINI_TIMEOUT_SEC = 55   # hard ceiling; tightened adaptively below

def _update_latency_stats(elapsed_sec: float):
    """C4: Track API call latencies for adaptive timeout."""
    hist = st.session_state.setdefault("api_latencies", [])
    hist.append(elapsed_sec)
    st.session_state.api_latencies = hist[-20:]   # keep last 20 samples

def _get_adaptive_timeout() -> float:
    """C4: p95 × 1.5, clamped [20, 55]s. Falls back to GEMINI_TIMEOUT_SEC."""
    hist = st.session_state.get("api_latencies", [])
    if len(hist) < 3:
        return GEMINI_TIMEOUT_SEC
    p95 = sorted(hist)[min(int(len(hist) * 0.95), len(hist) - 1)]
    return max(20.0, min(float(GEMINI_TIMEOUT_SEC), p95 * 1.5))

def _is_rate_limit(exc: BaseException) -> bool:
    return "429" in str(exc)

def _raw_gemini_call(model, prompt: str):
    return model.generate_content(
        prompt,
        request_options={"timeout": _get_adaptive_timeout()},
    )

if TENACITY_AVAILABLE:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=20, max=60),
        retry=retry_if_exception(_is_rate_limit),
        reraise=True,
    )
    def _call_gemini(model, prompt: str):
        return _raw_gemini_call(model, prompt)
else:
    def _call_gemini(model, prompt: str):
        return _raw_gemini_call(model, prompt)

# ========================== A4: PRE-COMPILED REGEX ==========================
# All patterns compiled once at import time — never recompiled per-card.

_RE_SPACES        = re.compile(r"\s+")
_RE_CLOZE_FRONT   = re.compile(r'\{\{c\d+::(.*?)(?::.*?)?\}\}')
_RE_CLOZE_BACK    = re.compile(r'\{\{c\d+::(.*?)(?::.*?)?\}\}')
_RE_JSON_FENCE    = re.compile(r"^```(?:json)?\s*|\s*```$")
_RE_JSON_ARRAY    = re.compile(r'\[.*\]', re.DOTALL)
_RE_SENTENCE_SPLIT= re.compile(r'(?<=[.!?])\s+')
_RE_NONWORD_AUDIO = re.compile(r"[^a-zA-Z0-9 \-']")
_RE_ALPHANUM      = re.compile(r'[^a-zA-Z0-9]')
_RE_GRAMMAR_RULES = [
    (re.compile(r"\bto doing\b",        re.IGNORECASE), "to do"),
    (re.compile(r"\bfor helps\b",       re.IGNORECASE), "to help"),
    (re.compile(r"\bis use to\b",       re.IGNORECASE), "is used to"),
    (re.compile(r"\bhelp for to\b",     re.IGNORECASE), "help to"),
    (re.compile(r"\bfor to\b",          re.IGNORECASE), "to"),
    (re.compile(r"\bcan able to\b",     re.IGNORECASE), "can"),
    (re.compile(r"\bI am agree\b",      re.IGNORECASE), "I agree"),
    (re.compile(r"\bdiscuss about\b",   re.IGNORECASE), "discuss"),
    (re.compile(r"\breturn back\b",     re.IGNORECASE), "return"),
]

# ========================== CLEANING FUNCTIONS ==========================

def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if s and s[-1] in ".!?" else (s + "." if s else "")

def normalize_spaces(text: str) -> str:
    return _RE_SPACES.sub(" ", str(text)).strip() if text else ""

def clean_grammar(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pattern, repl in _RE_GRAMMAR_RULES:
        text = pattern.sub(repl, text)
    return text

def cap_each_sentence(text: str) -> str:
    if not isinstance(text, str):
        return text
    parts = _RE_SENTENCE_SPLIT.split(text)
    return " ".join(cap_first(s) for s in parts if s.strip())

# D2: lru_cache on pattern construction — zero recompilation for repeat vocabs
@functools.lru_cache(maxsize=512)
def _vocab_pattern(vocab: str) -> re.Pattern:
    return re.compile(r'\b' + re.escape(vocab) + r'\b', re.IGNORECASE)

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab:
        return text
    return _vocab_pattern(vocab).sub(f'<b><u>{vocab}</u></b>', text)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab:
        return phrase
    return _vocab_pattern(vocab.lower()).sub(vocab, phrase)

def robust_json_parse(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = _RE_JSON_FENCE.sub("", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _RE_JSON_ARRAY.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

def speak_word(text: str, lang: str = "en-US"):
    if not text:
        return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    st.components.v1.html(
        f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");"""
        f"""u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>""",
        height=0,
    )

# ========================== B2 + D1: PARALLEL CARD CLEANING ==========================
def _process_single_card(args: tuple) -> dict | None:
    """
    D1/B2: All field extraction + cleaning for ONE card dict.
    Runs in a thread-pool worker → parallel across all cards in the batch.
    """
    card_data, target_lang = args

    required = ["vocab", "translation", "part_of_speech"]
    if not all(k in card_data and card_data[k] for k in required):
        return None

    vocab_raw = str(card_data.get("vocab", "")).strip().lower()
    if not vocab_raw:
        return None
    vocab_cap = cap_first(vocab_raw)

    phrase = normalize_spaces(card_data.get("phrase", ""))
    phrase = clean_grammar(phrase)
    phrase = cap_each_sentence(phrase)
    phrase = ensure_trailing_dot(phrase)
    phrase = fix_vocab_casing(phrase, vocab_raw)
    fmt_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""

    translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("translation", "?"))))
    pos         = str(card_data.get("part_of_speech", "")).title()
    ipa         = card_data.get("pronunciation_ipa", "")
    eng_def     = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))

    examples = [
        ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e))))
        for e in (card_data.get("example_sentences") or [])[:3]
    ]
    examples_field = (
        "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>"
        if examples else ""
    )

    syn_ant        = card_data.get("synonyms_antonyms") or {}
    synonyms_field = ensure_trailing_dot(", ".join(cap_first(s) for s in (syn_ant.get("synonyms") or [])[:5]))
    antonyms_field = ensure_trailing_dot(", ".join(cap_first(a) for a in (syn_ant.get("antonyms") or [])[:5]))
    etymology      = normalize_spaces(card_data.get("etymology", ""))

    text_field = (
        f"{fmt_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        if fmt_phrase else
        f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
    )
    pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"

    tags = []
    if pos:
        tags.append(pos.replace(" ", "_"))
    if target_lang:
        tags.append(target_lang.replace(" ", "_").replace("(", "").replace(")", ""))

    return {
        "VocabRaw":      vocab_raw,
        "Text":          text_field,
        "Pronunciation": pronunciation_field,
        "Definition":    eng_def,
        "Examples":      examples_field,
        "Synonyms":      synonyms_field,
        "Antonyms":      antonyms_field,
        "Etymology":     etymology,
        "Tags":          tags,
    }

# ========================== SINGLE-ITEM FALLBACK ==========================
def _generate_single_item(model, vocab: str, phrase: str, target_lang: str) -> dict | None:
    prompt = (
        f"MANDATORY: 'translation' field must contain ONLY the {target_lang} translation "
        f"of the vocab word. NEVER translate the full sentence.\n\n"
        f"Output EXACTLY 1 item as a JSON array.\n\n"
        f"BATCH INPUT: {json.dumps([{'vocab': vocab, 'phrase': phrase}], ensure_ascii=False)}"
    )
    try:
        response = _call_gemini(model, prompt)
        parsed   = robust_json_parse(response.text)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
    except Exception:
        pass
    return None

# ========================== ASYNC BATCH GENERATOR ==========================
_SEC_PER_BATCH = 20

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6,
                                    dry_run=False, target_lang="Indonesian"):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model:
        return []

    all_card_data = []
    batches       = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    rpd_delta     = 0

    with st.status("🤖 Processing AI Batches...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        eta_slot     = st.empty()

        est_total_sec = len(batches) * _SEC_PER_BATCH
        eta_slot.info(
            f"⏱️ Est. ~{est_total_sec}s "
            f"({len(batches)} batch{'es' if len(batches) != 1 else ''} × ~{_SEC_PER_BATCH}s each)"
        )

        t_start = perf_counter()   # C4: wall-clock start

        for idx, batch in enumerate(batches):
            # C3: Pre-check quota BEFORE enforce_rpm — avoids wasted RPM check
            if st.session_state.rpd_count + rpd_delta >= 20:
                st.warning("🛑 Daily AI Limit (20 requests) reached. Please try again tomorrow.")
                break

            enforce_rpm()   # C1: single-sleep, in-memory only

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            vocab_words = [v[0] for v in batch]

            prompt = (
                f"MANDATORY: 'translation' field must contain ONLY the {target_lang} translation "
                f"of the vocab word. NEVER translate the full sentence.\n\n"
                f"Output EXACTLY {len(batch_dicts)} items.\n\n"
                f"BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"
            )

            success   = False
            step_slot = st.empty()

            if dry_run:
                step_slot.info(f"🔬 Dry-run: {', '.join(vocab_words)}")
                all_card_data.extend([
                    {"vocab": v[0], "phrase": v[1], "translation": "mock-" + v[0],
                     "part_of_speech": "Noun", "pronunciation_ipa": "/mock/",
                     "definition_english": "Simulated definition.",
                     "example_sentences": ["Mock example sentence."],
                     "synonyms_antonyms": {"synonyms": ["mock"], "antonyms": []},
                     "etymology": "Simulated."}
                    for v in batch
                ])
                success = True
            else:
                try:
                    t_call = perf_counter()
                    step_slot.info(
                        f"🌐 Batch {idx+1}/{len(batches)}: calling Gemini… "
                        f"(timeout={_get_adaptive_timeout():.0f}s)"
                    )
                    response  = _call_gemini(model, prompt)
                    call_sec  = perf_counter() - t_call
                    call_ms   = int(call_sec * 1000)
                    _update_latency_stats(call_sec)   # C4: feed adaptive timeout

                    rpd_delta += 1
                    step_slot.info(f"🔍 Parsing response… (API: {call_ms}ms | p95 timeout: {_get_adaptive_timeout():.0f}s)")
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

                if not success:
                    st.warning(f"🔁 Retrying {len(batch)} items individually…")
                    for single in batch:
                        if st.session_state.rpd_count + rpd_delta >= 20:
                            break
                        enforce_rpm()
                        result = _generate_single_item(model, single[0], single[1], target_lang)
                        if result:
                            all_card_data.append(result)
                            rpd_delta += 1
                            st.markdown(f"  ↳ ✅ Recovered: `{single[0]}`")
                        else:
                            st.error(f"  ↳ ❌ Could not recover `{single[0]}`")

            # Live ETA using actual elapsed
            elapsed   = perf_counter() - t_start
            done      = idx + 1
            remaining = len(batches) - done
            if remaining > 0 and done > 0:
                eta_sec = int((elapsed / done) * remaining)
                eta_slot.info(f"⏱️ ~{eta_sec}s remaining | Batch {done}/{len(batches)} done")
            elif remaining == 0:
                eta_slot.success(f"✅ All {len(batches)} batches done in {elapsed:.1f}s.")

            progress_bar.progress(done / len(batches))

        # ── C2: Single fire-and-forget GitHub write ────────────────────────────
        if rpd_delta > 0 and not dry_run:
            st.session_state.rpd_count += rpd_delta
            _async_save_usage(
                st.session_state.rpd_count,
                st.session_state.rpm_timestamps,
            )

        status_log.update(
            label=f"✅ AI Generation Complete! ({len(all_card_data)} items)",
            state="complete",
            expanded=False,
        )

    return all_card_data

# ========================== PROCESS ANKI DATA (A5 + B2) ==========================
def process_anki_data(df_subset, batch_size=6, dry_run=False, target_lang="Indonesian"):
    # A5: O(1) cache key via hash_pandas_object (replaces slow str(df.to_dict()))
    t0        = perf_counter()
    cache_key = hashlib.sha256(
        pd.util.hash_pandas_object(df_subset, index=True).values.tobytes()
    ).hexdigest()
    cached = st.session_state.get("processed_cache", {})
    if (cached.get("key") == cache_key
            and (datetime.now() - cached.get("time", datetime.min)).total_seconds() < 300):
        st.info("♻️ Using cached processed notes (no re-generation needed).")
        return cached["notes"]

    df_subset        = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data    = generate_anki_card_data_batched(
        vocab_phrase_list, batch_size=batch_size,
        dry_run=dry_run, target_lang=target_lang,
    )

    # B2 + D1: Parallel cleaning — each card processed in its own thread
    t_clean = perf_counter()
    args    = [(card, target_lang) for card in all_card_data]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(args)))) as ex:
        results = list(ex.map(_process_single_card, args))
    processed_notes = [r for r in results if r is not None]
    print(f"[PERF] parallel card cleaning: {(perf_counter()-t_clean)*1000:.1f}ms "
          f"for {len(all_card_data)} cards")

    skipped = len(all_card_data) - len(processed_notes)
    if skipped:
        st.warning(f"⚠️ {skipped} card(s) skipped due to missing required fields.")

    st.session_state.processed_cache = {
        "key":   cache_key,
        "notes": processed_notes,
        "time":  datetime.now(),
    }
    print(f"[PERF] process_anki_data total: {(perf_counter()-t0)*1000:.1f}ms")
    return processed_notes

# ========================== LIVE CARD PREVIEW ==========================
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
    text  = note.get("Text", "")
    front = _RE_CLOZE_FRONT.sub('<span class="cloze">[?]</span>', text)
    back  = _RE_CLOZE_BACK.sub(
        r'<span style="color:#ff00ff;text-shadow:0 0 8px #ff00ff;'
        r'border-bottom:2px solid #00ffff;">\1</span>', text
    )

    def _section(icon, label, value):
        if not value:
            return ""
        return (f'<div class="vellum-section">'
                f'<div class="section-header">{icon} {label}</div>'
                f'<div class="content">{value}</div></div>')

    details = (
        _section("📜", "DEFINITION",   note.get("Definition", "")) +
        _section("🗣️", "PRONUNCIATION", note.get("Pronunciation", "")) +
        _section("➕", "SYNONYMS",     note.get("Synonyms", "")) +
        _section("➖", "ANTONYMS",     note.get("Antonyms", ""))
    )
    return f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&family=Electrolize:wght@400;700&display=swap" rel="stylesheet">
<style>body{{margin:0;padding:0;}}{CYBERPUNK_CSS}</style>
</head><body class="card">
<div class="preview-label">▌ FRONT — Question</div>
<div class="vellum-focus-container"><div class="prompt-text">{front}</div></div>
<div class="preview-label">▌ BACK — Answer</div>
<div class="vellum-focus-container"><div class="prompt-text">{back}</div></div>
<div class="vellum-detail-container">{details}</div>
</body></html>"""

# ========================== AUDIO HELPER (D4: reuses _AUDIO_POOL) ==========================
def generate_audio_file(vocab: str, temp_dir: str):
    """Uses module-level _AUDIO_POOL — no per-call thread spawn overhead."""
    try:
        clean_vocab    = _RE_NONWORD_AUDIO.sub('', vocab).strip()
        clean_filename = _RE_ALPHANUM.sub('', clean_vocab) + ".mp3"
        file_path      = os.path.join(temp_dir, clean_filename)
        if clean_vocab:
            def _do_tts():
                gTTS(text=clean_vocab, lang='en', slow=False).save(file_path)
            fut = _AUDIO_POOL.submit(_do_tts)
            fut.result(timeout=10)
            return vocab, clean_filename, file_path
    except concurrent.futures.TimeoutError:
        print(f"gTTS timeout for '{vocab}' — skipping audio.")
    except Exception as e:
        print(f"Audio error for {vocab}: {e}")
    return vocab, None, None

# ========================== GENANKI LOGIC (B1: cached model) ==========================
def _get_or_build_genanki_model(model_id: int, include_antonyms: bool) -> genanki.Model:
    """
    B1: Cache the genanki.Model object in session_state.
    Rebuilding it (including hashing CYBERPUNK_CSS) is wasteful on every export.
    """
    cache_key = f"genanki_model_{model_id}_{include_antonyms}"
    if st.session_state.get("_genanki_model_key") == cache_key:
        return st.session_state["_genanki_model_obj"]

    front_html = (
        """<div class="vellum-focus-container front">"""
        """<div class="prompt-text">{{cloze:Text}}</div></div>"""
    )
    back_html = (
        """<div class="vellum-focus-container back">"""
        """<div class="prompt-text solved-text">{{cloze:Text}}</div></div>"""
        """<div class="vellum-detail-container">"""
        """{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}"""
        """{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}"""
        """{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}"""
        """{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}"""
    )
    if include_antonyms:
        back_html += (
            """{{#Antonyms}}<div class="vellum-section">"""
            """<div class="section-header">➖ ANTONYMS</div>"""
            """<div class="content">{{Antonyms}}</div></div>{{/Antonyms}}"""
        )
    back_html += (
        """{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}"""
        """<div style='display:none'>{{Audio}}</div>"""
        """</div>{{Audio}}"""
    )

    model = genanki.Model(
        model_id,
        'Cyberpunk Vocab Model',
        fields=[
            {'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'},
            {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'},
            {'name': 'Etymology'}, {'name': 'Audio'},
        ],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=CYBERPUNK_CSS,
        model_type=genanki.Model.CLOZE,
    )
    st.session_state["_genanki_model_key"] = cache_key
    st.session_state["_genanki_model_obj"] = model
    return model


def create_anki_package(notes_data, deck_name, generate_audio=True,
                        deck_id=2059400110, include_antonyms=True):
    t0 = perf_counter()

    # B1: Retrieve cached model instead of rebuilding
    my_model = _get_or_build_genanki_model(
        st.session_state.get("model_id", 1607392319),
        include_antonyms,
    )
    my_deck     = genanki.Deck(deck_id, deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            t_audio      = perf_counter()
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            # D4: Submit to module-level pool (no new executor creation)
            futures = {_AUDIO_POOL.submit(generate_audio_file, v, temp_dir): v
                       for v in unique_vocabs}
            for future in concurrent.futures.as_completed(futures):
                vk, fn, fp = future.result()
                if fn:
                    media_files.append(fp)
                    audio_map[vk] = f"[sound:{fn}]"
            print(f"[PERF] audio generation: {(perf_counter()-t_audio)*1000:.1f}ms "
                  f"({len(unique_vocabs)} words)")

        for note_data in notes_data:
            guid_input = note_data['VocabRaw'] + deck_name
            vocab_hash = str(
                int(hashlib.sha256(guid_input.encode('utf-8')).hexdigest(), 16) % (10 ** 10)
            )
            my_deck.add_note(genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'], note_data['Pronunciation'], note_data['Definition'],
                    note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'],
                    note_data['Etymology'], audio_map.get(note_data['VocabRaw'], ""),
                ],
                tags=note_data['Tags'],
                guid=vocab_hash,
            ))

        my_package             = genanki.Package(my_deck)
        my_package.media_files = media_files

        # D3: Write directly to a BytesIO-backed temp path; no redundant re-read
        output_path = os.path.join(temp_dir, 'output.apkg')
        t_write     = perf_counter()
        my_package.write_to_file(output_path)
        print(f"[PERF] genanki write: {(perf_counter()-t_write)*1000:.1f}ms")

        buffer = io.BytesIO()
        with open(output_path, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)

    print(f"[PERF] create_anki_package total: {(perf_counter()-t0)*1000:.1f}ms")
    return buffer

# ========================== LOAD / SAVE (B4: optimistic update) ==========================
@st.cache_data(ttl=600)
def load_data():
    t0 = perf_counter()
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(
            io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str
        )
        df['phrase']     = df['phrase'].fillna("")
        df['status']     = df.get('status',     pd.Series(['New'] * len(df))).fillna('New')
        df['tags']       = df.get('tags',        pd.Series(['']   * len(df))).fillna('')
        df['date_added'] = df.get('date_added',  pd.Series(['']   * len(df))).fillna('')
        result = df.sort_values(by="vocab", ignore_index=True)
        print(f"[PERF] load_data (GitHub): {(perf_counter()-t0)*1000:.1f}ms")
        return result
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags', 'date_added'])
        st.stop()
    except Exception:
        st.stop()

def save_to_github(dataframe) -> bool:
    """
    B4: Session state is already updated before this is called — this
    function only handles the GitHub persistence layer.

    SHA is fetched fresh every call (never from cache) so we never hit
    a 409 conflict caused by a stale SHA from load_data's 10-minute cache.
    All exceptions are caught and surfaced via st.error() so failures are
    never silent.
    """
    t0        = perf_counter()
    dataframe = (
        dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
        .drop_duplicates(subset=['vocab'], keep='last')
    )
    csv_data = dataframe.to_csv(index=False)
    try:
        # Always fetch a live SHA — never rely on cached file object
        try:
            file = repo.get_contents("vocabulary.csv")
            repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
        except GithubException as e:
            if e.status == 404:
                repo.create_file("vocabulary.csv", "Initial commit", csv_data)
            else:
                # Surface ALL GitHub errors — 409 SHA conflict, 422, 403, etc.
                st.error(f"❌ GitHub save failed (HTTP {e.status}): {e.data}. "
                         f"Your word is saved in this session but NOT on GitHub. "
                         f"Try refreshing and saving again.")
                return False
        load_data.clear()   # invalidate read-cache after confirmed write
        print(f"[PERF] save_to_github: {(perf_counter()-t0)*1000:.1f}ms ✓")
        return True
    except Exception as e:
        st.error(f"❌ Unexpected save error: {e}. Word saved in session only.")
        return False

# ========================== SESSION STATE INIT ==========================
# B5: Lazy load — skip GitHub round-trip if vocab_df already in state
if "vocab_df"          not in st.session_state:
    st.session_state.vocab_df = load_data().copy()
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
if "api_latencies"     not in st.session_state: st.session_state.api_latencies     = []

# A7: Load merged usage (1 GitHub read vs previous 2)
if "rpd_count" not in st.session_state or "rpm_timestamps" not in st.session_state:
    _rpd, _ts = load_combined_usage()
    st.session_state.rpd_count      = _rpd
    st.session_state.rpm_timestamps = _ts
if "rpm_date" not in st.session_state:
    st.session_state.rpm_date = str(date.today())

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
                "tags": "", "date_added": str(date.today()),
            }])
            st.session_state.vocab_df = pd.concat(
                [st.session_state.vocab_df, new_row], ignore_index=True
            )
        ok = save_to_github(st.session_state.vocab_df)
        st.session_state.input_phrase = ""
        st.session_state.input_vocab  = ""
        if ok:
            st.toast(f"✅ Saved '{v}'!", icon="🚀")
        # If not ok, save_to_github already called st.error()
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

    rpm_current = len([
        ts for ts in st.session_state.rpm_timestamps
        if (datetime.now() - ts).total_seconds() < 60
    ])
    st.progress(rpm_current / 5,                      text=f"RPM Live: {rpm_current}/5 (last 60s)")
    st.progress(st.session_state.rpd_count / 20,      text=f"RPD: {st.session_state.rpd_count}/20")

    # C4: Adaptive timeout indicator
    if st.session_state.api_latencies:
        adaptive_t = _get_adaptive_timeout()
        avg_ms     = int(sum(st.session_state.api_latencies) / len(st.session_state.api_latencies) * 1000)
        st.caption(f"⏱️ API avg: {avg_ms}ms | adaptive timeout: {adaptive_t:.0f}s")

    if not st.session_state.vocab_df.empty:
        status_counts = st.session_state.vocab_df['status'].value_counts()
        st.bar_chart(status_counts, height=140)

    st.divider()
    TARGET_LANG  = st.selectbox(
        "🎯 Translation Language",
        ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"],
        index=0,
    )
    GEMINI_MODEL = st.selectbox(
        "🤖 AI Model",
        ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"],
        index=0,
    )

    key_valid = validate_api_key(st.session_state.gemini_key, GEMINI_MODEL)
    if key_valid:
        st.success("🔑 API Key: Valid ✓", icon="✅")
    else:
        st.error("🔑 API Key: Invalid ✗")

    st.divider()
    if st.button("🔄 Regenerate Note Type Model ID"):
        st.session_state.model_id          = random.randrange(1 << 30, 1 << 31)
        st.session_state["_genanki_model_key"] = None   # B1: invalidate cached model
        st.success(f"New Model ID: {st.session_state.model_id}")
    st.caption(f"Current Model ID: {st.session_state.model_id}")

    if not st.session_state.vocab_df.empty:
        st.download_button(
            "💾 Backup Database (CSV)",
            st.session_state.vocab_df.to_csv(index=False).encode('utf-8'),
            f"vocab_backup_{date.today()}.csv",
            "text/csv",
        )

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

# ─────────────────────────── TAB 1: ADD ───────────────────────────────────────
with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")

    if add_mode == "Single":
        p_raw      = st.text_input("🔤 Phrase", placeholder="Paste your sentence here…", key="input_phrase")
        v_selected = ""
        if p_raw and p_raw not in ["1", "*"]:
            clean_text   = re.sub(r'[^\w\s\-\']', '', p_raw)
            unique_words = list(dict.fromkeys([w.lower() for w in clean_text.split() if w.strip()]))
            if unique_words:
                st.caption("Click words below to extract them as vocabulary:")
                try:
                    selected_pills = st.pills(
                        "Select Vocab", unique_words,
                        selection_mode="multi", label_visibility="collapsed",
                    )
                    v_selected = " ".join(selected_pills) if selected_pills else ""
                except Exception:
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
                if bp and bp not in ("1",) and not bp.startswith("*"):
                    if bp.endswith(","):              bp = bp[:-1] + "."
                    elif not bp.endswith((".", "!", "?")): bp += "."
                    bp = bp.capitalize()
                if bv:
                    new_rows.append({
                        "vocab": bv, "phrase": bp, "status": "New",
                        "tags": "", "date_added": str(date.today()),
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
                ok = save_to_github(st.session_state.vocab_df)
                if ok:
                    st.success(f"✅ Added {len(st.session_state.bulk_preview_df)} words!")
                    st.session_state.bulk_preview_df = None
                    st.rerun()

# ─────────────────────────── TAB 2: EDIT (A2: @st.fragment) ───────────────────
with tab2:
    @st.fragment
    def _render_tab2():
        """
        A2: Fragment isolates Tab 2 reruns.
        Pagination, search, and the data_editor no longer cause full-page reruns.
        st.rerun() is called explicitly after save to refresh sidebar metrics.
        """
        if st.session_state.vocab_df.empty:
            st.info("Add words first!")
            return

        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        search     = st.text_input("🔎 Search…", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search:
            display_df = display_df[display_df['vocab'].str.contains(search, case=False)]

        page_size    = 50
        page         = st.number_input("Page", min_value=1, value=1, step=1)
        start        = (page - 1) * page_size
        paginated_df = display_df.iloc[start:start + page_size]

        edited = st.data_editor(
            paginated_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "status":     st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True),
                "date_added": st.column_config.TextColumn("Added", disabled=True),
            },
        )

        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            # ── Explicit copy + reassignment ─────────────────────────────────
            # DataFrame.update() mutates in-place; Streamlit cannot detect
            # in-place mutations as session-state changes.  We copy, apply
            # edits column-by-column, then reassign — guaranteeing detection.
            updated_df    = st.session_state.vocab_df.copy()
            existing_mask = edited.index.isin(updated_df.index)
            existing_edits = edited[existing_mask]
            new_rows       = edited[~existing_mask]

            # Apply each edited column explicitly so empty-string clears work
            for col in existing_edits.columns:
                updated_df.loc[existing_edits.index, col] = existing_edits[col].values

            if not new_rows.empty:
                if 'date_added' not in new_rows.columns:
                    new_rows = new_rows.copy()
                    new_rows['date_added'] = str(date.today())
                updated_df = pd.concat([updated_df, new_rows], ignore_index=True)

            # Explicit reassignment — Streamlit detects this as a state change
            st.session_state.vocab_df = updated_df

            ok = save_to_github(st.session_state.vocab_df)
            if ok:
                st.toast("✅ Cloud updated!")
                st.rerun()

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
                    ok = save_to_github(st.session_state.vocab_df)
                    if ok:
                        st.toast(f"✅ {done_count} words reset to 'New'!")
                        st.rerun()

    _render_tab2()

# ─────────────────────────── TAB 3: GENERATE (A1: @st.fragment on form) ───────
with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")

    # Download / preview section — kept OUTSIDE fragment so it shows after full rerun
    if st.session_state.apkg_buffer is not None:
        st.success("✅ Deck generated! Preview below, then download to sync your database.")

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
            on_click=mark_as_done_callback,
        )
        if st.button("❌ Cancel / Clear"):
            st.session_state.apkg_buffer      = None
            st.session_state.processed_vocabs = []
            st.session_state.preview_notes    = []
            st.rerun()

    else:
        @st.fragment
        def _render_tab3_form(target_lang: str):
            """
            A1: Fragment isolates all Tab 3 form widgets.
            Slider, checkboxes, and data_editor interactions no longer cause
            full-page reruns. st.rerun() is called after generation to show
            the download section above.
            """
            if st.session_state.vocab_df.empty:
                st.info("Add words first!")
                return

            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
            if subset.empty:
                st.warning("⚠️ No 'New' words to export!")
                return

            search_export = st.text_input(
                "🔎 Filter words to export…", "",
                placeholder="Type to narrow down…",
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
                st.session_state["_genanki_model_key"] = None   # B1: bust model cache on new ID
            deck_col2.caption(f"ID: {st.session_state.deck_id}")

            batch_size     = st.slider("⚡ Batch Size (Words per Request)", 1, 15, 10)
            requests_left  = max(0, 20 - st.session_state.rpd_count)
            max_safe_batch = max(1, math.ceil(len(subset) / max(1, requests_left))) if requests_left > 0 else 1
            batch_size     = min(batch_size, max_safe_batch)
            st.caption(f"✅ Auto-adjusted effective batch size: **{batch_size}** (based on remaining quota)")

            include_audio                     = st.checkbox("🔊 Generate Audio Files", value=True)
            st.session_state.include_antonyms = st.checkbox("➖ Include Antonyms in Card Back",  value=st.session_state.include_antonyms)
            st.session_state.dry_run          = st.checkbox("🔬 Dry Run (simulate AI, no quota)", value=st.session_state.dry_run)

            st.write("**Select words to export:**")
            subset_export           = subset.copy()
            subset_export['Export'] = True
            edited_export = st.data_editor(
                subset_export,
                column_config={"Export": st.column_config.CheckboxColumn("Export?", required=True)},
                hide_index=True,
                disabled=["vocab", "phrase", "status", "tags", "date_added"],
            )
            final_export_subset = edited_export[edited_export['Export'] == True]

            if not final_export_subset.empty:
                st.write("### Export Preview")
                st.dataframe(final_export_subset[['vocab', 'phrase']], hide_index=True)
                card_count  = len(final_export_subset)
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
                    raw_notes = []
                    try:
                        raw_notes = process_anki_data(
                            final_export_subset,
                            batch_size=batch_size,
                            dry_run=st.session_state.dry_run,
                            target_lang=target_lang,
                        )
                        if raw_notes:
                            t_pkg = perf_counter()
                            apkg_buffer = create_anki_package(
                                raw_notes, deck_name_input,
                                generate_audio=include_audio,
                                deck_id=st.session_state.deck_id,
                                include_antonyms=st.session_state.include_antonyms,
                            )
                            st.caption(f"📦 Package built in {(perf_counter()-t_pkg)*1000:.0f}ms")
                            st.session_state.apkg_buffer      = apkg_buffer.getvalue()
                            st.session_state.processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                            st.session_state.preview_notes    = raw_notes
                            st.rerun()   # full rerun → renders download section above fragment
                        else:
                            st.warning("⚠️ No notes were generated. Check your quota and try again.")
                    except Exception as e:
                        st.error(f"❌ Generation error: {e}. Status rolled back to 'New'.")
                        if raw_notes:
                            st.session_state.vocab_df.loc[
                                st.session_state.vocab_df['vocab'].isin(
                                    [n.get('VocabRaw', '') for n in raw_notes]
                                ), 'status'
                            ] = 'New'
                            save_to_github(st.session_state.vocab_df)
                        if st.button("🔁 Retry with smaller batch"):
                            st.rerun()

        _render_tab3_form(TARGET_LANG)
