import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import random
from datetime import date, datetime, timezone, timedelta
import google.generativeai as genai
import json
import re
import time
import os
import tempfile
import hashlib
import concurrent.futures
import math

try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")

THEME_COLOR = "#00ff41"
THEME_GLOW  = "rgba(0, 255, 65, 0.4)"
BG_COLOR    = "#111111"
BG_STRIPE   = "#181818"
TEXT_COLOR  = "#aaffaa"

# ========================== Y4-D: THEME CSS ==========================
LIGHT_MODE_CSS = """<style>
.stApp { background-color: #f0f7f0 !important; }
section[data-testid="stSidebar"] { background-color: #d8eed8 !important; }
section[data-testid="stSidebar"] * { color: #1a3a1a !important; }
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li { color: #1a3a1a !important; }
.stTextInput input, .stTextArea textarea { background-color: #e8f5e8 !important; color: #1a3a1a !important; }
</style>"""

# ========================== X1-B: STARTUP TIME PROFILER ==========================
# Module-level timestamps — captured before any GH API calls
_BOOT_T0 = time.perf_counter()

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

# ========================== SECRETS ==========================
try:
    token              = st.secrets["GITHUB_TOKEN"]
    repo_name          = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

# ========================== B8: MODULE-LEVEL PRE-COMPILED REGEX ==========================
_RE_SPACES       = re.compile(r"\s+")
_RE_SENT_SPLIT   = re.compile(r'(?<=[.!?])\s+')
_RE_JSON_FENCE_S = re.compile(r"^```(?:json)?\s*")
_RE_JSON_FENCE_E = re.compile(r"\s*```$")
_RE_JSON_ARRAY   = re.compile(r'\[.*\]', re.DOTALL)
_RE_CLEAN_TEXT   = re.compile(r'[^\w\s\-\']')
_RE_CLEAN_FNAME  = re.compile(r'[^a-zA-Z0-9]')
_RE_AUDIO_CLEAN  = re.compile(r'[^a-zA-Z0-9\s\-\']')
_RE_DECK_ILLEGAL = re.compile(r'[<>"/\\|?*]')
_RE_TAG_CLEAN    = re.compile(r'[^\w\-]')
_RE_STRIP_HTML   = re.compile(r'<[^>]+>')

_GRAMMAR_RULES = [
    (re.compile(r"\bto doing\b",      re.IGNORECASE), "to do"),
    (re.compile(r"\bfor helps\b",     re.IGNORECASE), "to help"),
    (re.compile(r"\bis use to\b",     re.IGNORECASE), "is used to"),
    (re.compile(r"\bhelp for to\b",   re.IGNORECASE), "help to"),
    (re.compile(r"\bfor to\b",        re.IGNORECASE), "to"),
    (re.compile(r"\bcan able to\b",   re.IGNORECASE), "can"),
    (re.compile(r"\bI am agree\b",    re.IGNORECASE), "I agree"),
    (re.compile(r"\bdiscuss about\b", re.IGNORECASE), "discuss"),
    (re.compile(r"\breturn back\b",   re.IGNORECASE), "return"),
]

# ========================== T3-C: SUBJECT PERSONAS ==========================
PERSONAS: dict[str, str] = {
    "General":           "",
    "Medical":           "You are a medical lexicographer. Use precise clinical terminology. ",
    "Legal":             "You are a legal lexicographer. Use precise juridical definitions. ",
    "Coding / Tech":     "You are a software-engineering lexicographer. Use technical CS context. ",
    "Language Learning": "You are an EFL/ESL teacher. Prioritize learner-friendly definitions and natural example sentences. ",
}

# ========================== T3-D: DIFFICULTY MODIFIERS ==========================
DIFFICULTY_SUFFIX: dict[str, str] = {
    "Beginner":     "Use simple vocabulary (A1-A2 CEFR). Short example sentences. Avoid jargon.",
    "Intermediate": "",
    "Advanced":     "Use sophisticated vocabulary (C1-C2 CEFR). Complex, nuanced examples with idioms.",
}

# ========================== Z4-E: SECOND LANGUAGE OPTIONS ==========================
LANG2_OPTIONS = [
    "None (disabled)",
    "Indonesian", "Spanish", "French", "German",
    "Japanese", "Chinese (Mandarin)", "English (Simple)",
]

# ========================== X3-C: REGISTER LABELS ==========================
REGISTER_VALUES = ["Formal", "Informal", "Slang", "Technical", "Neutral"]
REGISTER_BADGE_CSS: dict[str, str] = {
    "Formal":    "color:#00ffff;border:1px solid #00ffff",
    "Informal":  "color:#ffff66;border:1px solid #ffff66",
    "Slang":     "color:#ff6b6b;border:1px solid #ff6b6b",
    "Technical": "color:#c084fc;border:1px solid #c084fc",
    "Neutral":   "color:#aaffaa;border:1px solid #aaffaa",
}

# ========================== Z2-E: CARD BACK SECTION ORDER ==========================
BACK_SECTIONS_DEFAULT = [
    "Romanization", "Translation2", "Definition", "Pronunciation",
    "Register", "Examples", "Collocations", "Synonyms",
    "Antonyms", "Etymology", "Mnemonic",
]
BACK_SECTION_META: dict[str, tuple] = {
    "Romanization":  ("🈳 ROMANIZATION",  "Romanization",  False),
    "Translation2":  ("🌐 TRANSLATION 2", "Translation2",  False),
    "Definition":    ("📜 DEFINITION",    "Definition",    False),
    "Pronunciation": ("🗣️ PRONUNCIATION", "Pronunciation", False),
    "Register":      ("🏷 REGISTER",      "Register",      False),
    "Examples":      ("🖋️ EXAMPLES",     "Examples",      False),
    "Collocations":  ("🔗 COLLOCATIONS",  "Collocations",  False),
    "Synonyms":      ("➕ SYNONYMS",      "Synonyms",      False),
    "Antonyms":      ("➖ ANTONYMS",      "Antonyms",      True),
    "Etymology":     ("🏛️ ETYMOLOGY",    "Etymology",     False),
    "Mnemonic":      ("💡 MEMORY HOOK",   "Mnemonic",      False),
}

def build_back_html(section_order: list, include_antonyms: bool) -> str:
    """Z2-E: Dynamically builds back_html from user-specified section order."""
    html = """<div class="vellum-focus-container back">
<div class="prompt-text solved-text">{{cloze:Text}}</div></div>
<div class="vellum-detail-container">"""
    for key in section_order:
        meta = BACK_SECTION_META.get(key)
        if not meta:
            continue
        header, field, antonyms_only = meta
        if antonyms_only and not include_antonyms:
            continue
        html += f"""
{{{{#{field}}}}}<div class="vellum-section">
<div class="section-header">{header}</div>
<div class="content">{{{{{field}}}}}</div></div>{{{{/{field}}}}}"""
    html += """
<div style='display:none'>{{Audio}}</div>
</div>"""
    return html

# ========================== T1-D: TPM PRE-FLIGHT THRESHOLDS ==========================
TPM_WARN_THRESHOLD  = 700_000
TPM_BLOCK_THRESHOLD = 850_000

# ========================== X1-A: GITHUB WRITE COST TRACKER ==========================
_GH_WRITE_LOG: list = []
GH_WRITE_WARN_THRESHOLD = 80

def _gh_write_tick():
    """X1-A: Thread-safe GH write counter (list.append is GIL-atomic)."""
    _GH_WRITE_LOG.append(1)

def gh_write_count() -> int:
    return len(_GH_WRITE_LOG)

# ========================== Y3-E: WORD FREQUENCY HEURISTIC ==========================
def word_frequency_label(vocab: str) -> str:
    n = len(str(vocab).strip())
    if n <= 5:  return "🟢 Common"
    if n <= 9:  return "🟡 Uncommon"
    return "🔴 Rare"

# ========================== N2-D: CARD QUALITY SCORER ==========================
QUALITY_WARN_THRESHOLD = 60

def score_card(note_data: dict) -> int:
    score = 0
    defn  = _RE_STRIP_HTML.sub(' ', note_data.get("Definition", ""))
    if defn and len(defn.split()) >= 10:  score += 30
    if note_data.get("Examples", ""):     score += 30
    if note_data.get("Synonyms", ""):     score += 20
    if "/" in note_data.get("Pronunciation", ""):  score += 20
    return score

def quality_badge(score: int) -> str:
    if score >= 80: return "🟢"
    if score >= 60: return "🟡"
    return "🔴"

# ========================== Y3-B: VOCAB GAP DETECTOR ==========================
def detect_vocab_gaps(word_cache: dict) -> list:
    if len(word_cache) < 10:
        return []
    syn_map: dict[str, set] = {}
    for vocab, data in word_cache.items():
        syn_ant = data.get("synonyms_antonyms") or {}
        syns    = {s.lower().strip() for s in (syn_ant.get("synonyms") or []) if s.strip()}
        if len(syns) >= 2:
            syn_map[vocab] = syns
    clusters, visited, vocabs = [], set(), list(syn_map.keys())
    for i, v1 in enumerate(vocabs):
        if v1 in visited: continue
        cluster = [v1]
        for v2 in vocabs[i + 1:]:
            if v2 not in visited and len(syn_map[v1] & syn_map.get(v2, set())) >= 2:
                cluster.append(v2)
        if len(cluster) >= 2:
            for v in cluster: visited.add(v)
            shared = syn_map[v1].copy()
            for v in cluster[1:]: shared &= syn_map.get(v, shared)
            clusters.append({"words": cluster[:5], "shared_synonyms": sorted(shared)[:3]})
        if len(clusters) == 3: break
    return clusters

# ========================== A4: CYBERPUNK CSS ==========================
CYBERPUNK_CSS = f"""
.card {{
    font-family: 'Roboto Mono', 'Consolas', monospace;
    font-size: 18px; line-height: 1.5;
    color: {THEME_COLOR}; background-color: {BG_COLOR};
    background-image: repeating-linear-gradient(
        0deg, {BG_STRIPE}, {BG_STRIPE} 1px, {BG_COLOR} 1px, {BG_COLOR} 20px
    );
    padding: 30px 20px; text-align: left;
}}
.vellum-focus-container {{
    background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px;
    border: 2px solid {THEME_COLOR};
    box-shadow: 0 0 5px {THEME_COLOR}, 0 0 15px {THEME_GLOW};
    text-align: center;
}}
.prompt-text {{
    font-family: 'Electrolize', sans-serif; font-size: 1.8em; font-weight: 900;
    color: #ffffff;
    text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff;
}}
.cloze {{ color: {BG_COLOR}; background-color: {THEME_COLOR}; padding: 2px 4px; }}
.solved-text .cloze {{
    color: #ff00ff; background: none;
    border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff;
}}
.vellum-section {{ margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed {THEME_COLOR}; }}
.section-header {{
    font-weight: 600; color: #00ffff;
    border-left: 3px solid {THEME_COLOR}; padding-left: 10px;
}}
.content {{ color: {TEXT_COLOR}; padding-left: 13px; }}
.register-badge {{
    display: inline-block; font-size: 0.75em; font-weight: 700;
    padding: 1px 7px; border-radius: 3px; letter-spacing: 0.08em;
    text-transform: uppercase; margin-left: 6px;
}}
@media (max-width: 480px) {{
    .card {{ font-size: 16px; padding: 15px; }}
    .vellum-focus-container {{ padding: 15px; }}
}}
"""


# ========================== N2-C: CARD THEME DEFINITIONS ==========================

# ── MINIMAL THEME ──────────────────────────────────────────────────────────────
# Clean, paper-white aesthetic. Inter/system-ui sans-serif. Subtle warm shadows.
# High contrast for readability. Restrained colour palette: charcoal + accent blue.
MINIMAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.card {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 16px; line-height: 1.65; color: #1a1a2e;
    background: #fefefe;
    background-image: linear-gradient(135deg, #fefefe 0%, #f7f9fc 100%);
    padding: 28px 24px; text-align: left;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.vellum-focus-container {
    background: #ffffff; padding: 28px 24px; margin: 0 auto 32px;
    border-radius: 10px;
    border: 1.5px solid #e2e8f0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06), 0 0 0 1px rgba(99,102,241,0.08);
    text-align: center;
}
.prompt-text {
    font-family: 'Inter', sans-serif; font-size: 1.9em; font-weight: 700;
    color: #1a1a2e; letter-spacing: -0.02em; line-height: 1.2;
}
.cloze {
    color: #ffffff; background-color: #4f46e5;
    padding: 2px 8px; border-radius: 4px; font-weight: 600;
}
.solved-text .cloze {
    color: #4f46e5; background: rgba(79,70,229,0.08);
    border-bottom: 2px solid #4f46e5; font-weight: 600;
}
.vellum-section {
    margin: 12px 0; padding: 10px 0;
    border-bottom: 1px solid #f1f5f9;
}
.vellum-section:last-of-type { border-bottom: none; }
.section-header {
    font-size: 0.72em; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.content { color: #334155; padding-left: 0; font-size: 0.97em; }
.register-badge {
    display: inline-block; font-size: 0.68em; font-weight: 600;
    padding: 1px 8px; border-radius: 99px; letter-spacing: 0.06em;
    text-transform: uppercase; margin-left: 6px;
    background: rgba(99,102,241,0.1);
}
@media (max-width: 480px) {
    .card { font-size: 15px; padding: 16px; }
    .vellum-focus-container { padding: 16px; }
}
"""

# ── ACADEMIC THEME ─────────────────────────────────────────────────────────────
# Warm parchment tone. Merriweather serif headings + Source Serif body.
# Dictionary / textbook aesthetic. Deep navy + terracotta accent.
ACADEMIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700;900&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
.card {
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    font-size: 16px; line-height: 1.75; color: #2c2416;
    background: #fdf8ef;
    background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c8a96e' fill-opacity='0.06'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    padding: 28px 24px; text-align: left;
}
.vellum-focus-container {
    background: #fffdf7; padding: 24px 20px; margin: 0 auto 28px;
    border-radius: 2px;
    border-top: 4px solid #8b3a2a;
    border-bottom: 1px solid #d4b896;
    border-left: 1px solid #e8d9c0;
    border-right: 1px solid #e8d9c0;
    box-shadow: 0 2px 8px rgba(139,58,42,0.08);
    text-align: center;
}
.prompt-text {
    font-family: 'Merriweather', Georgia, serif; font-size: 1.75em; font-weight: 900;
    color: #1c1206; letter-spacing: -0.01em;
    text-shadow: none;
}
.cloze {
    color: #fffdf7; background-color: #8b3a2a;
    padding: 1px 6px; border-radius: 2px; font-style: normal;
}
.solved-text .cloze {
    color: #8b3a2a; background: rgba(139,58,42,0.08);
    border-bottom: 2px solid #8b3a2a;
}
.vellum-section {
    margin: 10px 0; padding: 8px 0;
    border-bottom: 1px dashed #d4b896;
}
.vellum-section:last-of-type { border-bottom: none; }
.section-header {
    font-family: 'Merriweather', serif; font-size: 0.68em;
    font-weight: 700; color: #8b3a2a;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.content { color: #2c2416; padding-left: 0; }
.register-badge {
    display: inline-block; font-size: 0.68em; font-weight: 600;
    padding: 1px 7px; border-radius: 2px; letter-spacing: 0.07em;
    text-transform: uppercase; margin-left: 6px;
    background: rgba(139,58,42,0.1);
}
@media (max-width: 480px) {
    .card { font-size: 15px; padding: 16px; }
    .vellum-focus-container { padding: 14px; }
}
"""

# ── THEME REGISTRY ──────────────────────────────────────────────────────────────
CARD_THEMES: dict[str, dict] = {
    "🟢 Cyberpunk": {
        "css":         None,          # filled below after CYBERPUNK_CSS is defined
        "description": "Dark matrix, glowing green borders",
        "front_color": "#ffffff",
        "accent":      "#00ff41",
    },
    "⬜ Minimal": {
        "css":         MINIMAL_CSS,
        "description": "Clean white, indigo accents, modern sans-serif",
        "front_color": "#1a1a2e",
        "accent":      "#4f46e5",
    },
    "📖 Academic": {
        "css":         ACADEMIC_CSS,
        "description": "Warm parchment, serif fonts, dictionary style",
        "front_color": "#1c1206",
        "accent":      "#8b3a2a",
    },
}

# N2-C: Fill Cyberpunk entry now that CYBERPUNK_CSS is defined
CARD_THEMES["🟢 Cyberpunk"]["css"] = CYBERPUNK_CSS

def get_active_css() -> str:
    """N2-C: Returns CSS string for the currently selected card theme."""
    theme_key = st.session_state.get("card_theme", "🟢 Cyberpunk")
    return CARD_THEMES.get(theme_key, CARD_THEMES["🟢 Cyberpunk"])["css"] or CYBERPUNK_CSS

# ========================== C12: BACKGROUND GITHUB EXECUTOR ==========================
@st.cache_resource
def _get_gh_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="gh_bg")

# ========================== GITHUB CONNECT ==========================
@st.cache_resource
def get_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except GithubException as e:
        st.error(f"❌ GitHub connection failed: {e}")
        st.stop()

_BOOT_T_GH_START = time.perf_counter()   # X1-B: before GH connect
repo = get_repo()
_BOOT_T_GH_DONE  = time.perf_counter()   # X1-B: after GH connect

# ========================== T1-A + T4-D: COMBINED USAGE FILE ==========================
_COMBINED_USAGE_FILE = "usage_combined.json"

def _legacy_load_rpd() -> int:
    try:
        file = repo.get_contents("usage.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return data.get("rpd_count", 0) if data.get("date") == str(date.today()) else 0
    except: return 0

def _legacy_load_rpm() -> list:
    try:
        file = repo.get_contents("usage_minute.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return [datetime.fromisoformat(ts) for ts in data.get("timestamps", [])]
    except: return []

def _bg_save_combined(rpd_count: int, timestamps: list):
    data = json.dumps({
        "date": str(date.today()), "rpd_count": rpd_count,
        "timestamps": [ts.isoformat() for ts in timestamps],
    })
    for attempt in range(3):
        try:
            try:
                file = repo.get_contents(_COMBINED_USAGE_FILE)
                repo.update_file(file.path, "Update combined usage", data, file.sha)
                _gh_write_tick()   # X1-A
                return
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(_COMBINED_USAGE_FILE, "Init combined usage", data)
                    _gh_write_tick()   # X1-A
                    return
                elif e.status == 409:
                    time.sleep(1 + attempt); continue
                raise
        except Exception as e:
            if attempt == 2: print(f"Combined usage save error: {e}")
            time.sleep(1)

def save_combined_usage(rpd_count: int, timestamps: list):
    _get_gh_executor().submit(_bg_save_combined, rpd_count, list(timestamps))

def load_combined_usage() -> tuple:
    try:
        file = repo.get_contents(_COMBINED_USAGE_FILE)
        data = json.loads(file.decoded_content.decode('utf-8'))
        rpd  = data.get("rpd_count", 0) if data.get("date") == str(date.today()) else 0
        tss  = [datetime.fromisoformat(ts) for ts in data.get("timestamps", [])]
        return rpd, tss
    except GithubException as e:
        if e.status == 404:
            rpd, tss = _legacy_load_rpd(), _legacy_load_rpm()
            save_combined_usage(rpd, tss)
            return rpd, tss
        return 0, []
    except: return 0, []

# ========================== T1-C: SAFETY BLOCK LOGGER ==========================
def _bg_log_safety_block(vocab_words: list, prompt_hash: str):
    entry = {"timestamp": datetime.now().isoformat(), "vocab": vocab_words, "prompt_hash": prompt_hash}
    try:
        existing, file_sha = [], None
        try:
            file     = repo.get_contents("safety_log.json")
            file_sha = file.sha
            existing = json.loads(file.decoded_content.decode('utf-8'))
            if not isinstance(existing, list): existing = []
        except GithubException as e:
            if e.status != 404: return
        existing.append(entry); existing = existing[-100:]
        data = json.dumps(existing, ensure_ascii=False, indent=2)
        if file_sha: repo.update_file("safety_log.json", "Safety block log", data, file_sha)
        else:        repo.create_file("safety_log.json", "Init safety log", data)
        _gh_write_tick()   # X1-A
    except Exception as e: print(f"Safety log write error: {e}")

def log_safety_block(vocab_words: list, prompt: str):
    _get_gh_executor().submit(_bg_log_safety_block, list(vocab_words),
                              hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16])

# ========================== T4-C: EXPORT HISTORY LOGGER ==========================
def _bg_save_export_history(deck_name: str, card_count: int, vocab_list: list):
    entry = {"timestamp": datetime.now().isoformat(), "deck_name": deck_name,
             "card_count": card_count, "vocabs": vocab_list[:50]}
    try:
        existing, file_sha = [], None
        try:
            file     = repo.get_contents("export_history.json")
            file_sha = file.sha
            existing = json.loads(file.decoded_content.decode('utf-8'))
            if not isinstance(existing, list): existing = []
        except GithubException as e:
            if e.status != 404: return
        existing.append(entry); existing = existing[-200:]
        data = json.dumps(existing, ensure_ascii=False, indent=2)
        if file_sha: repo.update_file("export_history.json", "Export history", data, file_sha)
        else:        repo.create_file("export_history.json", "Init export history", data)
        _gh_write_tick()   # X1-A
    except Exception as e: print(f"Export history write error: {e}")

def save_export_history(deck_name: str, card_count: int, vocab_list: list):
    _get_gh_executor().submit(_bg_save_export_history, deck_name, card_count, list(vocab_list))

# ========================== N4-E: EXPORT HISTORY LOADER ==========================
@st.cache_data(ttl=120)
def load_export_history() -> list:
    try:
        file = repo.get_contents("export_history.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return data if isinstance(data, list) else []
    except: return []

# ========================== Z1-C: WORD CACHE PERSISTENCE ==========================
_WORD_CACHE_FILE     = "word_cache.json"
_WORD_CACHE_TTL_DAYS = 30
_WORD_CACHE_MAX      = 500

def _bg_save_word_cache(word_cache: dict):
    now_str = datetime.now().isoformat()
    stamped = {}
    for k, v in word_cache.items():
        entry = dict(v)
        if '_cached_at' not in entry: entry['_cached_at'] = now_str
        stamped[k] = entry
    if len(stamped) > _WORD_CACHE_MAX:
        stamped = dict(list(stamped.items())[-_WORD_CACHE_MAX:])
    data = json.dumps(stamped, ensure_ascii=False)
    try:
        try:
            file = repo.get_contents(_WORD_CACHE_FILE)
            repo.update_file(file.path, "Update word cache", data, file.sha)
            _gh_write_tick()   # X1-A
        except GithubException as e:
            if e.status == 404:
                repo.create_file(_WORD_CACHE_FILE, "Init word cache", data)
                _gh_write_tick()   # X1-A
    except Exception as e:
        print(f"Word cache save error: {e}")

def save_word_cache(word_cache: dict):
    _get_gh_executor().submit(_bg_save_word_cache, dict(word_cache))

@st.cache_data(ttl=3600)
def load_word_cache() -> dict:
    try:
        file = repo.get_contents(_WORD_CACHE_FILE)
        data = json.loads(file.decoded_content.decode('utf-8'))
        if not isinstance(data, dict): return {}
        cutoff = datetime.now().timestamp() - (_WORD_CACHE_TTL_DAYS * 86400)
        pruned = {}
        for k, v in data.items():
            if not isinstance(v, dict): continue
            try:
                if datetime.fromisoformat(v.get('_cached_at', '')).timestamp() >= cutoff:
                    pruned[k] = v
            except: pruned[k] = v
        return pruned
    except: return {}

# ========================== A5: SMART RPM ENFORCEMENT ==========================
def enforce_rpm() -> float:
    t0  = time.perf_counter()
    now = datetime.now()
    st.session_state.rpm_timestamps = [
        ts for ts in st.session_state.rpm_timestamps
        if (now - ts).total_seconds() < 60
    ]
    save_combined_usage(st.session_state.rpd_count, st.session_state.rpm_timestamps)
    if len(st.session_state.rpm_timestamps) >= 5:
        _slot = st.empty()
        for r in range(12, 0, -1):
            _slot.warning(f"⏳ RPM limit (5/min). Resuming in **{r}s**...")
            time.sleep(1)
        _slot.empty()
    st.session_state.rpm_timestamps.append(now)
    save_combined_usage(st.session_state.rpd_count, st.session_state.rpm_timestamps)
    return time.perf_counter() - t0

# ========================== GEMINI ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
    except Exception as e:
        st.error(f"❌ Gemini key error: {e}"); return None

# ========================== B8: CLEANING FUNCTIONS ==========================
def cap_first(s: str) -> str:
    s = str(s).strip(); return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if s and s[-1] in ".!?" else (s + "." if s else "")

def normalize_spaces(text: str) -> str:
    return _RE_SPACES.sub(" ", str(text)).strip() if text else ""

def clean_grammar(text: str) -> str:
    if not isinstance(text, str): return text
    for pat, repl in _GRAMMAR_RULES: text = pat.sub(repl, text)
    return text

def cap_each_sentence(text: str) -> str:
    if not isinstance(text, str): return text
    return " ".join(cap_first(s) for s in _RE_SENT_SPLIT.split(text) if s.strip())

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    return re.compile(r'\b' + re.escape(vocab) + r'\b', re.IGNORECASE).sub(
        f'<b><u>{vocab}</u></b>', text)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    return re.compile(r'\b' + re.escape(vocab.lower()) + r'\b', re.IGNORECASE).sub(vocab, phrase)

def robust_json_parse(text: str):
    text = _RE_JSON_FENCE_S.sub("", text.strip())
    text = _RE_JSON_FENCE_E.sub("", text)
    try: return json.loads(text)
    except: pass
    m = _RE_JSON_ARRAY.search(text)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def normalize_phrase(p: str) -> str:
    p = p.strip()
    if not p or p == "1" or p.startswith("*"): return p
    if p.endswith(","): p = p[:-1] + "."
    elif not p.endswith((".", "!", "?")): p += "."
    return cap_first(p)

def _clean_field(text: str) -> str:
    return ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(text))))

def sanitize_tags(raw_tags: str) -> list:
    if not raw_tags or not raw_tags.strip(): return []
    tags = []
    for t in re.split(r'[,\s]+', raw_tags.strip()):
        t = t.strip()
        if not t: continue
        clean = _RE_TAG_CLEAN.sub('_', t)
        clean = re.sub(r'_+', '_', clean).strip('_')
        if clean: tags.append(clean)
    return tags[:10]

# ========================== T4-G + T1-D: TPM TRACKING ==========================
def log_tpm_chars(char_count: int):
    st.session_state.tpm_log.append({"ts": datetime.now(), "chars": char_count})

def get_rolling_tpm() -> int:
    now = datetime.now()
    st.session_state.tpm_log = [
        e for e in st.session_state.tpm_log if (now - e["ts"]).total_seconds() < 60
    ]
    return sum(e["chars"] for e in st.session_state.tpm_log) // 4

def check_tpm_preflight(prompt: str) -> tuple:
    projected = get_rolling_tpm() + len(prompt) // 4
    return projected < TPM_BLOCK_THRESHOLD, projected

# ========================== N1-C: QUOTA RESET COUNTDOWN ==========================
def quota_reset_countdown() -> str:
    utc_now  = datetime.now(timezone.utc)
    midnight = (utc_now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta    = midnight - utc_now
    hours, rem = divmod(int(delta.total_seconds()), 3600)
    return f"{hours}h {rem // 60}m"

# ========================== Y3-D: BULK PHRASE ENRICHMENT ==========================
def enrich_empty_phrases(vocab_list: list) -> dict:
    model = get_gemini_model(st.session_state.gemini_key,
                             st.session_state.get("gemini_model_name", "gemini-2.5-flash-lite"))
    if not model or not vocab_list: return {}
    capped = vocab_list[:15]
    prompt = (f"Generate ONE natural example sentence (max 15 words) for each vocabulary word below.\n"
              f'Return ONLY a JSON array: [{{"vocab": "word", "phrase": "sentence"}}, ...]\n'
              f"No commentary.\nWORDS: {json.dumps(capped)}")
    try:
        response = model.generate_content(prompt)
        st.session_state.rpd_count += 1
        save_combined_usage(st.session_state.rpd_count, st.session_state.rpm_timestamps)
        parsed = robust_json_parse(response.text)
        if isinstance(parsed, list):
            return {str(item.get("vocab", "")).strip().lower(): normalize_phrase(item.get("phrase", ""))
                    for item in parsed if item.get("vocab") and item.get("phrase")}
    except Exception as e:
        st.error(f"❌ Phrase enrichment failed: {e}")
    return {}

# ========================== C13 + C14: BATCH GENERATOR ==========================
def generate_anki_card_data_batched(
    vocab_phrase_list: list,
    batch_size: int = 6,
    dry_run: bool   = False
) -> list:
    TARGET_LANG    = st.session_state.get("target_lang", "Indonesian")
    model_name     = st.session_state.get("gemini_model_name", "gemini-2.5-flash-lite")
    model          = get_gemini_model(st.session_state.gemini_key, model_name)
    if not model: return []

    persona_prefix  = PERSONAS.get(st.session_state.get("persona", "General"), "")
    diff_str        = DIFFICULTY_SUFFIX.get(st.session_state.get("difficulty", "Intermediate"), "")
    use_mnemonic    = st.session_state.get("use_mnemonic", False)
    mnemonic_rule   = ("\n11. 'mnemonic': ONE memorable image or word-story hook to aid recall (max 20 words). "
                       "Example for SERENDIPITY: 'Imagine a SERENe DIP into a lucky pool.'"
                       if use_mnemonic else "")
    difficulty_rule = f"\n10. DIFFICULTY LEVEL: {diff_str} Tailor all outputs accordingly." if diff_str else ""
    is_cjk          = TARGET_LANG in ("Japanese", "Chinese (Mandarin)")
    romanization_rule = ("\n12. 'romanization': For Japanese provide Romaji. "
                         "For Chinese (Mandarin) provide Pinyin with tone marks."
                         if is_cjk else "")
    # Z4-E: Second language rule
    lang2           = st.session_state.get("target_lang2", "None (disabled)")
    lang2_rule      = (f"\n13. 'translation2': Provide ONLY the {lang2} translation of the 'vocab' word/phrase. "
                       f"NEVER translate the full example sentence."
                       if lang2 != "None (disabled)" else "")

    word_cache     = st.session_state.get("word_cache", {})
    cached_results = [word_cache[vp[0].strip().lower()]
                      for vp in vocab_phrase_list if vp[0].strip().lower() in word_cache]
    deduped_list   = [vp for vp in vocab_phrase_list if vp[0].strip().lower() not in word_cache]

    if cached_results:
        st.info(f"♻️ {len(cached_results)} word(s) served from cache — **zero quota used**.")
    if not deduped_list:
        return cached_results

    all_new_data = []
    batches      = [deduped_list[i:i + batch_size] for i in range(0, len(deduped_list), batch_size)]
    timings      = []

    # Y1-A: Restore from checkpoint
    checkpoint = st.session_state.get("generation_checkpoint", [])
    if checkpoint:
        all_new_data = list(checkpoint)
        # BUG-C: deduplicate against cached_results
        _ckpt_vocabs = {c.get('vocab', '').strip().lower() for c in all_new_data}
        cached_results = [c for c in cached_results if c.get('vocab', '').strip().lower() not in _ckpt_vocabs]

    with st.status("🤖 Processing AI Batches (RPM Throttled)...", expanded=True) as status_log:
        progress_bar = st.progress(0)

        for idx, batch in enumerate(batches):
            if st.session_state.rpd_count >= 20:
                st.warning("🛑 Daily AI Limit (20 requests) reached. Try again tomorrow.")
                for vp in batch: st.session_state.failed_words.append(vp[0])
                break

            t_rpm       = enforce_rpm()
            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            vocab_words = [v[0] for v in batch]

            prompt = f"""{persona_prefix}You are an expert educational lexicographer. Think step-by-step:
1. Identify primary sense from phrase or context.
2. Generate accurate fields.
3. Ensure JSON validity.

Output EXACTLY {len(batch_dicts)} items as a JSON array. No extra text or commentary.

SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.
RULES:
1. Copy ALL fields exactly.
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT.
3. IF 'phrase' is normal text: Use it ONLY to understand which meaning/nuance of the 'vocab' to use.
4. IF 'phrase' is empty: Generate ONE simple sentence (max 12 words) for the 'vocab'.
5. EXACT 'vocab' must remain unchanged.
6. MANDATORY: 'translation' must contain ONLY the {TARGET_LANG} translation of the 'vocab' word/phrase. NEVER translate the full example sentence.
7. 'part_of_speech' MUST be one of: Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection, Phrase.
8. 'collocations': provide exactly 2-3 of the most natural word combinations as a JSON array of strings.
9. 'register': MUST be exactly one of: Formal, Informal, Slang, Technical, Neutral.{difficulty_rule}{mnemonic_rule}{romanization_rule}{lang2_rule}

EXAMPLES:
[
  {{"vocab":"serendipity","phrase":"We found the perfect cafe by pure serendipity.","translation":"kebetulan","part_of_speech":"Noun","pronunciation_ipa":"/ˌsɛrənˈdɪpɪti/","definition_english":"The occurrence and development of events by chance in a happy or beneficial way.","example_sentences":["It was pure serendipity that we met."],"synonyms_antonyms":{{"synonyms":["chance","luck"],"antonyms":["misfortune"]}},"etymology":"Coined by Horace Walpole in 1754.","collocations":["by pure serendipity","happy serendipity","moment of serendipity"],"register":"Neutral","mnemonic":"Imagine a SERENe DIP into a lucky pool of fate.","romanization":"","translation2":""}},
  {{"vocab":"run","phrase":"*He decided to run for office","translation":"mencalonkan diri","part_of_speech":"Verb","pronunciation_ipa":"/rʌn/","definition_english":"To compete in an election.","example_sentences":["She will run for president."],"synonyms_antonyms":{{"synonyms":["campaign"],"antonyms":["withdraw"]}},"etymology":"Old English rinnan.","collocations":["run for office","run a campaign","run against"],"register":"Formal","mnemonic":"A person RUNning in a suit toward a ballot box.","romanization":"","translation2":""}}
]

BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

            log_tpm_chars(len(prompt))

            if not dry_run:
                _is_safe, _proj = check_tpm_preflight(prompt)
                if not _is_safe:
                    st.error(f"🛑 TPM Pre-flight blocked `{', '.join(vocab_words)}`: ~{_proj:,} tokens exceeds {TPM_BLOCK_THRESHOLD:,} limit.")
                    st.session_state.failed_words.extend(vocab_words)
                    timings.append({"batch": idx+1, "words": ", ".join(vocab_words),
                                    "rpm_wait_s": round(t_rpm, 3), "gemini_s": 0.0,
                                    "cached": False, "note": "TPM_BLOCKED"})
                    progress_bar.progress((idx + 1) / len(batches))
                    continue
                elif _proj > TPM_WARN_THRESHOLD:
                    st.warning(f"⚠️ TPM approaching limit: ~{_proj:,} / 1,000,000 projected tokens.")

            success     = False
            t_api_start = time.perf_counter()

            if dry_run:
                st.info(f"🔬 Dry-run: `{', '.join(vocab_words)}`")
                mock = [{"vocab": v[0], "phrase": v[1], "translation": "mock-" + v[0],
                         "part_of_speech": "Noun", "pronunciation_ipa": "/mock/",
                         "definition_english": "Simulated definition for testing purposes.",
                         "example_sentences": ["Mock example sentence for dry run."],
                         "synonyms_antonyms": {"synonyms": ["mock", "simulated"], "antonyms": []},
                         "etymology": "Simulated.", "collocations": ["mock one", "mock two"],
                         "register": "Neutral", "mnemonic": "Mock mnemonic.", "romanization": "",
                         "translation2": "mock-t2" if lang2 != "None (disabled)" else ""}
                        for v in batch]
                all_new_data.extend(mock)
                for card in mock: word_cache[card['vocab'].strip().lower()] = card
                success = True
            else:
                for attempt in range(3):
                    try:
                        response = model.generate_content(prompt)
                        st.session_state.rpd_count += 1
                        save_combined_usage(st.session_state.rpd_count, st.session_state.rpm_timestamps)
                        if hasattr(response, 'candidates') and response.candidates:
                            finish = str(response.candidates[0].finish_reason)
                            if finish in ("3", "SAFETY", "FinishReason.SAFETY"):
                                st.warning(f"🛡️ Safety filter blocked `{', '.join(vocab_words)}`. Logged.")
                                log_safety_block(vocab_words, prompt)
                                st.session_state.failed_words.extend(vocab_words)
                                break
                        parsed = robust_json_parse(response.text)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            all_new_data.extend(parsed)
                            for card in parsed: word_cache[card['vocab'].strip().lower()] = card
                            recovered = [c.get('vocab', '') for c in parsed]
                            missed    = [v for v in vocab_words if v not in recovered]
                            if missed:
                                st.session_state.failed_words.extend(missed)
                                st.warning(f"⚠️ Partial batch {idx+1}: {len(parsed)}/{len(batch_dicts)}. Missed: `{', '.join(missed)}`")
                            else:
                                st.markdown(f"✅ **Batch {idx+1}**: `{', '.join(vocab_words)}`")
                            st.session_state.generation_checkpoint = list(all_new_data)  # Y1-A
                            success = True
                            break
                    except Exception as e:
                        if "429" in str(e):
                            backoff = 20 + (2 ** attempt) + random.uniform(0, 1)
                            _slot   = st.empty()
                            for r in range(int(backoff), 0, -1):
                                _slot.warning(f"⚠️ 429 Rate Limit. Retrying in **{r}s**... (attempt {attempt+1}/3)")
                                time.sleep(1)
                            _slot.empty()
                        else:
                            time.sleep(2)

            t_api_elapsed = time.perf_counter() - t_api_start
            timings.append({"batch": idx+1, "words": ", ".join(vocab_words),
                            "rpm_wait_s": round(t_rpm, 3), "gemini_s": round(t_api_elapsed, 3),
                            "cached": False, "note": ""})
            if not success and not dry_run:
                st.error(f"❌ **Failed**: `{', '.join(vocab_words)}` — queued for retry")
                st.session_state.failed_words.extend(vocab_words)
            progress_bar.progress((idx + 1) / len(batches))

        total = len(all_new_data) + len(cached_results)
        status_log.update(label=f"✅ AI Complete! ({total} items | {len(cached_results)} cached)",
                          state="complete", expanded=False)

    if not st.session_state.failed_words:
        st.session_state.generation_checkpoint = []
    st.session_state.word_cache = word_cache
    save_word_cache(word_cache)   # Z1-C
    if timings:
        with st.expander("⏱️ Batch Performance Timings", expanded=False):
            st.dataframe(pd.DataFrame(timings), hide_index=True)
    return cached_results + all_new_data

# ========================== B9 + B10: PROCESS ANKI DATA ==========================
def process_anki_data(df_subset: pd.DataFrame, batch_size: int = 6, dry_run: bool = False) -> list:
    t0        = time.perf_counter()
    cache_key = str(pd.util.hash_pandas_object(df_subset).sum())
    cached    = st.session_state.get("processed_cache", {})
    if (cached.get("key") == cache_key
            and (datetime.now() - cached.get("time", datetime.min)).total_seconds() < 300):
        st.info("♻️ Using cached processed notes.")
        return cached["notes"]

    df_clean = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = (
        df_clean.reindex(columns=['vocab', 'phrase'], fill_value='')[['vocab', 'phrase']].values.tolist()
    )
    tags_lookup: dict[str, list] = {}
    if 'tags' in df_clean.columns:
        tags_series = (df_clean.assign(_vk=df_clean['vocab'].astype(str).str.strip().str.lower())
                       .set_index('_vk')['tags'].fillna(''))
        tags_lookup = {str(k): sanitize_tags(str(v)) for k, v in tags_series.items() if str(v).strip()}

    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size, dry_run=dry_run)
    processed_notes = []
    for card_data in all_card_data:
        required = ["vocab", "translation", "part_of_speech"]
        if not all(k in card_data and card_data[k] for k in required):
            st.error(f"⚠️ Missing required fields for `{card_data.get('vocab','?')}` — skipping"); continue
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue
        vocab_cap   = cap_first(vocab_raw)
        phrase      = fix_vocab_casing(_clean_field(card_data.get("phrase", "")), vocab_raw)
        formatted   = highlight_vocab(phrase, vocab_raw) if phrase else ""
        translation = _clean_field(card_data.get("translation", "?"))
        pos         = str(card_data.get("part_of_speech", "")).title()
        ipa         = card_data.get("pronunciation_ipa", "")
        eng_def     = _clean_field(card_data.get("definition_english", ""))
        examples    = [_clean_field(e) for e in (card_data.get("example_sentences") or [])[:3]]
        ex_field    = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""
        syn_ant     = card_data.get("synonyms_antonyms") or {}
        synonyms    = ensure_trailing_dot(", ".join(cap_first(s) for s in (syn_ant.get("synonyms") or [])[:5]))
        antonyms    = ensure_trailing_dot(", ".join(cap_first(a) for a in (syn_ant.get("antonyms") or [])[:5]))
        etymology   = normalize_spaces(card_data.get("etymology", ""))
        collocations_raw = card_data.get("collocations") or []
        if isinstance(collocations_raw, list): collocations = "; ".join(cap_first(c) for c in collocations_raw[:3] if c)
        elif isinstance(collocations_raw, str): collocations = cap_first(collocations_raw.strip())
        else: collocations = ""
        register_raw  = str(card_data.get("register", "") or "").strip().title()
        register      = register_raw if register_raw in REGISTER_VALUES else "Neutral"
        reg_css       = REGISTER_BADGE_CSS.get(register, REGISTER_BADGE_CSS["Neutral"])
        register_html = f'<span class="register-badge" style="{reg_css}">{register}</span>'
        mnemonic_raw  = str(card_data.get("mnemonic", "") or "").strip()
        mnemonic      = cap_first(mnemonic_raw) if mnemonic_raw else ""
        romanization  = normalize_spaces(str(card_data.get("romanization", "") or ""))
        # Z4-E
        translation2_raw = str(card_data.get("translation2", "") or "").strip()
        translation2     = _clean_field(translation2_raw) if translation2_raw else ""

        text_field = (f"{formatted}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
                      if formatted else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>")
        pron_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"

        note = {
            "VocabRaw":        vocab_raw,
            "Text":            text_field,
            "Pronunciation":   pron_field,
            "Definition":      eng_def,
            "Examples":        ex_field,
            "Synonyms":        synonyms,
            "Antonyms":        antonyms,
            "Etymology":       etymology,
            "Collocations":    collocations,
            "Register":        register_html,
            "RegisterLabel":   register,
            "Mnemonic":        mnemonic,
            "Romanization":    romanization,
            "Translation2":    translation2,     # Z4-E
            "TranslationPlain": translation,
            "Tags":            list(tags_lookup.get(vocab_raw, [])),
        }
        q_score = score_card(note)
        note["_quality_score"] = q_score
        if q_score >= 80:   note["Tags"].append("quality_high")
        elif q_score >= 60: note["Tags"].append("quality_medium")
        else:               note["Tags"].append("quality_low")
        processed_notes.append(note)

    st.session_state.processed_cache = {"key": cache_key, "notes": processed_notes, "time": datetime.now()}
    st.caption(f"⏱️ `process_anki_data`: {time.perf_counter() - t0:.3f}s — {len(processed_notes)} notes")
    return processed_notes

# ========================== D17: AUDIO HELPER ==========================
def generate_audio_file(args: tuple):
    vocab, temp_dir = args
    try:
        clean_vocab = _RE_AUDIO_CLEAN.sub('', vocab).strip()
        clean_fname = _RE_CLEAN_FNAME.sub('', clean_vocab) + ".mp3"
        file_path   = os.path.join(temp_dir, clean_fname)
        if clean_vocab:
            gTTS(text=clean_vocab, lang='en', slow=False).save(file_path)
            return vocab, clean_fname, file_path
    except Exception as e: print(f"Audio error for {vocab}: {e}")
    return vocab, None, None

# ========================== Z2-D: SENTENCE AUDIO HELPER ==========================
def generate_sentence_audio_file(args: tuple):
    """Z2-D: Generates audio for the first example sentence."""
    vocab_raw, sentence, temp_dir = args
    if not sentence or not sentence.strip():
        return vocab_raw, None, None
    try:
        clean_fname = "sent_" + _RE_CLEAN_FNAME.sub('', vocab_raw) + ".mp3"
        file_path   = os.path.join(temp_dir, clean_fname)
        clean_sent  = _RE_AUDIO_CLEAN.sub(' ', sentence).strip()
        if clean_sent:
            gTTS(text=clean_sent, lang='en', slow=False).save(file_path)
            return vocab_raw, clean_fname, file_path
    except Exception as e: print(f"Sentence audio error for {vocab_raw}: {e}")
    return vocab_raw, None, None

# ========================== GENANKI LOGIC ==========================
def create_anki_package(
    notes_data:       list,
    deck_name:        str,
    generate_audio:   bool = True,
    deck_id:          int  = 2059400110,
    include_antonyms: bool = True,
    include_reversed: bool = False,
    sentence_audio:   bool = False,   # Z2-D
) -> tuple:
    t0 = time.perf_counter()

    front_html = """<div class="vellum-focus-container front">
<div class="prompt-text">{{cloze:Text}}</div></div>"""

    # Z2-E: build back_html dynamically from user-defined section order
    _section_order = st.session_state.get("back_section_order", list(BACK_SECTIONS_DEFAULT))
    back_html      = build_back_html(_section_order, include_antonyms)

    model_id = st.session_state.get("model_id", 1607392319)
    my_model = genanki.Model(
        model_id, 'Cyberpunk Vocab Model',
        fields=[
            {'name': 'Text'},           {'name': 'Pronunciation'},
            {'name': 'Definition'},     {'name': 'Examples'},
            {'name': 'Collocations'},   {'name': 'Register'},
            {'name': 'Synonyms'},       {'name': 'Antonyms'},
            {'name': 'Etymology'},
            {'name': 'Romanization'},   # Y2-B
            {'name': 'Translation2'},   # Z4-E
            {'name': 'Mnemonic'},
            {'name': 'Audio'},
        ],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=get_active_css(),   # N2-C: theme-aware CSS
        model_type=genanki.Model.CLOZE
    )

    # Y2-A: Reversed card model — BUG-F stable ID
    reversed_model_id = st.session_state.get("reversed_model_id", (model_id + 7919) % (1 << 31))
    rev_front = """<div class="vellum-focus-container front">
<div class="prompt-text" style="color:#ffff66">{{Translation}}</div>
{{#Pronunciation}}<div style="color:#aaffaa;font-size:0.9em;margin-top:8px">{{Pronunciation}}</div>{{/Pronunciation}}
</div>"""
    rev_back  = """<div class="vellum-focus-container back">
<div class="prompt-text" style="color:#ff00ff">{{VocabWord}}</div></div>
<div class="vellum-detail-container">
{{#Definition}}<div class="vellum-section">
<div class="section-header">📜 DEFINITION</div>
<div class="content">{{Definition}}</div></div>{{/Definition}}
{{#Mnemonic}}<div class="vellum-section">
<div class="section-header">💡 MEMORY HOOK</div>
<div class="content">{{Mnemonic}}</div></div>{{/Mnemonic}}
</div>"""
    reversed_model = genanki.Model(
        reversed_model_id, 'Cyberpunk Vocab Reversed',
        fields=[{'name': 'Translation'}, {'name': 'Pronunciation'},
                {'name': 'VocabWord'},   {'name': 'Definition'}, {'name': 'Mnemonic'}],
        templates=[{'name': 'Reversed', 'qfmt': rev_front, 'afmt': rev_back}],
        css=get_active_css(),   # N2-C: theme-aware CSS
    )

    my_deck     = genanki.Deck(deck_id, deck_name)
    media_files = []

    # Y2-E: Deck statistics
    all_fields_check = ['Definition', 'Examples', 'Collocations', 'Synonyms',
                        'Antonyms', 'Mnemonic', 'Romanization', 'Translation2']
    scores      = [n.get('_quality_score', 0) for n in notes_data]
    avg_quality = int(sum(scores) / len(scores)) if scores else 0
    deck_stats  = {
        "total_cards":      len(notes_data),
        "avg_quality":      avg_quality,
        "field_completion": {f: sum(1 for n in notes_data if str(n.get(f, '')).strip())
                             for f in all_fields_check},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            t_audio       = time.perf_counter()
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                for vk, fn, fp in exc.map(generate_audio_file, [(v, temp_dir) for v in unique_vocabs]):
                    if fn:
                        media_files.append(fp)
                        audio_map[vk] = f"[sound:{fn}]"
            st.caption(f"⏱️ Audio: {time.perf_counter() - t_audio:.2f}s for {len(unique_vocabs)} words")

        # Z2-D: Sentence audio
        sent_audio_map = {}
        if generate_audio and sentence_audio:
            t_sent    = time.perf_counter()
            sent_args = []
            for n in notes_data:
                plain_sent = _RE_STRIP_HTML.sub(' ', n.get('Examples', '') or '').strip()
                first_sent = plain_sent.split('  ')[0].strip() if plain_sent else ''
                if first_sent: sent_args.append((n['VocabRaw'], first_sent, temp_dir))
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                for vk, fn, fp in exc.map(generate_sentence_audio_file, sent_args):
                    if fn:
                        media_files.append(fp)
                        sent_audio_map[vk] = f"[sound:{fn}]"
            st.caption(f"⏱️ Sentence audio: {time.perf_counter() - t_sent:.2f}s for {len(sent_audio_map)} sentences")

        exported_hashes = st.session_state.get("exported_hashes", set())
        for note_data in notes_data:
            guid_input = note_data['VocabRaw'] + deck_name
            vocab_hash = str(int(hashlib.sha256(guid_input.encode('utf-8')).hexdigest(), 16) % (10 ** 10))
            exported_hashes.add(hashlib.sha256(note_data['VocabRaw'].encode('utf-8')).hexdigest()[:16])
            my_deck.add_note(genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'],                     note_data['Pronunciation'],
                    note_data['Definition'],               note_data['Examples'],
                    note_data.get('Collocations', ''),     note_data.get('Register', ''),
                    note_data['Synonyms'],                 note_data['Antonyms'],
                    note_data['Etymology'],
                    note_data.get('Romanization', ''),
                    note_data.get('Translation2', ''),     # Z4-E
                    note_data.get('Mnemonic', ''),
                    # Z2-D: word audio + sentence audio
                    audio_map.get(note_data['VocabRaw'], "") + sent_audio_map.get(note_data['VocabRaw'], ""),
                ],
                tags=note_data['Tags'],
                guid=vocab_hash
            ))
            if include_reversed:
                rev_guid = str(int(hashlib.sha256(
                    (note_data['VocabRaw'] + deck_name + "_rev").encode('utf-8')).hexdigest(), 16
                ) % (10 ** 10))
                my_deck.add_note(genanki.Note(
                    model=reversed_model,
                    fields=[note_data.get('TranslationPlain', ''), note_data['Pronunciation'],
                            note_data['VocabRaw'], note_data['Definition'], note_data.get('Mnemonic', '')],
                    guid=rev_guid
                ))

        st.session_state.exported_hashes = exported_hashes
        my_package             = genanki.Package(my_deck)
        my_package.media_files = media_files
        output_path            = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(output_path)
        buffer = io.BytesIO()
        with open(output_path, "rb") as f: buffer.write(f.read())
        buffer.seek(0)

    save_export_history(deck_name=deck_name, card_count=len(notes_data),
                        vocab_list=[n['VocabRaw'] for n in notes_data])
    st.caption(f"⏱️ `create_anki_package` total: {time.perf_counter() - t0:.2f}s")
    return buffer, deck_stats

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str)
        df['phrase'] = df['phrase'].fillna("")
        df['status'] = df['status'].fillna('New') if 'status' in df.columns else 'New'
        df['tags']   = df['tags'].fillna('')      if 'tags'   in df.columns else ''
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404: return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags'])
        st.stop()
    except: st.stop()

def save_to_github(dataframe: pd.DataFrame) -> bool:
    st.session_state.undo_df = st.session_state.vocab_df.copy()   # N4-B
    t0        = time.perf_counter()
    mask      = dataframe['vocab'].astype(str).str.strip().str.len() > 0
    dataframe = dataframe[mask].drop_duplicates(subset=['vocab'], keep='last')
    drop_cols = [c for c in ['Export', '⚠️ Prev. Exported', '_quality_score', 'RegisterLabel']
                 if c in dataframe.columns]
    if drop_cols: dataframe = dataframe.drop(columns=drop_cols)
    csv_data  = dataframe.to_csv(index=False)
    csv_bytes = len(csv_data.encode('utf-8'))
    if csv_bytes > 500_000:   # Y1-D
        st.warning(f"⚠️ vocabulary.csv is **{csv_bytes / 1024:.0f} KB** — approaching GitHub limits.")
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
        _gh_write_tick()   # X1-A
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial commit", csv_data)
            _gh_write_tick()   # X1-A
    load_data.clear()
    st.caption(f"⏱️ GitHub save: {time.perf_counter() - t0:.2f}s ({csv_bytes/1024:.0f} KB)")
    return True

# ========================== SESSION STATE INIT ==========================
_BOOT_T_GH2 = time.perf_counter()   # X1-B: before usage load
if "rpd_count" not in st.session_state or "rpm_timestamps" not in st.session_state:
    _init_rpd, _init_rpm = load_combined_usage()
else:
    _init_rpd = st.session_state.rpd_count
    _init_rpm = st.session_state.rpm_timestamps
_BOOT_T_USAGE_DONE = time.perf_counter()   # X1-B: after usage load

st.session_state.setdefault("gemini_key",              DEFAULT_GEMINI_KEY)
st.session_state.setdefault("vocab_df",                load_data().copy())
st.session_state.setdefault("rpd_count",               _init_rpd)
st.session_state.setdefault("rpm_timestamps",          _init_rpm)
st.session_state.setdefault("deck_id",                 2059400110)
st.session_state.setdefault("bulk_preview_df",         None)
st.session_state.setdefault("apkg_buffer",             None)
st.session_state.setdefault("processed_vocabs",        [])
st.session_state.setdefault("model_id",                1607392319)
st.session_state.setdefault("reversed_model_id",       (1607392319 + 7919) % (1 << 31))  # BUG-F
st.session_state.setdefault("include_antonyms",        True)
st.session_state.setdefault("dry_run",                 False)
st.session_state.setdefault("processed_cache",         {})
_init_word_cache = load_word_cache() if "word_cache" not in st.session_state else st.session_state.word_cache
st.session_state.setdefault("word_cache",              _init_word_cache)
st.session_state.setdefault("input_phrase",            "")
st.session_state.setdefault("input_vocab",             "")
st.session_state.setdefault("_quota_cache_key",        None)
st.session_state.setdefault("_quota_cache",            (20, 0))
st.session_state.setdefault("target_lang",             "Indonesian")
st.session_state.setdefault("target_lang2",            "None (disabled)")   # Z4-E
st.session_state.setdefault("gemini_model_name",       "gemini-2.5-flash-lite")
st.session_state.setdefault("persona",                 "General")
st.session_state.setdefault("difficulty",              "Intermediate")
st.session_state.setdefault("tpm_log",                 [])
st.session_state.setdefault("failed_words",            [])
st.session_state.setdefault("exported_hashes",         set())
st.session_state.setdefault("preview_notes",           [])
st.session_state.setdefault("last_deck_name",          "-English Learning::Vocabulary")
st.session_state.setdefault("last_batch_size",         6)
st.session_state.setdefault("model_id_confirm",        False)
st.session_state.setdefault("undo_df",                 None)
st.session_state.setdefault("use_mnemonic",            False)
st.session_state.setdefault("editing_notes",           None)
st.session_state.setdefault("editing_deck_name",       "")
st.session_state.setdefault("editing_audio",           True)
st.session_state.setdefault("editing_reversed",        False)
st.session_state.setdefault("editing_sentence_audio",  False)   # Z2-D
st.session_state.setdefault("generation_checkpoint",   [])
st.session_state.setdefault("deck_stats",              {})
st.session_state.setdefault("light_mode",              False)
st.session_state.setdefault("session_words_added",     0)
st.session_state.setdefault("session_cards_generated", 0)
st.session_state.setdefault("_quota_reset_warned",     False)   # Z1-B
st.session_state.setdefault("session_api_calls_start", _init_rpd)  # X1-D
st.session_state.setdefault("_boot_profiled",          False)   # X1-B
st.session_state.setdefault("back_section_order",      list(BACK_SECTIONS_DEFAULT))  # Z2-E
st.session_state.setdefault("card_theme",             "🟢 Cyberpunk")  # N2-C

# Y4-D: Light mode CSS injection
if st.session_state.light_mode:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

st.title("📚 My Cloud Vocab")

# Z1-B: Quota reset toast — fires once per session when nearly out
if st.session_state.rpd_count >= 18 and not st.session_state.get("_quota_reset_warned", False):
    st.session_state["_quota_reset_warned"] = True
    st.toast(f"⚠️ Almost out of quota ({st.session_state.rpd_count}/20 used). "
             f"Resets in **{quota_reset_countdown()}**.", icon="🛑")

# X1-B: Cold boot profiler toast — fires once on first render
if not st.session_state.get("_boot_profiled", False):
    st.session_state["_boot_profiled"] = True
    _t_total = time.perf_counter() - _BOOT_T0
    _t_gh    = _BOOT_T_GH_DONE - _BOOT_T_GH_START
    _t_usage = _BOOT_T_USAGE_DONE - _BOOT_T_GH2
    st.toast(f"⚡ Cold boot: **{_t_total:.1f}s** (GH {_t_gh:.1f}s · usage {_t_usage:.1f}s)", icon="⏱️")

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
    st.session_state.editing_notes    = None

def save_single_word_callback():
    v = st.session_state.input_vocab.lower().strip()
    if v:
        p    = normalize_phrase(st.session_state.input_phrase)
        # Z1-D: Duplicate phrase detector
        if p and p not in ("1", "*"):
            _p_lower  = p.strip().lower()
            _df       = st.session_state.vocab_df
            _dup_mask = (_df['phrase'].astype(str).str.strip().str.lower() == _p_lower) & \
                        (_df['vocab'].astype(str).str.strip().str.lower() != v)
            if _dup_mask.any():
                _dup_word = _df.loc[_dup_mask, 'vocab'].iloc[0]
                st.warning(f"⚠️ This phrase is already used for **'{_dup_word}'**. "
                           f"Consider using a different example sentence.")
        mask = st.session_state.vocab_df['vocab'] == v
        if not st.session_state.vocab_df.empty and mask.any():
            st.session_state.vocab_df.loc[mask, ['phrase', 'status']] = [p, 'New']
        else:
            new_row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "tags": ""}])
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
        save_to_github(st.session_state.vocab_df)
        st.session_state.input_phrase      = ""
        st.session_state.input_vocab       = ""
        st.session_state.session_words_added += 1
        st.toast(f"✅ Saved '{v}'!", icon="🚀")
    else:
        st.error("⚠️ Enter a vocabulary word.")

def quick_add_callback():
    v = st.session_state.get("quick_add_vocab", "").lower().strip()
    if v:
        mask = st.session_state.vocab_df['vocab'] == v
        if not st.session_state.vocab_df.empty and mask.any():
            st.toast(f"⚠️ '{v}' already exists.", icon="⚠️"); return
        new_row = pd.DataFrame([{"vocab": v, "phrase": "", "status": "New", "tags": ""}])
        st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
        save_to_github(st.session_state.vocab_df)
        st.session_state.quick_add_vocab     = ""
        st.session_state.session_words_added += 1
        st.toast(f"✅ Quick-added '{v}'!", icon="⚡")
        st.session_state.pop("_edit_buf_key", None)
        st.session_state.pop("_edit_buffer", None)
    else:
        st.error("⚠️ Enter a word.")

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("⚙️ Settings")

    total_words = len(st.session_state.vocab_df)
    new_words   = len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'])
    col1, col2  = st.columns(2)
    col1.metric("📖 Total", total_words)
    col2.metric("✨ New",   new_words)
    st.metric("🤖 Daily AI Usage", f"{st.session_state.rpd_count}/20 Requests")

    # X1-A + X1-D + Y4-F: combined session counters
    _api_used_session = max(0, st.session_state.rpd_count - st.session_state.get("session_api_calls_start", 0))
    _gh_writes        = gh_write_count()
    st.caption(
        f"📝 Added: **{st.session_state.session_words_added}** · "
        f"🃏 Cards: **{st.session_state.session_cards_generated}** · "
        f"🤖 API: **{_api_used_session}** · "
        f"📡 GH: **{_gh_writes}**"
        + (" ⚠️" if _gh_writes >= GH_WRITE_WARN_THRESHOLD else "")
    )
    if _gh_writes >= GH_WRITE_WARN_THRESHOLD:
        st.caption(f"⚠️ {_gh_writes} GH writes this session — approaching hourly limit (100).")

    rpm_live = len([ts for ts in st.session_state.rpm_timestamps
                    if (datetime.now() - ts).total_seconds() < 60])
    st.progress(rpm_live / 5,                    text=f"RPM Live: {rpm_live}/5 (last 60s)")
    st.progress(st.session_state.rpd_count / 20, text=f"RPD: {st.session_state.rpd_count}/20")
    tpm_estimate = get_rolling_tpm()
    tpm_frac     = min(tpm_estimate / 1_000_000, 1.0)
    tpm_icon     = "🟢" if tpm_frac < 0.60 else ("🟡" if tpm_frac < 0.85 else "🔴")
    st.progress(tpm_frac, text=f"TPM est: {tpm_icon} {tpm_estimate:,} / 1,000,000 (last 60s)")
    st.caption(f"⏰ Quota resets in **{quota_reset_countdown()}** (UTC midnight)")

    st.divider()

    st.selectbox("🎯 Definition Language",
                 ["Indonesian", "Spanish", "French", "German", "Japanese",
                  "Chinese (Mandarin)", "English (Simple)"],
                 index=0, key="target_lang")
    st.selectbox("🤖 AI Model",
                 ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0, key="gemini_model_name")
    st.selectbox("🧠 Subject Persona", list(PERSONAS.keys()), index=0, key="persona",
                 help="Shapes definition style and example domain.")
    st.radio("📊 Difficulty Level", list(DIFFICULTY_SUFFIX.keys()),
             index=1, horizontal=True, key="difficulty", help="Appends a CEFR-level Rule 10.")
    if st.session_state.difficulty != "Intermediate":
        st.caption(f"📌 Rule 10: _{DIFFICULTY_SUFFIX[st.session_state.difficulty]}_")
    st.checkbox("💡 Generate Memory Hooks", key="use_mnemonic",
                help="Adds a mnemonic hook per card (Rule 11, ~25 extra tokens/word).")
    if st.session_state.use_mnemonic:
        st.caption("📌 Rule 11 active.")

    # Z4-E: Second language
    st.selectbox("🌐 Second Language (optional)", LANG2_OPTIONS, index=0, key="target_lang2",
                 help="Adds Translation2 field for a second target language on every card.")
    if st.session_state.target_lang2 != "None (disabled)":
        st.caption(f"📌 Rule 13 active: Translation2 → {st.session_state.target_lang2}")

    # N2-C: Card theme selector
    st.divider()
    st.selectbox(
        "🎨 Card Theme",
        options=list(CARD_THEMES.keys()),
        index=list(CARD_THEMES.keys()).index(st.session_state.get("card_theme", "🟢 Cyberpunk")),
        key="card_theme",
        help="Changes the visual style of your Anki cards."
    )
    _active_theme = CARD_THEMES[st.session_state.card_theme]
    st.caption(f"_{_active_theme['description']}_")

    # Z2-E: Card back section reorder
    with st.expander("🃏 Card Back Section Order", expanded=False):
        st.caption("Select and reorder which sections appear on the card back.")
        _ordered = st.multiselect(
            "Visible sections (top = first on card)",
            options=BACK_SECTIONS_DEFAULT,
            default=st.session_state.back_section_order,
            key="_back_section_ms",
            help="Remove a section to hide it."
        )
        if _ordered != st.session_state.back_section_order:
            st.session_state.back_section_order = _ordered

    # Y4-D: Theme toggle
    st.toggle("☀️ Light Mode", key="light_mode",
              help="Injects CSS overrides for a light background theme.")

    st.divider()

    has_exported = (len(st.session_state.processed_vocabs) > 0
                    or len(st.session_state.exported_hashes) > 0)
    if st.button("🔄 Regenerate Note Type Model ID"):
        if has_exported and not st.session_state.model_id_confirm:
            st.session_state.model_id_confirm = True
        else:
            new_mid = random.randrange(1 << 30, 1 << 31)
            st.session_state.model_id          = new_mid
            st.session_state.reversed_model_id = (new_mid + 7919) % (1 << 31)
            st.session_state.model_id_confirm  = False
            st.success(f"New Model ID: {st.session_state.model_id}")
    if st.session_state.model_id_confirm:
        st.warning("⚠️ Changing Model ID may orphan existing Anki cards. **Click again to confirm.**")
    st.caption(f"Current Model ID: {st.session_state.model_id}")

    if st.button("🗑️ Clear Word Cache"):
        st.session_state.word_cache            = {}
        st.session_state.processed_cache       = {}
        st.session_state.generation_checkpoint = []
        st.session_state["_gap_cache_key"]     = None
        load_word_cache.clear()
        save_word_cache({})
        st.toast("🗑️ Cache cleared.")

    if not st.session_state.vocab_df.empty:
        st.download_button("💾 Backup Database (CSV)",
                           st.session_state.vocab_df.to_csv(index=False).encode('utf-8'),
                           f"vocab_backup_{date.today()}.csv", "text/csv")

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

# ──────────────────────────── TAB 1 ────────────────────────────
with tab1:
    # Y3-A: Word of the Day — BUG-A: local Random instance
    done_words  = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'Done']['vocab'].tolist()
    cached_done = [w for w in done_words if w in st.session_state.word_cache]
    if len(cached_done) >= 5:
        _local_rng = random.Random(date.today().toordinal())
        wotd       = _local_rng.choice(cached_done)
        wotd_data  = st.session_state.word_cache.get(wotd, {})
        with st.expander(f"⭐ Word of the Day: **{wotd.title()}**", expanded=False):
            if wotd_data.get("pronunciation_ipa"): st.caption(f"🗣️ {wotd_data['pronunciation_ipa']}")
            if wotd_data.get("definition_english"): st.info(f"📜 {wotd_data['definition_english']}")
            if wotd_data.get("mnemonic"):           st.caption(f"💡 {wotd_data['mnemonic']}")

    # Y3-B: Vocab gap detector — BUG-D: cached
    _gap_cache_key = len(st.session_state.word_cache)
    if st.session_state.get("_gap_cache_key") != _gap_cache_key:
        st.session_state["_gap_cache_key"] = _gap_cache_key
        st.session_state["_gap_cache"]     = detect_vocab_gaps(st.session_state.word_cache)
    gaps = st.session_state["_gap_cache"]
    if gaps:
        with st.expander(f"🔍 Vocab Gap Alert ({len(gaps)} cluster{'s' if len(gaps)>1 else ''})", expanded=False):
            for g in gaps:
                st.warning(f"You have {len(g['words'])} words with similar meaning "
                           f"({', '.join(f'*{s}*' for s in g['shared_synonyms'])}): "
                           f"{', '.join(f'**{w}**' for w in g['words'])}. "
                           f"Consider adding contrasting vocabulary.")

    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")

    if add_mode == "Single":
        p_raw      = st.text_input("🔤 Phrase", placeholder="Paste your sentence here...", key="input_phrase")
        v_selected = ""
        if p_raw and p_raw not in ["1", "*"]:
            clean_text   = _RE_CLEAN_TEXT.sub('', p_raw)
            unique_words = list(dict.fromkeys([w.lower() for w in clean_text.split() if w.strip()]))
            if unique_words:
                st.caption("Click words below to extract them as vocabulary:")
                try:
                    selected_pills = st.pills("Select Vocab", unique_words,
                                              selection_mode="multi", label_visibility="collapsed")
                    v_selected = " ".join(selected_pills) if selected_pills else ""
                except:
                    selected_words, cols = [], st.columns(6)
                    for i, w in enumerate(unique_words):
                        if cols[i % 6].checkbox(w, key=f"chk_{w}"): selected_words.append(w)
                    v_selected = " ".join(selected_words)
        if v_selected and v_selected != st.session_state.input_vocab:
            st.session_state.input_vocab = v_selected
        st.text_input("📝 Vocab", placeholder="e.g. serendipity", key="input_vocab")
        v_check = st.session_state.input_vocab.lower().strip()
        if (v_check and not st.session_state.vocab_df.empty
                and (st.session_state.vocab_df['vocab'] == v_check).any()):
            st.warning(f"⚠️ '{v_check}' already exists. Saving will overwrite.")
        st.button("💾 Save to Cloud", type="primary", use_container_width=True,
                  on_click=save_single_word_callback)
    else:
        # Y4-B: CSV file upload
        uploaded_csv = st.file_uploader("📂 Upload CSV (vocab, phrase, tags columns)",
                                        type=["csv"], key="csv_upload")
        if uploaded_csv is not None:
            try:
                up_df   = pd.read_csv(uploaded_csv, dtype=str).fillna("")
                col_map = {}
                for col in up_df.columns:
                    cl = col.strip().lower()
                    if cl in ("vocab", "word", "term"):              col_map["vocab"]  = col
                    elif cl in ("phrase", "sentence", "example"):    col_map["phrase"] = col
                    elif cl in ("tags", "tag"):                      col_map["tags"]   = col
                if "vocab" not in col_map:
                    st.error("❌ No 'vocab' column found. Columns: " + ", ".join(up_df.columns))
                else:
                    csv_rows = []
                    for _, row in up_df.iterrows():
                        bv = str(row.get(col_map["vocab"], "")).strip().lower()
                        bp = normalize_phrase(str(row.get(col_map.get("phrase", ""), "")).strip())
                        bt = str(row.get(col_map.get("tags", ""), "")).strip()
                        if bv: csv_rows.append({"vocab": bv, "phrase": bp, "status": "New", "tags": bt})
                    if csv_rows:
                        st.session_state.bulk_preview_df = pd.DataFrame(csv_rows)
                        st.success(f"✅ Parsed {len(csv_rows)} words from uploaded CSV.")
            except Exception as e:
                st.error(f"❌ CSV parse error: {e}")
        st.divider()
        bulk_text = st.text_area("Or paste list (word, phrase)", height=120, key="bulk_input")
        if st.button("Preview Bulk Import"):
            lines, new_rows = [l.strip() for l in bulk_text.split('\n') if l.strip()], []
            for line in lines:
                parts = line.split(',', 1)
                bv    = parts[0].strip().lower()
                bp    = normalize_phrase(parts[1].strip() if len(parts) > 1 else "")
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "status": "New", "tags": ""})
            if new_rows: st.session_state.bulk_preview_df = pd.DataFrame(new_rows)

        if st.session_state.bulk_preview_df is not None:
            st.write("### Preview:")
            st.dataframe(st.session_state.bulk_preview_df, hide_index=True)
            empty_phrase_mask = st.session_state.bulk_preview_df['phrase'].astype(str).str.strip() == ''
            empty_count       = int(empty_phrase_mask.sum())
            if empty_count > 0 and st.session_state.rpd_count < 19:
                if st.button(f"✨ Auto-fill {empty_count} empty phrase(s) — costs 1 API request"):
                    empty_vocabs = st.session_state.bulk_preview_df.loc[empty_phrase_mask, 'vocab'].tolist()
                    with st.spinner("🤖 Generating example sentences..."):
                        phrase_map = enrich_empty_phrases(empty_vocabs)
                    if phrase_map:
                        # BUG-E: vectorized assignment
                        _df        = st.session_state.bulk_preview_df
                        _vk_series = _df['vocab'].astype(str).str.strip().str.lower()
                        _ph_series = _df['phrase'].astype(str).str.strip()
                        _mask      = _vk_series.isin(phrase_map) & (_ph_series == '')
                        _df.loc[_mask, 'phrase'] = _vk_series[_mask].map(phrase_map)
                        st.session_state.bulk_preview_df = _df
                        st.success(f"✅ Filled {int(_mask.sum())} phrase(s).")
                        st.rerun(scope="app")
            if st.button("💾 Confirm & Process Bulk", type="primary"):
                added = len(st.session_state.bulk_preview_df)
                st.session_state.vocab_df = pd.concat(
                    [st.session_state.vocab_df, st.session_state.bulk_preview_df]
                ).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(st.session_state.vocab_df)
                st.success(f"✅ Added {added} words!")
                st.session_state.session_words_added += added
                st.session_state.bulk_preview_df = None
                st.rerun(scope="app")

# ──────────────────────────── TAB 2 ────────────────────────────
with tab2:
    @st.fragment
    def render_tab2():
        if st.session_state.vocab_df.empty:
            st.info("Add words first!"); return
        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        with st.expander("⚡ Quick-Add Word", expanded=False):
            st.text_input("Vocab word", placeholder="e.g. tenacious", key="quick_add_vocab")
            st.button("➕ Add", on_click=quick_add_callback, use_container_width=True)
        if st.session_state.undo_df is not None:
            if st.button("↩️ Undo Last Save", use_container_width=True):
                st.session_state.vocab_df = st.session_state.undo_df.copy()
                st.session_state.undo_df  = None
                save_to_github(st.session_state.vocab_df)
                st.toast("↩️ Undo applied.", icon="↩️")
                st.rerun(scope="app")
        search     = st.text_input("🔎 Search...", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search:
            display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        page_size = 50
        page      = st.number_input("Page", min_value=1, value=1, step=1)
        start     = (page - 1) * page_size
        paginated = display_df.iloc[start:start + page_size]
        _buf_key  = f"_edit_buf_{page}_{search}"
        if st.session_state.get("_edit_buf_key") != _buf_key:
            st.session_state["_edit_buf_key"] = _buf_key
            st.session_state["_edit_buffer"]  = paginated.copy()
        edited = st.data_editor(
            st.session_state["_edit_buffer"],
            num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={"status": st.column_config.SelectboxColumn(
                "Status", options=["New", "Done"], required=True)}
        )
        st.session_state["_edit_buffer"] = edited
        col_save, col_quality = st.columns(2)
        if col_save.button("💾 Save Changes", type="primary", use_container_width=True):
            full_df      = st.session_state.vocab_df.copy()
            existing_idx = [i for i in edited.index if i in full_df.index]
            if existing_idx:
                full_df.loc[existing_idx, edited.columns.tolist()] = edited.loc[existing_idx].values
            new_mask = ~edited.index.isin(full_df.index)
            if new_mask.any():
                full_df = pd.concat([full_df, edited[new_mask]], ignore_index=True)
            deleted_idx = [i for i in paginated.index if i not in edited.index]
            if deleted_idx:
                full_df = full_df.drop(index=deleted_idx).reset_index(drop=True)
            st.session_state.vocab_df = full_df
            st.session_state.pop("_edit_buf_key", None)
            st.session_state.pop("_edit_buffer", None)
            save_to_github(st.session_state.vocab_df)
            st.toast("✅ Cloud updated!")
            st.rerun(scope="app")
        if col_quality.button("📊 Data Quality", use_container_width=True):
            df, total = st.session_state.vocab_df, len(st.session_state.vocab_df)
            if total == 0:
                st.info("No data to analyse.")
            else:
                with_phrase = (df['phrase'].astype(str).str.strip() != '').sum()
                with_tags   = (df['tags'].astype(str).str.strip() != '').sum() if 'tags' in df.columns else 0
                dups        = df['vocab'].duplicated().sum()
                short_vocab = (df['vocab'].astype(str).str.strip().str.len() <= 2).sum()
                pct = lambda n: f"{n / total * 100:.0f}%"
                st.dataframe(pd.DataFrame({
                    "Metric": ["Total","With phrases","With tags","Duplicates","Short (≤2)","New","Done"],
                    "Count":  [total, with_phrase, with_tags, dups, short_vocab,
                               (df['status']=='New').sum(), (df['status']=='Done').sum()],
                    "%":      ["100%", pct(with_phrase), pct(with_tags), pct(dups), pct(short_vocab),
                               pct((df['status']=='New').sum()), pct((df['status']=='Done').sum())],
                }), hide_index=True, use_container_width=True)
                if dups > 0:        st.warning(f"⚠️ {dups} duplicates detected.")
                if short_vocab > 0: st.warning(f"⚠️ {short_vocab} entries with vocab ≤2 chars.")
                if total > 0 and with_phrase / total < 0.5:
                    st.info(f"💡 Only {pct(with_phrase)} of words have phrases.")
        st.divider()
        with st.expander("🔄 Bulk Status Reset", expanded=False):
            confirmed = st.checkbox("✅ I confirm this will change ALL word statuses")
            col_new, col_done = st.columns(2)
            if col_new.button("🔄 Reset ALL to New", disabled=not confirmed, use_container_width=True):
                st.session_state.vocab_df['status'] = 'New'
                save_to_github(st.session_state.vocab_df)
                st.session_state.pop("_edit_buf_key", None)
                st.toast("🔄 All words reset to New.")
                st.rerun(scope="app")
            if col_done.button("✅ Mark ALL Done", disabled=not confirmed, use_container_width=True):
                st.session_state.vocab_df['status'] = 'Done'
                save_to_github(st.session_state.vocab_df)
                st.session_state.pop("_edit_buf_key", None)
                st.toast("✅ All words marked Done.")
                st.rerun(scope="app")
    render_tab2()

# ──────────────────────────── TAB 3 ────────────────────────────
with tab3:
    @st.fragment
    def render_tab3():

        # Y1-A: Resume checkpoint banner
        checkpoint = st.session_state.get("generation_checkpoint", [])
        if checkpoint and st.session_state.editing_notes is None and st.session_state.apkg_buffer is None:
            with st.expander(f"⏸️ Resume partial generation? ({len(checkpoint)} cards saved)", expanded=True):
                st.info("A previous generation was interrupted. You can resume from the saved checkpoint.")
                col_resume, col_discard = st.columns(2)
                if col_resume.button("▶️ Resume (use saved cards)", type="primary"):
                    st.session_state.editing_notes     = checkpoint
                    st.session_state.editing_deck_name = st.session_state.last_deck_name
                    st.session_state.editing_audio     = True
                    st.rerun(scope="app")
                if col_discard.button("🗑️ Discard checkpoint"):
                    st.session_state.generation_checkpoint = []
                    st.rerun(scope="app")

        # ── PHASE 2: Card editor ──────────────────────────────────────────
        if st.session_state.editing_notes is not None:
            st.subheader("✏️ Edit Generated Cards")
            st.caption("Review and fix AI output. All changes reflected in the final .apkg.")
            EDITABLE_COLS   = ["VocabRaw", "Definition", "Collocations",
                               "RegisterLabel", "Synonyms", "Antonyms", "Mnemonic"]
            notes_df = pd.DataFrame([
                {**{col: n.get(col, "") for col in EDITABLE_COLS},
                 "Freq": word_frequency_label(n.get("VocabRaw", "")),
                 "Q":    f"{quality_badge(n.get('_quality_score', 0))} {n.get('_quality_score', 0)}"}
                for n in st.session_state.editing_notes
            ])
            edited_notes_df = st.data_editor(
                notes_df, num_rows="fixed", use_container_width=True, hide_index=True,
                column_config={
                    "VocabRaw":      st.column_config.TextColumn("Vocab",        disabled=True),
                    "Freq":          st.column_config.TextColumn("Freq",         disabled=True, width="small"),
                    "Q":             st.column_config.TextColumn("Quality",      disabled=True, width="small"),
                    "Definition":    st.column_config.TextColumn("Definition",   width="large"),
                    "Collocations":  st.column_config.TextColumn("Collocations", width="medium"),
                    "RegisterLabel": st.column_config.SelectboxColumn("Register", options=REGISTER_VALUES, required=True),
                    "Synonyms":      st.column_config.TextColumn("Synonyms",     width="medium"),
                    "Antonyms":      st.column_config.TextColumn("Antonyms",     width="medium"),
                    "Mnemonic":      st.column_config.TextColumn("Memory Hook",  width="large"),
                }
            )
            scores = [n.get("_quality_score", 0) for n in st.session_state.editing_notes]
            avg_q  = int(sum(scores) / len(scores)) if scores else 0
            low_q  = sum(1 for s in scores if s < QUALITY_WARN_THRESHOLD)
            st.info(f"📊 **{len(scores)} cards** · Avg quality: {quality_badge(avg_q)} **{avg_q}/100**"
                    + (f" · ⚠️ {low_q} card(s) below {QUALITY_WARN_THRESHOLD}" if low_q else ""))

            # Z3-B: Antonym advisory
            _antonym_gaps = [n['VocabRaw'] for n in st.session_state.editing_notes
                             if (len([s for s in n.get('Synonyms', '').split(',') if s.strip()]) >= 3
                                 and not n.get('Antonyms', '').strip())]
            if _antonym_gaps:
                st.warning(f"⚠️ **{len(_antonym_gaps)} card(s)** have ≥3 synonyms but no antonyms: "
                           f"`{', '.join(_antonym_gaps[:5])}{'...' if len(_antonym_gaps) > 5 else ''}`. "
                           f"Consider adding antonyms above.")

            # Y4-G: Export card data as CSV
            csv_export_cols = ["VocabRaw", "Definition", "Collocations", "RegisterLabel",
                               "Synonyms", "Antonyms", "Mnemonic", "Etymology"]
            st.download_button(
                "💾 Export card data as CSV",
                pd.DataFrame([{c: n.get(c, '') for c in csv_export_cols}
                               for n in st.session_state.editing_notes]).to_csv(index=False).encode('utf-8'),
                f"card_data_{date.today()}.csv", "text/csv", use_container_width=True
            )

            col_pack, col_cancel = st.columns(2)
            if col_pack.button("📦 Pack & Download .apkg", type="primary", use_container_width=True):
                updated_notes = []
                for i, note in enumerate(st.session_state.editing_notes):
                    if i < len(edited_notes_df):
                        row  = edited_notes_df.iloc[i]
                        note = dict(note)
                        note["Definition"]    = str(row.get("Definition",    note["Definition"]))
                        note["Collocations"]  = str(row.get("Collocations",  note["Collocations"]))
                        note["Synonyms"]      = str(row.get("Synonyms",      note["Synonyms"]))
                        note["Antonyms"]      = str(row.get("Antonyms",      note["Antonyms"]))
                        note["Mnemonic"]      = str(row.get("Mnemonic",      note.get("Mnemonic", "")))
                        new_reg               = str(row.get("RegisterLabel", note.get("RegisterLabel", "Neutral")))
                        new_reg               = new_reg if new_reg in REGISTER_VALUES else "Neutral"
                        reg_css               = REGISTER_BADGE_CSS.get(new_reg, REGISTER_BADGE_CSS["Neutral"])
                        note["RegisterLabel"] = new_reg
                        note["Register"]      = f'<span class="register-badge" style="{reg_css}">{new_reg}</span>'
                        note["_quality_score"]= score_card(note)
                    updated_notes.append(note)
                with st.spinner("🎵 Generating audio & packing .apkg..."):
                    buffer, deck_stats = create_anki_package(
                        updated_notes,
                        st.session_state.editing_deck_name,
                        generate_audio=st.session_state.editing_audio,
                        deck_id=st.session_state.deck_id,
                        include_antonyms=st.session_state.include_antonyms,
                        include_reversed=st.session_state.editing_reversed,
                        sentence_audio=st.session_state.get("editing_sentence_audio", False),   # Z2-D
                    )
                st.session_state.apkg_buffer             = buffer.getvalue()
                st.session_state.processed_vocabs        = [n['VocabRaw'] for n in updated_notes]
                st.session_state.preview_notes           = updated_notes[:3]
                st.session_state.editing_notes           = None
                st.session_state.deck_stats              = deck_stats
                st.session_state.generation_checkpoint   = []
                st.session_state.session_cards_generated += len(updated_notes)
                st.rerun(scope="app")
            if col_cancel.button("❌ Discard & Start Over", use_container_width=True):
                st.session_state.editing_notes = None
                st.rerun(scope="app")
            return

        # ── PHASE 3: Download ──────────────────────────────────────────────
        if st.session_state.apkg_buffer is not None:
            st.success("✅ Deck packed! Download below.")
            ds = st.session_state.get("deck_stats", {})
            if ds:
                st.subheader("📊 Deck Statistics")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Total Cards",    ds.get("total_cards", 0))
                sc2.metric("Avg Quality",    f"{ds.get('avg_quality', 0)}/100")
                fc = ds.get("field_completion", {})
                tc = ds.get("total_cards", 1)
                sc3.metric("With Examples",  f"{fc.get('Examples', 0)}/{tc}")
                sc4.metric("With Mnemonics", f"{fc.get('Mnemonic', 0)}/{tc}")
            if st.session_state.get("preview_notes"):
                show_styled = st.toggle("🃏 Show Anki-style render", value=False)
                with st.expander("👁️ Card Preview (first 3 cards)", expanded=True):
                    for i, note in enumerate(st.session_state.preview_notes, 1):
                        q_score = note.get("_quality_score", 0)
                        st.markdown(f"**Card {i} — FRONT** &nbsp; {quality_badge(q_score)} Quality: **{q_score}/100**")
                        if q_score < QUALITY_WARN_THRESHOLD:
                            st.caption(f"⚠️ Low quality score ({q_score}/100).")
                        front_preview = re.sub(r'\{\{c\d+::(.*?)\}\}', r'[___]', note['Text'])
                        plain_front   = re.sub(r'<[^>]+>', ' ', front_preview).strip()
                        if show_styled:
                            # N2-C: use active theme CSS for preview
                            _preview_css    = get_active_css()
                            _preview_accent = CARD_THEMES[st.session_state.get("card_theme", "🟢 Cyberpunk")]["accent"]
                            _preview_fg     = CARD_THEMES[st.session_state.get("card_theme", "🟢 Cyberpunk")]["front_color"]
                            st.markdown(
                                f"<style>{_preview_css}</style>"
                                f"<div class='card vellum-focus-container'>"
                                f"<div class='prompt-text' style='color:{_preview_fg}'>{front_preview}</div></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(f"<div style='background:#1a1a1a; border:1px solid #00ff41; "
                                        f"padding:10px 14px; border-radius:4px; font-family:monospace; "
                                        f"color:#aaffaa; line-height:1.6'>{plain_front}</div>",
                                        unsafe_allow_html=True)
                        st.markdown(f"**Card {i} — BACK**")
                        back_items = []
                        if note.get("Pronunciation"):  back_items.append(f"🗣️ {note['Pronunciation']}")
                        if note.get("Definition"):     back_items.append(f"📜 {note['Definition']}")
                        if note.get("Romanization"):   back_items.append(f"🈳 {note['Romanization']}")
                        if note.get("Translation2"):   back_items.append(f"🌐 {note['Translation2']}")
                        if note.get("Examples"):
                            plain_ex = re.sub(r'<[^>]+>', ' ', note['Examples']).strip()
                            back_items.append(f"🖋️ {plain_ex}")
                        if note.get("Collocations"):   back_items.append(f"🔗 {note['Collocations']}")
                        if note.get("RegisterLabel"):  back_items.append(f"🏷 {note['RegisterLabel']}")
                        if note.get("Synonyms"):       back_items.append(f"➕ {note['Synonyms']}")
                        if note.get("Mnemonic"):       back_items.append(f"💡 {note['Mnemonic']}")
                        if note.get("Tags"):           back_items.append(f"🔖 {', '.join(note['Tags'])}")
                        st.markdown("<div style='background:#1a1a1a; border:1px solid #00ffff; "
                                    "padding:10px 14px; border-radius:4px; font-family:monospace; "
                                    f"color:#aaffaa; line-height:1.8'>{'<br>'.join(back_items)}</div>",
                                    unsafe_allow_html=True)
                        if i < len(st.session_state.preview_notes): st.divider()
            st.download_button("📥 Download .apkg", data=st.session_state.apkg_buffer,
                               file_name=f"AnkiDeck_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg",
                               mime="application/octet-stream", use_container_width=True,
                               on_click=mark_as_done_callback)
            if st.button("❌ Cancel / Clear"):
                st.session_state.apkg_buffer      = None
                st.session_state.processed_vocabs = []
                st.session_state.preview_notes    = []
                st.session_state.deck_stats       = {}
                st.rerun(scope="app")
            return

        # ── PHASE 1: Generation UI ────────────────────────────────────────
        if st.session_state.vocab_df.empty:
            st.info("Add words first!"); return
        subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
        if subset.empty:
            st.warning("⚠️ No 'New' words to export! All words are marked 'Done'."); return

        # Z3-D: POS filter
        _pos_in_cache = sorted({
            str(st.session_state.word_cache.get(v, {}).get('part_of_speech', '') or '').title()
            for v in subset['vocab'].str.strip().str.lower()
            if str(st.session_state.word_cache.get(v, {}).get('part_of_speech', '')).strip()
        })
        if len(_pos_in_cache) >= 2:
            _pos_filter = st.multiselect("🔠 Filter by Part of Speech (optional)",
                                         options=_pos_in_cache, default=[],
                                         help="Leave blank to include all.")
            if _pos_filter:
                _matching = {v for v in subset['vocab'].str.strip().str.lower()
                             if str(st.session_state.word_cache.get(v, {}).get('part_of_speech', '') or '').title()
                             in _pos_filter}
                subset = subset[subset['vocab'].str.strip().str.lower().isin(_matching)]
                if subset.empty:
                    st.warning("⚠️ No New words match the selected POS filter."); return
                st.caption(f"🔠 Showing **{len(subset)}** word(s) matching: {', '.join(_pos_filter)}")

        # T1-B: Failed words retry panel
        if st.session_state.failed_words:
            with st.expander(f"⚠️ {len(st.session_state.failed_words)} word(s) failed — click to retry",
                             expanded=True):
                st.dataframe(pd.DataFrame({"Queued for Retry": st.session_state.failed_words}),
                             hide_index=True)
                retry_delay = st.select_slider("⏳ Wait before retry", options=[0, 30, 60, 120],
                                               value=30, format_func=lambda x: "Immediate" if x == 0 else f"{x}s delay")
                col_retry, col_dismiss = st.columns(2)
                if col_retry.button("🔁 Retry Failed Words", type="primary"):
                    if retry_delay > 0:
                        _slot = st.empty()
                        for r in range(retry_delay, 0, -1):
                            _slot.warning(f"⏳ Waiting {r}s before retry...")
                            time.sleep(1)
                        _slot.empty()
                    retry_df = pd.DataFrame({
                        "vocab":  st.session_state.failed_words,
                        "phrase": [""] * len(st.session_state.failed_words),
                        "status": ["New"] * len(st.session_state.failed_words),
                        "tags":   [""] * len(st.session_state.failed_words),
                    })
                    st.session_state.failed_words = []
                    retry_notes = process_anki_data(retry_df, batch_size=st.session_state.last_batch_size,
                                                    dry_run=st.session_state.dry_run)
                    if retry_notes:
                        st.session_state.editing_notes     = retry_notes
                        st.session_state.editing_deck_name = st.session_state.last_deck_name
                        st.session_state.editing_audio     = True
                        st.rerun(scope="app")
                    else:
                        st.error("❌ Retry failed. Check your API quota.")
                if col_dismiss.button("🗑️ Dismiss"):
                    st.session_state.failed_words = []
                    st.rerun(scope="app")

        st.subheader("📇 Generate Cyberpunk Anki Deck")

        # T2-B: Deck hierarchy
        deck_col1, deck_col2 = st.columns([3, 1])
        deck_name_raw   = deck_col1.text_input("📦 Deck Name (use :: for sub-decks)",
                                               value=st.session_state.last_deck_name)
        deck_parts_raw  = [p.strip() for p in deck_name_raw.split("::") if p.strip()]
        deck_parts      = [_RE_DECK_ILLEGAL.sub("", p) for p in deck_parts_raw]
        deck_name_input = "::".join(deck_parts) if deck_parts else "Vocabulary"
        if deck_name_raw: st.session_state.last_deck_name = deck_name_input
        if _RE_DECK_ILLEGAL.search(deck_name_raw.replace("::", "")):
            st.warning("⚠️ Illegal characters removed from deck name.")
        if len(deck_parts) > 1:
            st.caption("📂 Hierarchy: " + " → ".join(deck_parts))
        if deck_col2.button("🎲 New Deck ID"):
            st.session_state.deck_id = random.randrange(1 << 30, 1 << 31)
        deck_col2.caption(f"ID: {st.session_state.deck_id}")

        requests_left = max(0, 20 - st.session_state.rpd_count)

        # Z4-D: Smart batch size recommendation
        _n_words     = len(subset)
        _r_left      = max(1, requests_left)
        _recommended = min(15, max(1, math.ceil(_n_words / max(1, int(_r_left * 0.6)))))
        raw_batch    = st.slider("⚡ Batch Size (Words per Request)", 1, 15, _recommended,
                                 help=f"Recommended **{_recommended}** — uses ~{math.ceil(_n_words / _recommended)} "
                                      f"of your {requests_left} remaining requests (60% quota rule).")
        if raw_batch == _recommended:
            st.caption(f"✅ Using recommended batch size **{_recommended}** "
                       f"({math.ceil(_n_words / _recommended)} request(s) needed).")
        else:
            st.caption(f"⚙️ Custom batch size: **{raw_batch}** (recommended was {_recommended}).")
        max_safe  = max(1, math.ceil(_n_words / max(1, requests_left))) if requests_left > 0 else 1
        batch_size = min(raw_batch, max_safe)
        st.session_state.last_batch_size = batch_size
        if batch_size != raw_batch:
            st.caption(f"⚠️ Capped to **{batch_size}** by quota limit.")

        include_audio          = st.checkbox("🔊 Generate Audio Files",                    value=True)
        include_sentence_audio = st.checkbox("🔊 Also audio for example sentences (Z2-D)", value=False,   # Z2-D
                                             help="Generates a second MP3 per card for the first example sentence.")
        include_reversed       = st.checkbox("🔄 Include Reversed Cards (Translation→Word)", value=False)
        st.session_state.include_antonyms = st.checkbox("➖ Include Antonyms in Card Back",  value=st.session_state.include_antonyms)
        st.session_state.dry_run          = st.checkbox("🔬 Dry Run Mode (simulate, no quota)", value=st.session_state.dry_run)

        def _is_dup(vocab_raw: str) -> bool:
            return hashlib.sha256(str(vocab_raw).lower().encode('utf-8')).hexdigest()[:16] \
                in st.session_state.exported_hashes

        subset_display                       = subset.copy()
        subset_display['Export']             = True
        subset_display['⚠️ Prev. Exported'] = subset_display['vocab'].apply(_is_dup)
        st.write("**Select words to export:**")
        edited_export = st.data_editor(
            subset_display,
            column_config={
                "Export": st.column_config.CheckboxColumn("Export?", required=True),
                "⚠️ Prev. Exported": st.column_config.CheckboxColumn("Prev. Exported?", disabled=True),
            },
            hide_index=True,
            disabled=["vocab", "phrase", "status", "tags", "⚠️ Prev. Exported"]
        )
        final_export = edited_export[edited_export['Export'].astype(bool)]
        dup_count    = int(final_export['⚠️ Prev. Exported'].astype(bool).sum()) \
            if '⚠️ Prev. Exported' in final_export.columns else 0
        if dup_count > 0:
            st.warning(f"⚠️ **{dup_count}** selected word(s) were previously exported this session.")

        if not final_export.empty:
            st.write("### Export Preview")
            st.dataframe(final_export[['vocab', 'phrase']], hide_index=True)
            card_count  = len(final_export)
            per_card_kb = 35.0 if include_audio else 2.5
            if include_reversed:       per_card_kb *= 1.15
            if include_sentence_audio: per_card_kb *= 1.8
            est_size_kb = card_count * per_card_kb
            size_label  = f"{est_size_kb / 1024:.2f} MB" if est_size_kb > 1024 else f"{est_size_kb:.1f} KB"
            st.info(f"📊 **{card_count} cards** • Est. .apkg size: **{size_label}**"
                    + (" + reversed" if include_reversed else "")
                    + (" + sentence audio" if include_sentence_audio else ""))

        # Y1-C: TPM projection warning
        if not final_export.empty and not st.session_state.dry_run:
            avg_chars  = (sum(len(str(r['vocab'])) + len(str(r['phrase']))
                              for _, r in final_export.iterrows()) / max(len(final_export), 1))
            est_tpm    = int(len(final_export) * (avg_chars + 650) / 4)
            tpm_remain = 1_000_000 - get_rolling_tpm()
            if est_tpm > tpm_remain * 0.6:
                st.warning(f"⚠️ This generation may use ~{est_tpm:,} tokens "
                           f"({est_tpm / 1_000_000 * 100:.0f}% of remaining hourly TPM budget).")

        quota_key = (st.session_state.rpd_count, len(final_export), batch_size)
        if st.session_state._quota_cache_key != quota_key:
            r_left = max(0, 20 - st.session_state.rpd_count)
            r_req  = math.ceil(len(final_export) / batch_size) if not final_export.empty else 0
            st.session_state._quota_cache     = (r_left, r_req)
            st.session_state._quota_cache_key = quota_key
        requests_left, required_requests = st.session_state._quota_cache
        st.info(f"💡 You have **{requests_left}** API requests left today. "
                f"This batch requires **{required_requests}** request(s).")

        if final_export.empty:
            st.warning("Select at least one word to export.")
        elif required_requests > requests_left and not st.session_state.dry_run:
            st.error("🛑 Exceeds Daily Limit! Reduce your selection or increase batch size.")
        else:
            if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
                st.session_state.failed_words = []
                raw_notes = []
                try:
                    raw_notes = process_anki_data(final_export, batch_size=batch_size,
                                                  dry_run=st.session_state.dry_run)
                    if raw_notes:
                        st.session_state.editing_notes          = raw_notes
                        st.session_state.editing_deck_name      = deck_name_input
                        st.session_state.editing_audio          = include_audio
                        st.session_state.editing_reversed       = include_reversed
                        st.session_state.editing_sentence_audio = include_sentence_audio   # Z2-D
                        st.rerun(scope="app")
                except Exception as e:
                    st.error(f"❌ Generation error: {e} — Status rolled back to 'New'.")
                    if raw_notes:
                        failed = [n.get('VocabRaw', '') for n in raw_notes]
                        st.session_state.vocab_df.loc[
                            st.session_state.vocab_df['vocab'].isin(failed), 'status'] = 'New'
                        save_to_github(st.session_state.vocab_df)

        # X1-D: Session Summary
        st.divider()
        with st.expander("📊 Session Summary", expanded=False):
            _api_this = max(0, st.session_state.rpd_count - st.session_state.get("session_api_calls_start", 0))
            _q_remain = max(0, 20 - st.session_state.rpd_count)
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Words Added",     st.session_state.session_words_added)
            sc2.metric("Cards Generated", st.session_state.session_cards_generated)
            sc3.metric("API Calls Used",  _api_this)
            sc4.metric("Quota Remaining", f"{_q_remain}/20")
            st.caption(f"📡 GitHub writes: **{gh_write_count()}** "
                       f"{'⚠️ near limit' if gh_write_count() >= GH_WRITE_WARN_THRESHOLD else '✅ healthy'} · "
                       f"⏰ Quota resets in **{quota_reset_countdown()}**")

        # N4-E: Export History
        st.divider()
        with st.expander("📜 Export History", expanded=False):
            if st.button("🔄 Load History", use_container_width=True):
                load_export_history.clear()
            history = load_export_history()
            if not history:
                st.info("No export history found.")
            else:
                rows = []
                for entry in reversed(history):
                    ts = entry.get("timestamp", "")
                    try: ts = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
                    except: pass
                    rows.append({"Timestamp": ts, "Deck": entry.get("deck_name", ""),
                                 "Cards": entry.get("card_count", 0),
                                 "Vocab": ", ".join(entry.get("vocabs", [])[:10])
                                          + ("…" if len(entry.get("vocabs", [])) > 10 else "")})
                hist_df = pd.DataFrame(rows)
                h_page  = st.number_input("History page", min_value=1,
                                          max_value=max(1, math.ceil(len(hist_df) / 10)),
                                          value=1, step=1, key="hist_page")
                h_start = (h_page - 1) * 10
                st.dataframe(hist_df.iloc[h_start:h_start + 10],
                             hide_index=True, use_container_width=True)
                st.caption(f"Showing {min(h_start + 10, len(hist_df))} of {len(hist_df)} records.")

    render_tab3()
