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
import requests

try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("Missing libraries! Please add gTTS and genanki to requirements.txt")
    st.stop()

st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")

THEME_COLOR = "#00ff41"
THEME_GLOW  = "rgba(0, 255, 65, 0.4)"
BG_COLOR    = "#111111"
BG_STRIPE   = "#181818"
TEXT_COLOR  = "#aaffaa"

_SPIN = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

CHANGELOG = [
    ("v3.1", "🖼️ Unsplash images · 🎨 POS badges · 🗣️ Pron guide · 🧠 4× smarter prompts"),
    ("v3.0", "🌸 Pastel theme · 📱 Mobile ↑↓ section reorder · 🐢 Slow audio"),
    ("v2.9", "💭 Hint field · 🎴 Cloze sentence mode · 🏷️ needs_review auto-tag"),
    ("v2.8", "🤖 Chain-of-thought verifier · 📏 Field length limits · ✅ Antonym gate"),
    ("v2.7", "⚡ Lite Mode prompt · 💰 Session budget · 🏷️ Named checkpoints"),
    ("v2.6", "📊 Quota forecast · 🎲 Few-shot rotator · 🔀 Sort bar · 📜 Changelog"),
]

LIGHT_MODE_CSS = """<style>
.stApp { background-color: #f0f7f0 !important; }
section[data-testid="stSidebar"] { background-color: #d8eed8 !important; }
section[data-testid="stSidebar"] * { color: #1a3a1a !important; }
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li { color: #1a3a1a !important; }
.stTextInput input, .stTextArea textarea { background-color: #e8f5e8 !important; color: #1a3a1a !important; }
</style>"""

_BOOT_T0 = time.perf_counter()

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

try:
    token              = st.secrets["GITHUB_TOKEN"]
    repo_name          = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

try:
    UNSPLASH_ACCESS_KEY = st.secrets.get("UNSPLASH_ACCESS_KEY", "")
except Exception:
    UNSPLASH_ACCESS_KEY = ""

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

PERSONAS: dict[str, str] = {
    "General":           "",
    "Medical":           "You are a medical lexicographer. Use precise clinical terminology. ",
    "Legal":             "You are a legal lexicographer. Use precise juridical definitions. ",
    "Coding / Tech":     "You are a software-engineering lexicographer. Use technical CS context. ",
    "Language Learning": "You are an EFL/ESL teacher. Prioritize learner-friendly definitions and natural example sentences. ",
}

DIFFICULTY_SUFFIX: dict[str, str] = {
    "Beginner":     "Use simple vocabulary (A1-A2 CEFR). Short example sentences. Avoid jargon.",
    "Intermediate": "",
    "Advanced":     "Use sophisticated vocabulary (C1-C2 CEFR). Complex, nuanced examples with idioms.",
}

LANG2_OPTIONS = [
    "None (disabled)",
    "Indonesian","Spanish","French","German",
    "Japanese","Chinese (Mandarin)","English (Simple)",
]

REGISTER_VALUES = ["Formal","Informal","Slang","Technical","Neutral"]
REGISTER_BADGE_CSS: dict[str, str] = {
    "Formal":    "color:#00ffff;border:1px solid #00ffff",
    "Informal":  "color:#ffff66;border:1px solid #ffff66",
    "Slang":     "color:#ff6b6b;border:1px solid #ff6b6b",
    "Technical": "color:#c084fc;border:1px solid #c084fc",
    "Neutral":   "color:#aaffaa;border:1px solid #aaffaa",
}

POS_BADGE_COLORS: dict[str, tuple] = {
    "Noun":         ("#00b4d8", "rgba(0,180,216,0.18)"),
    "Verb":         ("#06d6a0", "rgba(6,214,160,0.18)"),
    "Adjective":    ("#f77f00", "rgba(247,127,0,0.18)"),
    "Adverb":       ("#ffd166", "rgba(255,209,102,0.18)"),
    "Pronoun":      ("#c77dff", "rgba(199,125,255,0.18)"),
    "Preposition":  ("#ff6b9d", "rgba(255,107,157,0.18)"),
    "Conjunction":  ("#4cc9f0", "rgba(76,201,240,0.18)"),
    "Interjection": ("#ff4d6d", "rgba(255,77,109,0.18)"),
    "Phrase":       ("#aaffaa", "rgba(170,255,170,0.18)"),
}
POS_EMOJI: dict[str, str] = {
    "Noun": "📦", "Verb": "⚡", "Adjective": "🎨", "Adverb": "🔄",
    "Pronoun": "👤", "Preposition": "🔗", "Conjunction": "➕",
    "Interjection": "💬", "Phrase": "📝",
}

def make_pos_badge(pos: str) -> str:
    if not pos:
        return ""
    pos_title = pos.title()
    color, bg = POS_BADGE_COLORS.get(pos_title, ("#aaffaa", "rgba(170,255,170,0.18)"))
    emoji     = POS_EMOJI.get(pos_title, "📌")
    return (
        f'<span style="display:inline-block;font-size:0.72em;font-weight:700;'
        f'padding:3px 12px;border-radius:4px;letter-spacing:0.1em;text-transform:uppercase;'
        f'border:1px solid {color};color:{color};background:{bg};'
        f'font-family:monospace;white-space:nowrap">'
        f'{emoji} {pos_title}</span>'
    )

BACK_SECTIONS_DEFAULT = [
    "Romanization","Translation2","Definition","Pronunciation",
    "Hint","Register","Examples","Collocations","Synonyms",
    "Antonyms","Etymology","Mnemonic",
]
BACK_SECTION_META: dict[str, tuple] = {
    "Romanization":  ("🈳 ROMANIZATION",  "Romanization",  False),
    "Translation2":  ("🌐 TRANSLATION 2", "Translation2",  False),
    "Definition":    ("📜 DEFINITION",    "Definition",    False),
    "Pronunciation": ("🗣️ PRONUNCIATION", "Pronunciation", False),
    "Hint":          ("💭 HINT",          "Hint",          False),
    "Register":      ("🏷 REGISTER",      "Register",      False),
    "Examples":      ("🖋️ EXAMPLES",     "Examples",      False),
    "Collocations":  ("🔗 COLLOCATIONS",  "Collocations",  False),
    "Synonyms":      ("➕ SYNONYMS",      "Synonyms",      False),
    "Antonyms":      ("➖ ANTONYMS",      "Antonyms",      True),
    "Etymology":     ("🏛️ ETYMOLOGY",    "Etymology",     False),
    "Mnemonic":      ("💡 MEMORY HOOK",   "Mnemonic",      False),
}

def build_front_html() -> str:
    return (
        "{{#POSBadge}}<div style=\"text-align:center;padding:6px 0 4px\">{{POSBadge}}</div>{{/POSBadge}}\n"
        "{{#Image}}<div class=\"card-image-container\"><img src=\"{{Image}}\" class=\"card-image\"></div>{{/Image}}\n"
        "<div class=\"vellum-focus-container front\">\n"
        "<div class=\"prompt-text\">{{cloze:Text}}</div></div>"
    )

def build_back_html(section_order: list, include_antonyms: bool) -> str:
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
{{#AudioSlow}}<div style='display:none'>{{AudioSlow}}</div>{{/AudioSlow}}
</div>"""
    return html

_FEW_SHOT_POOL = [
    {"vocab":"serendipity","phrase":"We found the perfect cafe by pure serendipity.","translation":"kebetulan","part_of_speech":"Noun","pronunciation_ipa":"/ˌsɛrənˈdɪpɪti/","pronunciation_guide":"SEHR-en-DIP-ih-tee","definition_english":"The occurrence of events by chance in a happy way.","example_sentences":["It was pure serendipity that we met."],"synonyms_antonyms":{"synonyms":["chance","luck"],"antonyms":[]},"etymology":"Coined by Horace Walpole in 1754.","collocations":["by pure serendipity","happy serendipity","moment of serendipity"],"register":"Neutral","mnemonic":"SERENe DIP into a lucky pool — serene dipping is serendipity!","romanization":"","translation2":""},
    {"vocab":"run","phrase":"*He decided to run for office","translation":"mencalonkan diri","part_of_speech":"Verb","pronunciation_ipa":"/rʌn/","pronunciation_guide":"RUN","definition_english":"To compete in an election.","example_sentences":["She will run for president."],"synonyms_antonyms":{"synonyms":["campaign"],"antonyms":[]},"etymology":"Old English rinnan.","collocations":["run for office","run a campaign","run against"],"register":"Formal","mnemonic":"Picture a RUNner in a suit sprinting toward a ballot box.","romanization":"","translation2":""},
    {"vocab":"ephemeral","phrase":"The ephemeral beauty of cherry blossoms draws millions.","translation":"singkat","part_of_speech":"Adjective","pronunciation_ipa":"/ɪˈfɛm(ə)r(ə)l/","pronunciation_guide":"ih-FEM-er-ul","definition_english":"Lasting for only a short time.","example_sentences":["Fame can be ephemeral."],"synonyms_antonyms":{"synonyms":["fleeting","transient"],"antonyms":["permanent","lasting"]},"etymology":"Greek ephemeros, lasting a day.","collocations":["ephemeral beauty","ephemeral trend","ephemeral pleasure"],"register":"Formal","mnemonic":"EPH sounds like POOF — it vanishes instantly!","romanization":"","translation2":""},
    {"vocab":"ubiquitous","phrase":"Smartphones are now ubiquitous in modern life.","translation":"ada di mana-mana","part_of_speech":"Adjective","pronunciation_ipa":"/juːˈbɪkwɪtəs/","pronunciation_guide":"yoo-BIK-wih-tus","definition_english":"Present, appearing, or found everywhere.","example_sentences":["Coffee shops have become ubiquitous."],"synonyms_antonyms":{"synonyms":["omnipresent","pervasive"],"antonyms":["rare","scarce"]},"etymology":"Latin ubique, everywhere.","collocations":["ubiquitous presence","seemingly ubiquitous","become ubiquitous"],"register":"Formal","mnemonic":"YOU-BIK-wih-tus: YOU cannot quit seeing it — it is everywhere!","romanization":"","translation2":""},
    {"vocab":"ameliorate","phrase":"The policy aims to ameliorate living conditions.","translation":"memperbaiki","part_of_speech":"Verb","pronunciation_ipa":"/əˈmiːlɪəreɪt/","pronunciation_guide":"uh-MEE-lee-uh-rayt","definition_english":"To make something bad better.","example_sentences":["Aid organizations work to ameliorate poverty."],"synonyms_antonyms":{"synonyms":["improve","alleviate"],"antonyms":[]},"etymology":"Latin meliorare, to make better.","collocations":["ameliorate conditions","ameliorate suffering","ameliorate the situation"],"register":"Formal","mnemonic":"A MEAL improves your mood — a-MEALiorate!","romanization":"","translation2":""},
    {"vocab":"sycophant","phrase":"The boss was surrounded by sycophants who never disagreed.","translation":"penjilat","part_of_speech":"Noun","pronunciation_ipa":"/ˈsɪkəfant/","pronunciation_guide":"SIK-oh-fant","definition_english":"A person who flatters powerful people for personal gain.","example_sentences":["Politicians are often surrounded by sycophants."],"synonyms_antonyms":{"synonyms":["flatterer","yes-man"],"antonyms":["critic","detractor"]},"etymology":"Greek sykophantes, informer.","collocations":["office sycophant","surrounded by sycophants","shameless sycophant"],"register":"Informal","mnemonic":"SIC-O-FANT: sick of their fake praise — a sycophant!","romanization":"","translation2":""},
    {"vocab":"nonchalant","phrase":"She answered the difficult question in a nonchalant tone.","translation":"masa bodoh","part_of_speech":"Adjective","pronunciation_ipa":"/ˌnɒnʃəˈlɑːnt/","pronunciation_guide":"non-shuh-LAHNT","definition_english":"Feeling or appearing casually calm and relaxed.","example_sentences":["He was nonchalant about his promotion."],"synonyms_antonyms":{"synonyms":["casual","indifferent"],"antonyms":["anxious","concerned"]},"etymology":"French, not warm, not concerned.","collocations":["nonchalant attitude","act nonchalant","nonchalant tone"],"register":"Neutral","mnemonic":"NON-shuh-LAHNT: NON-care, like a cat in the sun.","romanization":"","translation2":""},
    {"vocab":"catharsis","phrase":"Writing in a journal provides emotional catharsis.","translation":"pelepasan emosi","part_of_speech":"Noun","pronunciation_ipa":"/kəˈθɑːsɪs/","pronunciation_guide":"kuh-THAR-sis","definition_english":"The process of releasing strong or repressed emotions.","example_sentences":["Crying during a film can be cathartic."],"synonyms_antonyms":{"synonyms":["release","purging"],"antonyms":[]},"etymology":"Greek katharsis, purification.","collocations":["emotional catharsis","provide catharsis","moment of catharsis"],"register":"Formal","mnemonic":"kuh-THAR-sis: THAR she blows — pressure released!","romanization":"","translation2":""},
]

def _get_few_shot_examples() -> str:
    picks = random.sample(_FEW_SHOT_POOL, min(2, len(_FEW_SHOT_POOL)))
    return json.dumps(picks, ensure_ascii=False)

TPM_WARN_THRESHOLD  = 700_000
TPM_BLOCK_THRESHOLD = 850_000

_GH_WRITE_LOG: list = []
GH_WRITE_WARN_THRESHOLD = 80

def _gh_write_tick():
    _GH_WRITE_LOG.append(1)

def gh_write_count() -> int:
    return len(_GH_WRITE_LOG)

def word_frequency_label(vocab: str) -> str:
    n = len(str(vocab).strip())
    if n <= 5:  return "🟢 Common"
    if n <= 9:  return "🟡 Uncommon"
    return "🔴 Rare"

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

def make_hint(text: str) -> str:
    if not text: return ""
    parts = str(text).strip().split()
    result = []
    for p in parts:
        if len(p) > 1:
            result.append(p[0] + "·" * (len(p) - 1))
        elif len(p) == 1:
            result.append(p)
    return " ".join(result)

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

def fetch_unsplash_url(args) -> str:
    vocab, access_key = args
    if not access_key or not vocab:
        return ""
    try:
        query = str(vocab).strip().split()[0]
        resp  = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": 3, "orientation": "squarish", "content_filter": "high"},
            headers={"Authorization": f"Client-ID {access_key}"},
            timeout=8,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                raw_url = results[0]["urls"]["small"]
                if "?" in raw_url:
                    return raw_url + "&fm=jpg&q=80"
                return raw_url + "?fm=jpg&q=80"
        elif resp.status_code == 403:
            print(f"Unsplash 403 for '{vocab}': invalid key or rate-limited.")
    except Exception as exc:
        print(f"Unsplash fetch error for '{vocab}': {exc}")
    return ""

def download_image_file(args) -> tuple:
    vocab_raw, image_url, temp_dir = args
    if not image_url:
        return vocab_raw, None, None
    try:
        resp = requests.get(
            image_url, timeout=10, stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; VocabApp/3.1)"},
        )
        if resp.status_code == 200:
            clean_base  = _RE_CLEAN_FNAME.sub("", vocab_raw)
            clean_fname = f"img_{clean_base}.jpg"
            file_path   = os.path.join(temp_dir, clean_fname)
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            if os.path.getsize(file_path) < 500:
                print(f"Image too small for '{vocab_raw}' — likely an error page.")
                return vocab_raw, None, None
            return vocab_raw, clean_fname, file_path
    except Exception as exc:
        print(f"Image download error for '{vocab_raw}': {exc}")
    return vocab_raw, None, None

CYBERPUNK_CSS = f"""
.card {{
    font-family: 'Roboto Mono', 'Consolas', monospace;
    font-size: 18px; line-height: 1.5;
    color: {THEME_COLOR}; background-color: {BG_COLOR};
    background-image: repeating-linear-gradient(0deg, {BG_STRIPE}, {BG_STRIPE} 1px, {BG_COLOR} 1px, {BG_COLOR} 20px);
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
    color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff;
}}
.cloze {{ color: {BG_COLOR}; background-color: {THEME_COLOR}; padding: 2px 4px; }}
.solved-text .cloze {{ color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }}
.vellum-section {{ margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed {THEME_COLOR}; }}
.section-header {{ font-weight: 600; color: #00ffff; border-left: 3px solid {THEME_COLOR}; padding-left: 10px; }}
.content {{ color: {TEXT_COLOR}; padding-left: 13px; }}
.register-badge {{ display: inline-block; font-size: 0.75em; font-weight: 700; padding: 1px 7px; border-radius: 3px; letter-spacing: 0.08em; text-transform: uppercase; margin-left: 6px; }}
.card-image-container {{ text-align: center; margin: 0 auto 14px; }}
.card-image {{ max-width: 90%; max-height: 160px; border-radius: 4px; border: 1px solid {THEME_COLOR}; box-shadow: 0 0 8px rgba(0,255,65,0.25); object-fit: cover; }}
@media (max-width: 480px) {{ .card {{ font-size: 16px; padding: 15px; }} .vellum-focus-container {{ padding: 15px; }} }}
"""

MINIMAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.card { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 16px; line-height: 1.65; color: #1a1a2e; background: #fefefe; padding: 28px 24px; text-align: left; }
.vellum-focus-container { background: #ffffff; padding: 28px 24px; margin: 0 auto 32px; border-radius: 10px; border: 1.5px solid #e2e8f0; box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center; }
.prompt-text { font-family: 'Inter', sans-serif; font-size: 1.9em; font-weight: 700; color: #1a1a2e; letter-spacing: -0.02em; line-height: 1.2; }
.cloze { color: #ffffff; background-color: #4f46e5; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.solved-text .cloze { color: #4f46e5; background: rgba(79,70,229,0.08); border-bottom: 2px solid #4f46e5; font-weight: 600; }
.vellum-section { margin: 12px 0; padding: 10px 0; border-bottom: 1px solid #f1f5f9; }
.section-header { font-size: 0.72em; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.content { color: #334155; padding-left: 0; font-size: 0.97em; }
.register-badge { display: inline-block; font-size: 0.68em; font-weight: 600; padding: 1px 8px; border-radius: 99px; letter-spacing: 0.06em; text-transform: uppercase; margin-left: 6px; }
.card-image-container { text-align: center; margin: 0 auto 14px; }
.card-image { max-width: 90%; max-height: 160px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.10); object-fit: cover; }
@media (max-width: 480px) { .card { font-size: 15px; padding: 16px; } }
"""

ACADEMIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700;900&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
.card { font-family: 'Source Serif 4', Georgia, serif; font-size: 16px; line-height: 1.75; color: #2c2416; background: #fdf8ef; padding: 28px 24px; text-align: left; }
.vellum-focus-container { background: #fffdf7; padding: 24px 20px; margin: 0 auto 28px; border-radius: 2px; border-top: 4px solid #8b3a2a; border-bottom: 1px solid #d4b896; border-left: 1px solid #e8d9c0; border-right: 1px solid #e8d9c0; box-shadow: 0 2px 8px rgba(139,58,42,0.08); text-align: center; }
.prompt-text { font-family: 'Merriweather', Georgia, serif; font-size: 1.75em; font-weight: 900; color: #1c1206; }
.cloze { color: #fffdf7; background-color: #8b3a2a; padding: 1px 6px; border-radius: 2px; }
.solved-text .cloze { color: #8b3a2a; background: rgba(139,58,42,0.08); border-bottom: 2px solid #8b3a2a; }
.vellum-section { margin: 10px 0; padding: 8px 0; border-bottom: 1px dashed #d4b896; }
.section-header { font-family: 'Merriweather', serif; font-size: 0.68em; font-weight: 700; color: #8b3a2a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.content { color: #2c2416; padding-left: 0; }
.register-badge { display: inline-block; font-size: 0.68em; font-weight: 600; padding: 1px 7px; border-radius: 2px; letter-spacing: 0.07em; text-transform: uppercase; margin-left: 6px; }
.card-image-container { text-align: center; margin: 0 auto 14px; }
.card-image { max-width: 90%; max-height: 160px; border-radius: 2px; border: 1px solid #d4b896; box-shadow: 2px 2px 6px rgba(139,58,42,0.10); object-fit: cover; filter: sepia(15%); }
@media (max-width: 480px) { .card { font-size: 15px; padding: 16px; } }
"""

PASTEL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;600;700&family=Nunito:wght@400;500;600;700&display=swap');
.card { font-family: 'Nunito', 'Comic Sans MS', cursive, sans-serif; font-size: 16px; line-height: 1.75; color: #3d2b4e; background: linear-gradient(135deg, #fdf4ff 0%, #f0f4ff 50%, #fff4f9 100%); padding: 28px 24px; text-align: left; }
.vellum-focus-container { background: #fff8fe; padding: 26px 22px; margin: 0 auto 28px; border-radius: 20px; border: 2px solid #d4b0e8; box-shadow: 4px 4px 0px #e8c4f0, 0 0 24px rgba(212,176,232,0.25); text-align: center; }
.prompt-text { font-family: 'Caveat', cursive; font-size: 2.3em; font-weight: 700; color: #6b21a8; letter-spacing: 0.01em; line-height: 1.2; }
.cloze { color: #fff; background: linear-gradient(135deg, #a855f7, #ec4899); padding: 2px 10px; border-radius: 14px; font-weight: 700; }
.solved-text .cloze { color: #a855f7; background: rgba(168,85,247,0.12); border-bottom: 2px dashed #ec4899; }
.vellum-section { margin: 10px 0; padding: 10px 14px; background: rgba(255,255,255,0.65); border-radius: 12px; border-left: 4px solid #d4b0e8; box-shadow: 2px 2px 0px rgba(212,176,232,0.3); }
.section-header { font-family: 'Caveat', cursive; font-size: 1.05em; font-weight: 700; color: #9333ea; margin-bottom: 4px; }
.content { color: #4a3060; font-size: 0.96em; }
.register-badge { display: inline-block; font-size: 0.7em; font-weight: 700; padding: 2px 10px; border-radius: 99px; letter-spacing: 0.05em; text-transform: uppercase; margin-left: 6px; background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(236,72,153,0.15)); border: 1px solid #d4b0e8; color: #7c3aed; }
.card-image-container { text-align: center; margin: 0 auto 14px; }
.card-image { max-width: 90%; max-height: 160px; border-radius: 16px; box-shadow: 3px 3px 0px rgba(212,176,232,0.5); object-fit: cover; }
@media (max-width: 480px) { .card { font-size: 15px; padding: 16px; } .vellum-focus-container { border-radius: 16px; } }
"""

CARD_THEMES: dict[str, dict] = {
    "🟢 Cyberpunk": {"css": None,         "description": "Dark matrix, glowing green borders",                 "front_color": "#ffffff", "accent": "#00ff41"},
    "⬜ Minimal":   {"css": MINIMAL_CSS,   "description": "Clean white, indigo accents, modern sans-serif",     "front_color": "#1a1a2e", "accent": "#4f46e5"},
    "📖 Academic":  {"css": ACADEMIC_CSS,  "description": "Warm parchment, serif fonts, dictionary style",      "front_color": "#1c1206", "accent": "#8b3a2a"},
    "🌸 Pastel":    {"css": PASTEL_CSS,    "description": "Soft purple & pink, Caveat handwritten font, lo-fi", "front_color": "#6b21a8", "accent": "#a855f7"},
}
CARD_THEMES["🟢 Cyberpunk"]["css"] = CYBERPUNK_CSS

def get_active_css() -> str:
    theme_key = st.session_state.get("card_theme", "🟢 Cyberpunk")
    return CARD_THEMES.get(theme_key, CARD_THEMES["🟢 Cyberpunk"])["css"] or CYBERPUNK_CSS

@st.cache_resource
def _get_gh_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="gh_bg")

@st.cache_resource
def get_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except GithubException as e:
        st.error(f"GitHub connection failed: {e}")
        st.stop()

_BOOT_T_GH_START = time.perf_counter()
repo = get_repo()
_BOOT_T_GH_DONE  = time.perf_counter()

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
    data = json.dumps({"date": str(date.today()), "rpd_count": rpd_count,
                       "timestamps": [ts.isoformat() for ts in timestamps]})
    for attempt in range(3):
        try:
            try:
                file = repo.get_contents(_COMBINED_USAGE_FILE)
                repo.update_file(file.path, "Update combined usage", data, file.sha)
                _gh_write_tick(); return
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(_COMBINED_USAGE_FILE, "Init combined usage", data)
                    _gh_write_tick(); return
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
            save_combined_usage(rpd, tss); return rpd, tss
        return 0, []
    except: return 0, []

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
        _gh_write_tick()
    except Exception as e: print(f"Safety log write error: {e}")

def log_safety_block(vocab_words: list, prompt: str):
    _get_gh_executor().submit(_bg_log_safety_block, list(vocab_words),
                              hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16])

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
        _gh_write_tick()
    except Exception as e: print(f"Export history write error: {e}")

def save_export_history(deck_name: str, card_count: int, vocab_list: list):
    _get_gh_executor().submit(_bg_save_export_history, deck_name, card_count, list(vocab_list))

@st.cache_data(ttl=120)
def load_export_history() -> list:
    try:
        file = repo.get_contents("export_history.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return data if isinstance(data, list) else []
    except: return []

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
            _gh_write_tick()
        except GithubException as e:
            if e.status == 404:
                repo.create_file(_WORD_CACHE_FILE, "Init word cache", data)
                _gh_write_tick()
    except Exception as e: print(f"Word cache save error: {e}")

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
            frame = _SPIN[r % len(_SPIN)]
            _slot.warning(f"{frame} RPM limit (5/min). Resuming in **{r}s**... _(quota governor active)_")
            time.sleep(1)
        _slot.empty()
    st.session_state.rpm_timestamps.append(now)
    save_combined_usage(st.session_state.rpd_count, st.session_state.rpm_timestamps)
    return time.perf_counter() - t0

@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
    except Exception as e:
        st.error(f"Gemini key error: {e}"); return None

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

def quota_reset_countdown() -> str:
    utc_now  = datetime.now(timezone.utc)
    midnight = (utc_now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta    = midnight - utc_now
    hours, rem = divmod(int(delta.total_seconds()), 3600)
    return f"{hours}h {rem // 60}m"

def render_quota_forecast(n_words: int, batch_size: int, requests_left: int):
    n_batches = math.ceil(n_words / max(batch_size, 1)) if n_words > 0 else 0
    pct = min(100.0, n_batches / max(requests_left, 1) * 100)
    icon = "🟢" if pct <= 40 else ("🟡" if pct <= 70 else "🔴")
    session_budget = st.session_state.get("session_budget", 10)
    session_start  = st.session_state.get("session_api_calls_start", 0)
    session_used   = max(0, st.session_state.rpd_count - session_start)
    session_left   = max(0, session_budget - session_used)
    budget_ok      = n_batches <= session_left
    budget_icon    = "✅" if budget_ok else "⚠️"
    st.markdown(
        f"""<div style="border:1px solid #444;border-radius:8px;padding:12px 14px;margin:8px 0;background:rgba(255,255,255,0.03)">
        <b>{icon} Quota Forecast</b><br>
        📦 <b>{n_words}</b> words → <b>{n_batches}</b> request(s) at batch size <b>{batch_size}</b><br>
        📊 Daily: uses <b>{n_batches}/{requests_left}</b> remaining ({pct:.0f}% of today's budget)<br>
        {budget_icon} Session budget: <b>{session_used + n_batches}/{session_budget}</b> ({session_left} left in session)<br>
        ⏰ Daily quota resets in <b>{quota_reset_countdown()}</b></div>""",
        unsafe_allow_html=True
    )

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
        st.error(f"Phrase enrichment failed: {e}")
    return {}

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6, dry_run=False):
    TARGET_LANG    = st.session_state.get("target_lang", "Indonesian")
    model_name     = st.session_state.get("gemini_model_name", "gemini-2.5-flash-lite")
    model          = get_gemini_model(st.session_state.gemini_key, model_name)
    if not model: return []

    persona_prefix = PERSONAS.get(st.session_state.get("persona", "General"), "")
    diff_str       = DIFFICULTY_SUFFIX.get(st.session_state.get("difficulty", "Intermediate"), "")
    use_lite_mode  = st.session_state.get("use_lite_mode", False)
    session_budget = st.session_state.get("session_budget", 10)
    session_start  = st.session_state.get("session_api_calls_start", 0)

    lite_note = (
        "\nLITE MODE ACTIVE: Return empty string '' for 'collocations', 'etymology', and 'mnemonic'."
        if use_lite_mode else ""
    )
    use_mnemonic = st.session_state.get("use_mnemonic", False) and not use_lite_mode
    mnemonic_rule = (
        "\n11. 'mnemonic' — Follow these steps silently; output ONLY the final sentence:\n"
        "    Step 1 → Find the most phonetically STRIKING syllable cluster in 'vocab'\n"
        "    Step 2 → Find a common English word or vivid image that SOUNDS like that cluster\n"
        "    Step 3 → Build an ABSURD, memorable story connecting that sound to the meaning\n"
        "    Output: ONE punchy sentence, max 20 words. Must include the sound hook."
        if use_mnemonic else ""
    )
    difficulty_rule = f"\n10. DIFFICULTY: {diff_str} Tailor all outputs accordingly." if diff_str else ""
    is_cjk = TARGET_LANG in ("Japanese", "Chinese (Mandarin)")
    romanization_rule = ("\n12. 'romanization': Japanese=Romaji, Chinese=Pinyin with tone marks." if is_cjk else "")
    lang2 = st.session_state.get("target_lang2", "None (disabled)")
    lang2_rule = (f"\n13. 'translation2': ONLY the {lang2} translation of 'vocab'. NEVER a full sentence."
                  if lang2 != "None (disabled)" else "")
    length_limits = (
        "\n14. FIELD LENGTH LIMITS: 'definition_english' max 25 words. "
        "Each 'example_sentences' item max 15 words. "
        "'etymology' max 20 words. 'mnemonic' max 20 words."
    )
    pronunciation_guide_rule = (
        "\n15. 'pronunciation_guide': Phonetic spelling using SIMPLE English syllables only. "
        "UPPERCASE the PRIMARY stressed syllable. Hyphens between syllables. Max 8 syllables. "
        "Examples: 'SEHR-en-DIP-ih-tee', 'NON-shuh-lahnt', 'uh-MEE-lee-uh-rayt'. "
        "Output ONLY the phonetics — NO 'say:' prefix."
    )

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

    checkpoint = st.session_state.get("generation_checkpoint", [])
    if checkpoint:
        all_new_data = list(checkpoint)
        _ckpt_vocabs = {c.get('vocab','').strip().lower() for c in all_new_data}
        cached_results = [c for c in cached_results if c.get('vocab','').strip().lower() not in _ckpt_vocabs]

    with st.status("🤖 Processing AI Batches (RPM Throttled)...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        _anim_slot   = st.empty()

        for idx, batch in enumerate(batches):
            _anim_slot.info(
                f"{_SPIN[idx % len(_SPIN)]} **Batch {idx+1}/{len(batches)}** — "
                f"preparing `{'`, `'.join(v[0] for v in batch)}`..."
            )

            if st.session_state.rpd_count >= 20:
                st.warning("🛑 Daily AI Limit (20 requests) reached. Try again tomorrow.")
                for vp in batch: st.session_state.failed_words.append(vp[0])
                break

            session_used = st.session_state.rpd_count - session_start
            if session_used >= session_budget:
                st.warning(
                    f"🛑 Session budget reached ({session_used}/{session_budget} calls). "
                    f"Raise your budget in ⚙️ Settings."
                )
                for vp in batch: st.session_state.failed_words.append(vp[0])
                break

            t_rpm         = enforce_rpm()
            batch_dicts   = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            vocab_words   = [v[0] for v in batch]
            few_shot_json = _get_few_shot_examples()

            prompt = f"""{persona_prefix}You are an expert educational lexicographer. Think step-by-step:
1. Identify primary sense from phrase or context.
2. Generate accurate fields.
3. Ensure JSON validity.

STEP 0 — SILENT SELF-CHECK (do NOT output this step): Before writing JSON, silently verify:
(a) Will 'translation' contain ONLY the {TARGET_LANG} word/phrase, NOT a full sentence?
(b) Does every item have ALL required keys including 'pronunciation_guide'?
(c) Is 'register' exactly one of: Formal, Informal, Slang, Technical, Neutral?
(d) Are field lengths within limits in Rule 14?
(e) For polysemous words: has the DISAMBIGUATION RULE been applied?
(f) Is 'pronunciation_guide' clean phonetics with NO 'say:' prefix?
(g) For 'etymology': am I >90% confident? If not, have I written "Origin uncertain."?

Output EXACTLY {len(batch_dicts)} items as a JSON array. No extra text or commentary.{lite_note}

SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.

DISAMBIGUATION RULE: If 'vocab' is polysemous (has multiple common meanings, e.g. run/set/bar/pitch/bank/break/fall/draw/lead/plant):
(a) Silently enumerate all major senses of the word
(b) Select the sense that BEST aligns with the 'phrase' or context clues provided
(c) If no phrase is given, select the MOST COMMON everyday sense
(d) ALL output fields must reflect this single selected sense consistently — no mixing of senses

RULES:
1. Copy ALL input fields exactly.
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT.
3. IF 'phrase' is normal text: Use ONLY to determine which meaning of 'vocab' to use.
4. IF 'phrase' is empty: Generate ONE simple sentence (max 12 words).
5. EXACT 'vocab' must remain unchanged.
6. MANDATORY: 'translation' = ONLY the {TARGET_LANG} translation of the word. NEVER a full sentence.
7. 'part_of_speech' MUST be: Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection, or Phrase.
8. 'collocations': exactly 2-3 natural word combinations as a JSON array.
9. 'register' MUST be exactly one of: Formal, Informal, Slang, Technical, Neutral.
   DETECTION METHOD — scan for linguistic markers before assigning:
   • SLANG markers: youth language, contractions (gonna/wanna/lit/vibe/sick/dude/lowkey), expletives
   • FORMAL markers: Latinate vocabulary, passive constructions, juridical/bureaucratic terminology
   • TECHNICAL markers: domain-specific jargon, specialized nomenclature, acronyms, symbols
   • INFORMAL markers: casual everyday language, simple syntax, spoken-style contractions
   When phrase context and the word itself signal DIFFERENT registers, prioritize the PHRASE.
   ANTONYM RULE: If no true antonym exists, return [] for 'antonyms'. NEVER invent forced antonyms.{difficulty_rule}{mnemonic_rule}{romanization_rule}{lang2_rule}{length_limits}{pronunciation_guide_rule}
   ETYMOLOGY RULE: If you are NOT >90% confident in the word's historical origin,
   write exactly "Origin uncertain." NEVER fabricate plausible-sounding but unverified etymologies.

EXAMPLES (varied each batch for diversity):
{few_shot_json}

BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}

/* REQUIRED JSON KEYS per item:
   vocab, translation, part_of_speech, pronunciation_ipa, pronunciation_guide,
   definition_english, example_sentences, synonyms_antonyms,
   etymology, collocations, register, mnemonic, romanization, translation2 */"""

            log_tpm_chars(len(prompt))

            if not dry_run:
                _is_safe, _proj = check_tpm_preflight(prompt)
                if not _is_safe:
                    st.error(f"🛑 TPM blocked `{', '.join(vocab_words)}`: ~{_proj:,} tokens > {TPM_BLOCK_THRESHOLD:,}.")
                    st.session_state.failed_words.extend(vocab_words)
                    timings.append({"batch": idx+1, "words": ", ".join(vocab_words),
                                    "rpm_wait_s": round(t_rpm,3), "gemini_s": 0.0, "cached": False, "note": "TPM_BLOCKED"})
                    progress_bar.progress((idx+1)/len(batches))
                    _anim_slot.empty(); continue
                elif _proj > TPM_WARN_THRESHOLD:
                    st.warning(f"⚠️ TPM approaching: ~{_proj:,} / 1,000,000.")

            success     = False
            t_api_start = time.perf_counter()

            if dry_run:
                st.info(f"🔬 Dry-run: `{', '.join(vocab_words)}`")
                mock = [{"vocab": v[0], "phrase": v[1], "translation": "mock-"+v[0],
                         "part_of_speech": "Noun", "pronunciation_ipa": "/mock/",
                         "pronunciation_guide": "MOCK",
                         "definition_english": "Simulated definition for testing.",
                         "example_sentences": ["Mock example sentence for dry run."],
                         "synonyms_antonyms": {"synonyms": ["mock","simulated"], "antonyms": []},
                         "etymology": "Simulated.", "collocations": ["mock one","mock two"],
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
                            if finish in ("3","SAFETY","FinishReason.SAFETY"):
                                st.warning(f"🛡️ Safety filter blocked `{', '.join(vocab_words)}`.")
                                log_safety_block(vocab_words, prompt)
                                st.session_state.failed_words.extend(vocab_words)
                                break
                        parsed = robust_json_parse(response.text)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            all_new_data.extend(parsed)
                            for card in parsed: word_cache[card['vocab'].strip().lower()] = card
                            recovered = [c.get('vocab','') for c in parsed]
                            missed    = [v for v in vocab_words if v not in recovered]
                            if missed:
                                st.session_state.failed_words.extend(missed)
                                st.warning(f"⚠️ Partial batch {idx+1}: {len(parsed)}/{len(batch_dicts)}. Missed: `{', '.join(missed)}`")
                            else:
                                st.markdown(f"✅ **Batch {idx+1}**: `{', '.join(vocab_words)}`")
                            ckpt_name = f"{len(all_new_data)} cards · {datetime.now().strftime('%I:%M %p')}"
                            st.session_state.generation_checkpoint = list(all_new_data)
                            st.session_state.checkpoint_name       = ckpt_name
                            success = True; break
                    except Exception as e:
                        if "429" in str(e):
                            backoff = 20 + (2 ** attempt) + random.uniform(0, 1)
                            _slot = st.empty()
                            for r in range(int(backoff), 0, -1):
                                frame = _SPIN[r % len(_SPIN)]
                                _slot.warning(f"{frame} 429 Rate Limit. Retrying in **{r}s**... (attempt {attempt+1}/3)")
                                time.sleep(1)
                            _slot.empty()
                        else:
                            time.sleep(2)

            t_api_elapsed = time.perf_counter() - t_api_start
            timings.append({"batch": idx+1, "words": ", ".join(vocab_words),
                            "rpm_wait_s": round(t_rpm,3), "gemini_s": round(t_api_elapsed,3),
                            "cached": False, "note": ""})
            if not success and not dry_run:
                st.error(f"❌ **Failed**: `{', '.join(vocab_words)}` — queued for retry")
                st.session_state.failed_words.extend(vocab_words)
            progress_bar.progress((idx+1)/len(batches))

        _anim_slot.empty()
        total = len(all_new_data) + len(cached_results)
        status_log.update(label=f"✅ AI Complete! ({total} items | {len(cached_results)} cached)",
                          state="complete", expanded=False)

    if not st.session_state.failed_words:
        st.session_state.generation_checkpoint = []
        st.session_state.checkpoint_name       = ""
    st.session_state.word_cache = word_cache
    save_word_cache(word_cache)
    if timings:
        with st.expander("⏱️ Batch Performance Timings", expanded=False):
            st.dataframe(pd.DataFrame(timings), hide_index=True)
    return cached_results + all_new_data


def process_anki_data(df_subset, batch_size=6, dry_run=False):
    t0        = time.perf_counter()
    cache_key = str(pd.util.hash_pandas_object(df_subset).sum())
    cached    = st.session_state.get("processed_cache", {})
    if (cached.get("key") == cache_key
            and (datetime.now() - cached.get("time", datetime.min)).total_seconds() < 300):
        st.info("♻️ Using cached processed notes.")
        return cached["notes"]

    df_clean          = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = (
        df_clean.reindex(columns=['vocab','phrase'], fill_value='')[['vocab','phrase']].values.tolist()
    )
    tags_lookup: dict[str, list] = {}
    if 'tags' in df_clean.columns:
        tags_series = (df_clean.assign(_vk=df_clean['vocab'].astype(str).str.strip().str.lower())
                       .set_index('_vk')['tags'].fillna(''))
        tags_lookup = {str(k): sanitize_tags(str(v)) for k, v in tags_series.items() if str(v).strip()}

    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size, dry_run=dry_run)

    img_url_lookup: dict[str, str] = {}
    use_images = st.session_state.get("use_images", False) and bool(UNSPLASH_ACCESS_KEY)
    if use_images and all_card_data:
        wc         = st.session_state.get("word_cache", {})
        all_vocabs = [c.get("vocab","").strip().lower() for c in all_card_data if c.get("vocab")]
        for v in all_vocabs:
            cached_url = wc.get(v, {}).get("_unsplash_url", "")
            if cached_url:
                img_url_lookup[v] = cached_url
        missing_img = [v for v in all_vocabs if v not in img_url_lookup]
        if missing_img:
            with st.spinner(f"🖼️ Fetching {len(missing_img)} image(s) from Unsplash…"):
                fetch_args = [(v, UNSPLASH_ACCESS_KEY) for v in missing_img]
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                    fetched_urls = list(exc.map(fetch_unsplash_url, fetch_args))
            for v, url in zip(missing_img, fetched_urls):
                if url:
                    img_url_lookup[v] = url
                    if v in wc:
                        wc[v]["_unsplash_url"] = url
            st.session_state.word_cache = wc
            save_word_cache(wc)
            st.caption(f"🖼️ Fetched {len([u for u in fetched_urls if u])} / {len(missing_img)} images from Unsplash")

    cloze_sentence  = st.session_state.get("cloze_sentence_mode", False)
    processed_notes = []

    for card_data in all_card_data:
        required = ["vocab","translation","part_of_speech"]
        if not all(k in card_data and card_data[k] for k in required):
            st.error(f"⚠️ Missing required fields for `{card_data.get('vocab','?')}` — skipping"); continue
        vocab_raw = str(card_data.get("vocab","")).strip().lower()
        if not vocab_raw: continue

        vocab_cap        = cap_first(vocab_raw)
        phrase           = fix_vocab_casing(_clean_field(card_data.get("phrase","")), vocab_raw)
        formatted        = highlight_vocab(phrase, vocab_raw) if phrase else ""
        translation      = _clean_field(card_data.get("translation","?"))
        pos              = str(card_data.get("part_of_speech","")).title()
        ipa              = card_data.get("pronunciation_ipa","")
        eng_def          = _clean_field(card_data.get("definition_english",""))
        examples         = [_clean_field(e) for e in (card_data.get("example_sentences") or [])[:3]]
        ex_field         = "<ul>"+"".join(f"<li><i>{e}</i></li>" for e in examples)+"</ul>" if examples else ""
        syn_ant          = card_data.get("synonyms_antonyms") or {}
        synonyms         = ensure_trailing_dot(", ".join(cap_first(s) for s in (syn_ant.get("synonyms") or [])[:5]))
        antonyms         = ensure_trailing_dot(", ".join(cap_first(a) for a in (syn_ant.get("antonyms") or [])[:5]))
        etymology        = normalize_spaces(card_data.get("etymology",""))
        collocations_raw = card_data.get("collocations") or []
        if isinstance(collocations_raw, list):   collocations = "; ".join(cap_first(c) for c in collocations_raw[:3] if c)
        elif isinstance(collocations_raw, str):  collocations = cap_first(collocations_raw.strip())
        else:                                    collocations = ""
        register_raw  = str(card_data.get("register","") or "").strip().title()
        register      = register_raw if register_raw in REGISTER_VALUES else "Neutral"
        reg_css       = REGISTER_BADGE_CSS.get(register, REGISTER_BADGE_CSS["Neutral"])
        register_html = f'<span class="register-badge" style="{reg_css}">{register}</span>'
        mnemonic_raw  = str(card_data.get("mnemonic","") or "").strip()
        mnemonic      = cap_first(mnemonic_raw) if mnemonic_raw else ""
        romanization  = normalize_spaces(str(card_data.get("romanization","") or ""))
        translation2_raw = str(card_data.get("translation2","") or "").strip()
        translation2     = _clean_field(translation2_raw) if translation2_raw else ""
        hint             = make_hint(translation)
        pos_badge_html   = make_pos_badge(pos)

        pron_guide_raw = str(card_data.get("pronunciation_guide","") or "").strip()
        pron_guide     = re.sub(r'^say:\s*', '', pron_guide_raw, flags=re.IGNORECASE).strip()
        pron_field     = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"
        if pron_guide:
            pron_field += (
                f"<br><small style='opacity:0.75;font-style:italic;letter-spacing:0.03em'>"
                f"say: {pron_guide}</small>"
            )

        if cloze_sentence and examples:
            first_plain = _RE_STRIP_HTML.sub('', examples[0]).strip()
            _cloze_pat  = re.compile(r'\b' + re.escape(vocab_raw) + r'\b', re.IGNORECASE)
            if _cloze_pat.search(first_plain):
                text_field = _cloze_pat.sub(f'{{{{c1::{vocab_cap}}}}}', first_plain, count=1)
            else:
                text_field = (f"{formatted}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
                              if formatted else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>")
        else:
            text_field = (f"{formatted}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
                          if formatted else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>")

        note = {
            "VocabRaw":         vocab_raw,
            "Text":             text_field,
            "Pronunciation":    pron_field,
            "Definition":       eng_def,
            "Examples":         ex_field,
            "Synonyms":         synonyms,
            "Antonyms":         antonyms,
            "Etymology":        etymology,
            "Collocations":     collocations,
            "Register":         register_html,
            "RegisterLabel":    register,
            "Mnemonic":         mnemonic,
            "Romanization":     romanization,
            "Translation2":     translation2,
            "TranslationPlain": translation,
            "Hint":             hint,
            "Tags":             list(tags_lookup.get(vocab_raw, [])),
            "POSBadge":         pos_badge_html,
            "_unsplash_url":    img_url_lookup.get(vocab_raw, ""),
        }
        q_score = score_card(note)
        note["_quality_score"] = q_score
        if q_score >= 80:   note["Tags"].append("quality_high")
        elif q_score >= 60: note["Tags"].append("quality_medium")
        else:
            note["Tags"].append("quality_low")
            note["Tags"].append("needs_review")
        processed_notes.append(note)

    st.session_state.processed_cache = {"key": cache_key, "notes": processed_notes, "time": datetime.now()}
    st.caption(f"⏱️ `process_anki_data`: {time.perf_counter()-t0:.3f}s — {len(processed_notes)} notes")
    return processed_notes


def generate_audio_file(args):
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

def generate_slow_audio_file(args):
    vocab, temp_dir = args
    try:
        clean_vocab = _RE_AUDIO_CLEAN.sub('', vocab).strip()
        clean_fname = _RE_CLEAN_FNAME.sub('', clean_vocab) + "_slow.mp3"
        file_path   = os.path.join(temp_dir, clean_fname)
        if clean_vocab:
            gTTS(text=clean_vocab, lang='en', slow=True).save(file_path)
            return vocab, clean_fname, file_path
    except Exception as e: print(f"Slow audio error for {vocab}: {e}")
    return vocab, None, None

def generate_sentence_audio_file(args):
    vocab_raw, sentence, temp_dir = args
    if not sentence or not sentence.strip(): return vocab_raw, None, None
    try:
        clean_fname = "sent_" + _RE_CLEAN_FNAME.sub('', vocab_raw) + ".mp3"
        file_path   = os.path.join(temp_dir, clean_fname)
        clean_sent  = _RE_AUDIO_CLEAN.sub(' ', sentence).strip()
        if clean_sent:
            gTTS(text=clean_sent, lang='en', slow=False).save(file_path)
            return vocab_raw, clean_fname, file_path
    except Exception as e: print(f"Sentence audio error for {vocab_raw}: {e}")
    return vocab_raw, None, None


def create_anki_package(notes_data, deck_name, generate_audio=True, deck_id=2059400110,
                        include_antonyms=True, include_reversed=False,
                        sentence_audio=False, slow_audio=False, generate_images=False):
    t0         = time.perf_counter()
    front_html = build_front_html()
    _section_order = st.session_state.get("back_section_order", list(BACK_SECTIONS_DEFAULT))
    back_html      = build_back_html(_section_order, include_antonyms)

    model_id = st.session_state.get("model_id", 1607392319)
    my_model = genanki.Model(
        model_id, 'Cyberpunk Vocab Model',
        fields=[
            {'name': 'Text'},         {'name': 'Pronunciation'},
            {'name': 'Definition'},   {'name': 'Examples'},
            {'name': 'Collocations'}, {'name': 'Register'},
            {'name': 'Synonyms'},     {'name': 'Antonyms'},
            {'name': 'Etymology'},    {'name': 'Romanization'},
            {'name': 'Translation2'}, {'name': 'Mnemonic'},
            {'name': 'Hint'},         {'name': 'Audio'},
            {'name': 'AudioSlow'},    {'name': 'POSBadge'},
            {'name': 'Image'},
        ],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=get_active_css(), model_type=genanki.Model.CLOZE
    )

    reversed_model_id = st.session_state.get("reversed_model_id", (model_id+7919)%(1<<31))
    rev_front = """<div class="vellum-focus-container front">
<div class="prompt-text" style="color:#ffff66">{{Translation}}</div>
{{#Pronunciation}}<div style="color:#aaffaa;font-size:0.9em;margin-top:8px">{{Pronunciation}}</div>{{/Pronunciation}}
</div>"""
    rev_back = """<div class="vellum-focus-container back">
<div class="prompt-text" style="color:#ff00ff">{{VocabWord}}</div></div>
<div class="vellum-detail-container">
{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div>
<div class="content">{{Definition}}</div></div>{{/Definition}}
{{#Mnemonic}}<div class="vellum-section"><div class="section-header">💡 MEMORY HOOK</div>
<div class="content">{{Mnemonic}}</div></div>{{/Mnemonic}}
</div>"""
    reversed_model = genanki.Model(
        reversed_model_id, 'Cyberpunk Vocab Reversed',
        fields=[{'name':'Translation'},{'name':'Pronunciation'},
                {'name':'VocabWord'},  {'name':'Definition'},{'name':'Mnemonic'}],
        templates=[{'name':'Reversed','qfmt':rev_front,'afmt':rev_back}],
        css=get_active_css(),
    )

    my_deck     = genanki.Deck(deck_id, deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map, slow_audio_map, sent_audio_map, image_map = {}, {}, {}, {}

        if generate_audio:
            t_audio       = time.perf_counter()
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                for vk, fn, fp in exc.map(generate_audio_file, [(v, temp_dir) for v in unique_vocabs]):
                    if fn: media_files.append(fp); audio_map[vk] = f"[sound:{fn}]"
            st.caption(f"⏱️ Audio: {time.perf_counter()-t_audio:.2f}s for {len(unique_vocabs)} words")

            if slow_audio:
                t_slow = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                    for vk, fn, fp in exc.map(generate_slow_audio_file, [(v, temp_dir) for v in unique_vocabs]):
                        if fn: media_files.append(fp); slow_audio_map[vk] = f"[sound:{fn}]"
                st.caption(f"⏱️ Slow audio: {time.perf_counter()-t_slow:.2f}s for {len(slow_audio_map)} words")

            if sentence_audio:
                t_sent    = time.perf_counter()
                sent_args = []
                for n in notes_data:
                    plain_sent = _RE_STRIP_HTML.sub(' ', n.get('Examples','') or '').strip()
                    first_sent = plain_sent.split('  ')[0].strip() if plain_sent else ''
                    if first_sent: sent_args.append((n['VocabRaw'], first_sent, temp_dir))
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                    for vk, fn, fp in exc.map(generate_sentence_audio_file, sent_args):
                        if fn: media_files.append(fp); sent_audio_map[vk] = f"[sound:{fn}]"
                st.caption(f"⏱️ Sentence audio: {time.perf_counter()-t_sent:.2f}s for {len(sent_audio_map)} sentences")

        if generate_images:
            t_img       = time.perf_counter()
            needs_refetch = [(n['VocabRaw'], n) for n in notes_data
                             if not n.get('_unsplash_url','') and UNSPLASH_ACCESS_KEY]
            if needs_refetch:
                st.caption(f"🔄 Re-fetching {len(needs_refetch)} missing Unsplash URL(s)…")
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                    refetched = list(exc.map(fetch_unsplash_url,
                                             [(vk, UNSPLASH_ACCESS_KEY) for vk, _ in needs_refetch]))
                for (vk, note), url in zip(needs_refetch, refetched):
                    if url:
                        note['_unsplash_url'] = url
            img_args    = [(n['VocabRaw'], n.get('_unsplash_url',''), temp_dir) for n in notes_data]
            failed_imgs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                for vk, fn, fp in exc.map(download_image_file, img_args):
                    if fn:
                        media_files.append(fp)
                        image_map[vk] = fn
                    else:
                        failed_imgs.append(vk)
            st.caption(f"⏱️ Images: {time.perf_counter()-t_img:.2f}s · {len(image_map)} downloaded · {len(failed_imgs)} failed")
            if failed_imgs:
                st.warning(f"⚠️ Image download failed for: `{', '.join(failed_imgs)}` — cards will show no image (not broken icon).")

        all_fields_check = ['Definition','Examples','Collocations','Synonyms',
                            'Antonyms','Mnemonic','Romanization','Translation2','Hint']
        scores      = [n.get('_quality_score',0) for n in notes_data]
        avg_quality = int(sum(scores)/len(scores)) if scores else 0
        deck_stats  = {
            "total_cards":      len(notes_data),
            "avg_quality":      avg_quality,
            "field_completion": {f: sum(1 for n in notes_data if str(n.get(f,'')).strip())
                                 for f in all_fields_check},
            "images_embedded":  len(image_map),
        }

        exported_hashes = st.session_state.get("exported_hashes", set())
        for note_data in notes_data:
            guid_input = note_data['VocabRaw'] + deck_name
            vocab_hash = str(int(hashlib.sha256(guid_input.encode('utf-8')).hexdigest(),16) % (10**10))
            exported_hashes.add(hashlib.sha256(note_data['VocabRaw'].encode('utf-8')).hexdigest()[:16])
            my_deck.add_note(genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'],
                    note_data['Pronunciation'],
                    note_data['Definition'],
                    note_data['Examples'],
                    note_data.get('Collocations',''),
                    note_data.get('Register',''),
                    note_data['Synonyms'],
                    note_data['Antonyms'],
                    note_data['Etymology'],
                    note_data.get('Romanization',''),
                    note_data.get('Translation2',''),
                    note_data.get('Mnemonic',''),
                    note_data.get('Hint',''),
                    audio_map.get(note_data['VocabRaw'],'') + sent_audio_map.get(note_data['VocabRaw'],''),
                    slow_audio_map.get(note_data['VocabRaw'],''),
                    note_data.get('POSBadge',''),
                    image_map.get(note_data['VocabRaw'],''),
                ],
                tags=note_data['Tags'], guid=vocab_hash
            ))
            if include_reversed:
                rev_guid = str(int(hashlib.sha256(
                    (note_data['VocabRaw']+deck_name+"_rev").encode('utf-8')).hexdigest(),16) % (10**10))
                my_deck.add_note(genanki.Note(
                    model=reversed_model,
                    fields=[note_data.get('TranslationPlain',''), note_data['Pronunciation'],
                            note_data['VocabRaw'], note_data['Definition'], note_data.get('Mnemonic','')],
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
    st.caption(f"⏱️ `create_anki_package` total: {time.perf_counter()-t0:.2f}s")
    return buffer, deck_stats


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
        if e.status == 404: return pd.DataFrame(columns=['vocab','phrase','status','tags'])
        st.stop()
    except: st.stop()

def save_to_github(dataframe: pd.DataFrame) -> bool:
    st.session_state.undo_df = st.session_state.vocab_df.copy()
    t0        = time.perf_counter()
    mask      = dataframe['vocab'].astype(str).str.strip().str.len() > 0
    dataframe = dataframe[mask].drop_duplicates(subset=['vocab'], keep='last')
    drop_cols = [c for c in ['Export','⚠️ Prev. Exported','_quality_score','RegisterLabel']
                 if c in dataframe.columns]
    if drop_cols: dataframe = dataframe.drop(columns=drop_cols)
    csv_data  = dataframe.to_csv(index=False)
    csv_bytes = len(csv_data.encode('utf-8'))
    if csv_bytes > 500_000:
        st.warning(f"⚠️ vocabulary.csv is **{csv_bytes/1024:.0f} KB** — approaching GitHub limits.")
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
        _gh_write_tick()
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial commit", csv_data)
            _gh_write_tick()
    load_data.clear()
    st.caption(f"⏱️ GitHub save: {time.perf_counter()-t0:.2f}s ({csv_bytes/1024:.0f} KB)")
    return True

_BOOT_T_GH2 = time.perf_counter()
if "rpd_count" not in st.session_state or "rpm_timestamps" not in st.session_state:
    _init_rpd, _init_rpm = load_combined_usage()
else:
    _init_rpd = st.session_state.rpd_count
    _init_rpm = st.session_state.rpm_timestamps
_BOOT_T_USAGE_DONE = time.perf_counter()

st.session_state.setdefault("gemini_key",              DEFAULT_GEMINI_KEY)
st.session_state.setdefault("vocab_df",                load_data().copy())
st.session_state.setdefault("rpd_count",               _init_rpd)
st.session_state.setdefault("rpm_timestamps",          _init_rpm)
st.session_state.setdefault("deck_id",                 2059400110)
st.session_state.setdefault("bulk_preview_df",         None)
st.session_state.setdefault("apkg_buffer",             None)
st.session_state.setdefault("processed_vocabs",        [])
st.session_state.setdefault("model_id",                1607392319)
st.session_state.setdefault("reversed_model_id",       (1607392319+7919)%(1<<31))
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
st.session_state.setdefault("target_lang2",            "None (disabled)")
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
st.session_state.setdefault("editing_sentence_audio",  False)
st.session_state.setdefault("editing_slow_audio",      False)
st.session_state.setdefault("editing_images",          False)
st.session_state.setdefault("generation_checkpoint",   [])
st.session_state.setdefault("checkpoint_name",         "")
st.session_state.setdefault("deck_stats",              {})
st.session_state.setdefault("light_mode",              False)
st.session_state.setdefault("session_words_added",     0)
st.session_state.setdefault("session_cards_generated", 0)
st.session_state.setdefault("_quota_reset_warned",     False)
st.session_state.setdefault("session_api_calls_start", _init_rpd)
st.session_state.setdefault("_boot_profiled",          False)
st.session_state.setdefault("card_theme",              "🟢 Cyberpunk")
st.session_state.setdefault("use_lite_mode",           False)
st.session_state.setdefault("session_budget",          10)
st.session_state.setdefault("cloze_sentence_mode",     False)
st.session_state.setdefault("editing_cloze_sentence",  False)
st.session_state.setdefault("use_images",              False)

st.session_state.setdefault("back_section_order", list(BACK_SECTIONS_DEFAULT))
_bso     = st.session_state.back_section_order
_missing = [s for s in BACK_SECTIONS_DEFAULT if s not in _bso]
if _missing:
    st.session_state.back_section_order = _bso + _missing

if st.session_state.light_mode:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

st.title("📚 My Cloud Vocab")

if st.session_state.rpd_count >= 18 and not st.session_state.get("_quota_reset_warned", False):
    st.session_state["_quota_reset_warned"] = True
    st.toast(f"⚠️ Almost out of quota ({st.session_state.rpd_count}/20). "
             f"Resets in **{quota_reset_countdown()}**.", icon="🛑")

if not st.session_state.get("_boot_profiled", False):
    st.session_state["_boot_profiled"] = True
    _t_total = time.perf_counter() - _BOOT_T0
    _t_gh    = _BOOT_T_GH_DONE - _BOOT_T_GH_START
    _t_usage = _BOOT_T_USAGE_DONE - _BOOT_T_GH2
    st.toast(f"⚡ Cold boot: **{_t_total:.1f}s** (GH {_t_gh:.1f}s · usage {_t_usage:.1f}s)", icon="⏱️")


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
        p = normalize_phrase(st.session_state.input_phrase)
        if p and p not in ("1", "*"):
            _p_lower  = p.strip().lower()
            _df       = st.session_state.vocab_df
            _dup_mask = (_df['phrase'].astype(str).str.strip().str.lower() == _p_lower) & \
                        (_df['vocab'].astype(str).str.strip().str.lower() != v)
            if _dup_mask.any():
                _dup_word = _df.loc[_dup_mask, 'vocab'].iloc[0]
                st.warning(f"⚠️ This phrase is already used for **'{_dup_word}'**.")
        mask = st.session_state.vocab_df['vocab'] == v
        if not st.session_state.vocab_df.empty and mask.any():
            st.session_state.vocab_df.loc[mask, ['phrase','status']] = [p, 'New']
        else:
            new_row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "tags": ""}])
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
        save_to_github(st.session_state.vocab_df)
        st.session_state.input_phrase        = ""
        st.session_state.input_vocab         = ""
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
        st.session_state.quick_add_vocab      = ""
        st.session_state.session_words_added += 1
        st.toast(f"✅ Quick-added '{v}'!", icon="⚡")
        st.session_state.pop("_edit_buf_key", None)
        st.session_state.pop("_edit_buffer",  None)
    else:
        st.error("⚠️ Enter a word.")

def render_section_reorder_mobile():
    order  = list(st.session_state.back_section_order)
    hidden = [s for s in BACK_SECTIONS_DEFAULT if s not in order]
    st.caption("Tap ↑ ↓ to reorder · ✖ to hide a section")
    for i, section in enumerate(order):
        meta  = BACK_SECTION_META.get(section)
        label = meta[0] if meta else section
        c_lbl, c_up, c_dn, c_rm = st.columns([5, 1, 1, 1])
        c_lbl.markdown(f"**{label}**")
        if c_up.button("↑", key=f"sup_{i}", use_container_width=True, disabled=(i == 0)):
            order[i], order[i-1] = order[i-1], order[i]
            st.session_state.back_section_order = order; st.rerun()
        if c_dn.button("↓", key=f"sdn_{i}", use_container_width=True, disabled=(i == len(order)-1)):
            order[i], order[i+1] = order[i+1], order[i]
            st.session_state.back_section_order = order; st.rerun()
        if c_rm.button("✖", key=f"srm_{i}", use_container_width=True):
            order.remove(section)
            st.session_state.back_section_order = order; st.rerun()
    if hidden:
        st.caption("Hidden — tap ＋ to restore:")
        for section in hidden:
            meta  = BACK_SECTION_META.get(section)
            label = meta[0] if meta else section
            c_lbl2, c_add = st.columns([5, 1])
            c_lbl2.markdown(f"~~{label}~~")
            if c_add.button("＋", key=f"sadd_{section}", use_container_width=True):
                order.append(section)
                st.session_state.back_section_order = order; st.rerun()


with st.sidebar:
    st.header("⚙️ Settings")
    total_words = len(st.session_state.vocab_df)
    new_words   = len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'])
    col1, col2  = st.columns(2)
    col1.metric("📖 Total", total_words)
    col2.metric("✨ New",   new_words)
    st.metric("🤖 Daily AI Usage", f"{st.session_state.rpd_count}/20 Requests")

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
        st.caption(f"⚠️ {_gh_writes} GH writes — approaching hourly limit (100).")

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
                 ["Indonesian","Spanish","French","German","Japanese","Chinese (Mandarin)","English (Simple)"],
                 index=0, key="target_lang")
    st.selectbox("🤖 AI Model",
                 ["gemini-2.5-flash-lite","gemini-2.0-flash-exp"], index=0, key="gemini_model_name")
    st.selectbox("🧠 Subject Persona", list(PERSONAS.keys()), index=0, key="persona")
    st.radio("📊 Difficulty Level", list(DIFFICULTY_SUFFIX.keys()), index=1, horizontal=True, key="difficulty")
    if st.session_state.difficulty != "Intermediate":
        st.caption(f"📌 Rule 10: _{DIFFICULTY_SUFFIX[st.session_state.difficulty]}_")

    st.checkbox("💡 Generate Memory Hooks (Chain-of-Thought)", key="use_mnemonic")
    if st.session_state.use_mnemonic and not st.session_state.use_lite_mode:
        st.caption("📌 Rule 11 active (Chain-of-Thought).")

    st.checkbox("⚡ Lite Mode Prompt", key="use_lite_mode",
                help="Skips Collocations, Etymology & Mnemonic — saves ~30% tokens per batch.")
    if st.session_state.use_lite_mode:
        st.caption("📌 Lite Mode ON — collocations, etymology & mnemonic will be empty.")

    st.number_input("💰 Session API Budget", min_value=1, max_value=20,
                    value=st.session_state.session_budget, step=1, key="session_budget")
    _sess_used    = max(0, st.session_state.rpd_count - st.session_state.get("session_api_calls_start", 0))
    _sess_left    = max(0, st.session_state.session_budget - _sess_used)
    _budget_color = "🟢" if _sess_left > 3 else ("🟡" if _sess_left > 0 else "🔴")
    st.caption(f"{_budget_color} Session used: **{_sess_used}/{st.session_state.session_budget}** · {_sess_left} left")

    st.selectbox("🌐 Second Language (optional)", LANG2_OPTIONS, index=0, key="target_lang2")
    if st.session_state.target_lang2 != "None (disabled)":
        st.caption(f"📌 Rule 13 active: Translation2 → {st.session_state.target_lang2}")

    st.checkbox("🎴 Cloze Sentence Mode", key="cloze_sentence_mode")
    if st.session_state.cloze_sentence_mode:
        st.caption("📌 Cloze sentence mode ON.")

    st.divider()
    if UNSPLASH_ACCESS_KEY:
        st.checkbox("🖼️ Include Word Images (Unsplash)", key="use_images",
                    help="Fetches one image per vocab word. Only Access Key needed. ~50 calls/hour free.")
        if st.session_state.use_images:
            st.caption("🖼️ Images ON · URLs cached in word cache to avoid re-fetching.")
    else:
        st.caption("🖼️ _Images disabled_ — add `UNSPLASH_ACCESS_KEY` to secrets.toml to enable.")

    st.divider()
    st.selectbox("🎨 Card Theme", options=list(CARD_THEMES.keys()),
                 index=list(CARD_THEMES.keys()).index(st.session_state.get("card_theme","🟢 Cyberpunk")),
                 key="card_theme")
    st.caption(f"_{CARD_THEMES[st.session_state.card_theme]['description']}_")

    with st.expander("🃏 Card Back Section Order", expanded=False):
        render_section_reorder_mobile()

    st.toggle("☀️ Light Mode", key="light_mode")
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
    st.caption("ℹ️ v3.1 adds POSBadge + Image fields. Regenerate Model ID if you have existing cards.")

    if st.button("🗑️ Clear Word Cache"):
        st.session_state.word_cache            = {}
        st.session_state.processed_cache       = {}
        st.session_state.generation_checkpoint = []
        st.session_state.checkpoint_name       = ""
        st.session_state["_gap_cache_key"]     = None
        load_word_cache.clear()
        save_word_cache({})
        st.toast("🗑️ Cache cleared.")

    if not st.session_state.vocab_df.empty:
        st.download_button("💾 Backup Database (CSV)",
                           st.session_state.vocab_df.to_csv(index=False).encode('utf-8'),
                           f"vocab_backup_{date.today()}.csv", "text/csv")

    with st.expander("📜 What's New", expanded=False):
        for version, note in CHANGELOG:
            st.markdown(f"**{version}** — {note}")


tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    done_words  = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'Done']['vocab'].tolist()
    cached_done = [w for w in done_words if w in st.session_state.word_cache]
    if len(cached_done) >= 5:
        _local_rng = random.Random(date.today().toordinal())
        wotd       = _local_rng.choice(cached_done)
        wotd_data  = st.session_state.word_cache.get(wotd, {})
        with st.expander(f"⭐ Word of the Day: **{wotd.title()}**", expanded=False):
            if wotd_data.get("pronunciation_ipa"):    st.caption(f"🗣️ {wotd_data['pronunciation_ipa']}")
            if wotd_data.get("pronunciation_guide"):  st.caption(f"say: _{wotd_data['pronunciation_guide']}_")
            if wotd_data.get("definition_english"):   st.info(f"📜 {wotd_data['definition_english']}")
            if wotd_data.get("mnemonic"):             st.caption(f"💡 {wotd_data['mnemonic']}")

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
    add_mode = st.radio("Mode", ["Single","Bulk"], horizontal=True, label_visibility="collapsed")

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
        uploaded_csv = st.file_uploader("📂 Upload CSV (vocab, phrase, tags columns)", type=["csv"], key="csv_upload")
        if uploaded_csv is not None:
            try:
                up_df   = pd.read_csv(uploaded_csv, dtype=str).fillna("")
                col_map = {}
                for col in up_df.columns:
                    cl = col.strip().lower()
                    if cl in ("vocab","word","term"):           col_map["vocab"]  = col
                    elif cl in ("phrase","sentence","example"): col_map["phrase"] = col
                    elif cl in ("tags","tag"):                  col_map["tags"]   = col
                if "vocab" not in col_map:
                    st.error("❌ No 'vocab' column found. Columns: " + ", ".join(up_df.columns))
                else:
                    csv_rows = []
                    for _, row in up_df.iterrows():
                        bv = str(row.get(col_map["vocab"],"")).strip().lower()
                        bp = normalize_phrase(str(row.get(col_map.get("phrase",""),"")).strip())
                        bt = str(row.get(col_map.get("tags",""),"")).strip()
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

        with st.expander("🔀 Sort & Filter", expanded=False):
            st.radio("Sort by", ["A→Z","Z→A","New first","Done first","No phrase"],
                     horizontal=True, index=0, key="sort_mode")

        search     = st.text_input("🔎 Search...", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search:
            display_df = display_df[display_df['vocab'].str.contains(search, case=False, na=False)]

        sort_mode = st.session_state.get("sort_mode", "A→Z")
        if   sort_mode == "A→Z":      display_df = display_df.sort_values("vocab",  ascending=True)
        elif sort_mode == "Z→A":      display_df = display_df.sort_values("vocab",  ascending=False)
        elif sort_mode == "New first": display_df = display_df.sort_values("status", ascending=True)
        elif sort_mode == "Done first":display_df = display_df.sort_values("status", ascending=False)
        elif sort_mode == "No phrase": display_df = display_df[display_df['phrase'].astype(str).str.strip() == '']

        page_size = 50
        page      = st.number_input("Page", min_value=1, value=1, step=1)
        start     = (page - 1) * page_size
        paginated = display_df.iloc[start:start + page_size]
        _buf_key  = f"_edit_buf_{page}_{search}_{sort_mode}"
        if st.session_state.get("_edit_buf_key") != _buf_key:
            st.session_state["_edit_buf_key"] = _buf_key
            st.session_state["_edit_buffer"]  = paginated.copy()
        edited = st.data_editor(
            st.session_state["_edit_buffer"],
            num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={"status": st.column_config.SelectboxColumn(
                "Status", options=["New","Done"], required=True)}
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
            st.session_state.pop("_edit_buffer",  None)
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
                pct = lambda n: f"{n/total*100:.0f}%"
                st.dataframe(pd.DataFrame({
                    "Metric": ["Total","With phrases","With tags","Duplicates","Short (≤2)","New","Done"],
                    "Count":  [total, with_phrase, with_tags, dups, short_vocab,
                               (df['status']=='New').sum(), (df['status']=='Done').sum()],
                    "%":      ["100%",pct(with_phrase),pct(with_tags),pct(dups),pct(short_vocab),
                               pct((df['status']=='New').sum()),pct((df['status']=='Done').sum())],
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

with tab3:
    @st.fragment
    def render_tab3():
        checkpoint = st.session_state.get("generation_checkpoint", [])
        ckpt_name  = st.session_state.get("checkpoint_name", f"{len(checkpoint)} cards")
        if checkpoint and st.session_state.editing_notes is None and st.session_state.apkg_buffer is None:
            with st.expander(f"⏸️ Resume partial generation? ({ckpt_name})", expanded=True):
                st.info("A previous generation was interrupted. You can resume from the saved checkpoint.")
                col_resume, col_discard = st.columns(2)
                if col_resume.button("▶️ Resume (use saved cards)", type="primary"):
                    st.session_state.editing_notes     = checkpoint
                    st.session_state.editing_deck_name = st.session_state.last_deck_name
                    st.session_state.editing_audio     = True
                    st.rerun(scope="app")
                if col_discard.button("🗑️ Discard checkpoint"):
                    st.session_state.generation_checkpoint = []
                    st.session_state.checkpoint_name       = ""
                    st.rerun(scope="app")

        if st.session_state.editing_notes is not None:
            st.subheader("✏️ Edit Generated Cards")
            st.caption("Review and fix AI output. All changes reflected in the final .apkg.")
            EDITABLE_COLS = ["VocabRaw","Definition","Collocations",
                             "RegisterLabel","Synonyms","Antonyms","Mnemonic","Hint"]
            notes_df = pd.DataFrame([
                {**{col: n.get(col,"") for col in EDITABLE_COLS},
                 "Freq": word_frequency_label(n.get("VocabRaw","")),
                 "Q":    f"{quality_badge(n.get('_quality_score',0))} {n.get('_quality_score',0)}"}
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
                    "Hint":          st.column_config.TextColumn("Hint",         width="small", disabled=True),
                }
            )
            scores = [n.get("_quality_score",0) for n in st.session_state.editing_notes]
            avg_q  = int(sum(scores)/len(scores)) if scores else 0
            low_q  = sum(1 for s in scores if s < QUALITY_WARN_THRESHOLD)
            st.info(f"📊 **{len(scores)} cards** · Avg quality: {quality_badge(avg_q)} **{avg_q}/100**"
                    + (f" · ⚠️ {low_q} card(s) below {QUALITY_WARN_THRESHOLD}" if low_q else ""))

            _needs_review = [n['VocabRaw'] for n in st.session_state.editing_notes
                             if 'needs_review' in n.get('Tags', [])]
            if _needs_review:
                st.warning(f"🏷️ **{len(_needs_review)}** card(s) auto-tagged `needs_review`: "
                           f"`{', '.join(_needs_review[:8])}{'...' if len(_needs_review) > 8 else ''}`")

            _antonym_gaps = [n['VocabRaw'] for n in st.session_state.editing_notes
                             if (len([s for s in n.get('Synonyms','').split(',') if s.strip()]) >= 3
                                 and not n.get('Antonyms','').strip())]
            if _antonym_gaps:
                st.warning(f"⚠️ **{len(_antonym_gaps)} card(s)** have ≥3 synonyms but no antonyms: "
                           f"`{', '.join(_antonym_gaps[:5])}{'...' if len(_antonym_gaps) > 5 else ''}`")

            _pos_counts: dict[str, int] = {}
            for n in st.session_state.editing_notes:
                _raw = _RE_STRIP_HTML.sub('', n.get('POSBadge','')).strip()
                if _raw: _pos_counts[_raw] = _pos_counts.get(_raw, 0) + 1
            if _pos_counts:
                _pos_summary = " · ".join(f"{v} {k}" for k, v in sorted(_pos_counts.items(), key=lambda x: -x[1]))
                st.caption(f"🎨 POS breakdown: {_pos_summary}")

            csv_export_cols = ["VocabRaw","Definition","Collocations","RegisterLabel",
                               "Synonyms","Antonyms","Mnemonic","Etymology","Hint"]
            st.download_button(
                "💾 Export card data as CSV",
                pd.DataFrame([{c: n.get(c,'') for c in csv_export_cols}
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
                        note["Mnemonic"]      = str(row.get("Mnemonic",      note.get("Mnemonic","")))
                        new_reg               = str(row.get("RegisterLabel", note.get("RegisterLabel","Neutral")))
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
                        sentence_audio=st.session_state.get("editing_sentence_audio", False),
                        slow_audio=st.session_state.get("editing_slow_audio", False),
                        generate_images=st.session_state.get("editing_images", False),
                    )
                st.session_state.apkg_buffer              = buffer.getvalue()
                st.session_state.processed_vocabs         = [n['VocabRaw'] for n in updated_notes]
                st.session_state.preview_notes            = updated_notes[:3]
                st.session_state.editing_notes            = None
                st.session_state.deck_stats               = deck_stats
                st.session_state.generation_checkpoint    = []
                st.session_state.checkpoint_name          = ""
                st.session_state.session_cards_generated += len(updated_notes)
                st.rerun(scope="app")
            if col_cancel.button("❌ Discard & Start Over", use_container_width=True):
                st.session_state.editing_notes = None
                st.rerun(scope="app")
            return

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
                sc3.metric("With Examples",  f"{fc.get('Examples',0)}/{tc}")
                sc4.metric("With Mnemonics", f"{fc.get('Mnemonic',0)}/{tc}")
                if ds.get("images_embedded", 0) > 0:
                    st.caption(f"🖼️ {ds['images_embedded']} image(s) embedded from Unsplash.")
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
                        if note.get('POSBadge'):
                            pos_plain = _RE_STRIP_HTML.sub('', note['POSBadge']).strip()
                            st.markdown(f"🎨 {pos_plain}", unsafe_allow_html=False)
                        if show_styled:
                            _preview_css = get_active_css()
                            _preview_fg  = CARD_THEMES[st.session_state.get("card_theme","🟢 Cyberpunk")]["front_color"]
                            st.markdown(
                                f"<style>{_preview_css}</style>"
                                f"<div class='card vellum-focus-container'>"
                                f"<div class='prompt-text' style='color:{_preview_fg}'>{front_preview}</div></div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(f"<div style='background:#1a1a1a;border:1px solid #00ff41;"
                                        f"padding:10px 14px;border-radius:4px;font-family:monospace;"
                                        f"color:#aaffaa;line-height:1.6'>{plain_front}</div>",
                                        unsafe_allow_html=True)
                        st.markdown(f"**Card {i} — BACK**")
                        back_items = []
                        if note.get("Pronunciation"):  back_items.append(f"🗣️ {note['Pronunciation']}")
                        if note.get("Definition"):     back_items.append(f"📜 {note['Definition']}")
                        if note.get("Hint"):           back_items.append(f"💭 Hint: {note['Hint']}")
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
                        st.markdown("<div style='background:#1a1a1a;border:1px solid #00ffff;"
                                    "padding:10px 14px;border-radius:4px;font-family:monospace;"
                                    f"color:#aaffaa;line-height:1.8'>{'<br>'.join(back_items)}</div>",
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

        if st.session_state.vocab_df.empty:
            st.info("Add words first!"); return
        subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
        if subset.empty:
            st.warning("⚠️ No 'New' words to export! All words are marked 'Done'."); return

        _pos_in_cache = sorted({
            str(st.session_state.word_cache.get(v,{}).get('part_of_speech','') or '').title()
            for v in subset['vocab'].str.strip().str.lower()
            if str(st.session_state.word_cache.get(v,{}).get('part_of_speech','')).strip()
        })
        if len(_pos_in_cache) >= 2:
            _pos_filter = st.multiselect("🔠 Filter by Part of Speech (optional)", options=_pos_in_cache, default=[])
            if _pos_filter:
                _matching = {v for v in subset['vocab'].str.strip().str.lower()
                             if str(st.session_state.word_cache.get(v,{}).get('part_of_speech','') or '').title()
                             in _pos_filter}
                subset = subset[subset['vocab'].str.strip().str.lower().isin(_matching)]
                if subset.empty:
                    st.warning("⚠️ No New words match the selected POS filter."); return
                st.caption(f"🔠 Showing **{len(subset)}** word(s) matching: {', '.join(_pos_filter)}")

        if st.session_state.failed_words:
            with st.expander(f"⚠️ {len(st.session_state.failed_words)} word(s) failed — click to retry", expanded=True):
                st.dataframe(pd.DataFrame({"Queued for Retry": st.session_state.failed_words}), hide_index=True)
                retry_delay = st.select_slider("⏳ Wait before retry", options=[0,30,60,120],
                                               value=30, format_func=lambda x: "Immediate" if x==0 else f"{x}s delay")
                col_retry, col_dismiss = st.columns(2)
                if col_retry.button("🔁 Retry Failed Words", type="primary"):
                    if retry_delay > 0:
                        _slot = st.empty()
                        for r in range(retry_delay, 0, -1):
                            _slot.warning(f"⏳ Waiting {r}s before retry..."); time.sleep(1)
                        _slot.empty()
                    retry_df = pd.DataFrame({
                        "vocab":  st.session_state.failed_words,
                        "phrase": [""]*len(st.session_state.failed_words),
                        "status": ["New"]*len(st.session_state.failed_words),
                        "tags":   [""]*len(st.session_state.failed_words),
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

        deck_col1, deck_col2 = st.columns([3, 1])
        deck_name_raw   = deck_col1.text_input("📦 Deck Name (use :: for sub-decks)",
                                               value=st.session_state.last_deck_name)
        deck_parts_raw  = [p.strip() for p in deck_name_raw.split("::") if p.strip()]
        deck_parts      = [_RE_DECK_ILLEGAL.sub("", p) for p in deck_parts_raw]
        deck_name_input = "::".join(deck_parts) if deck_parts else "Vocabulary"
        if deck_name_raw: st.session_state.last_deck_name = deck_name_input
        if _RE_DECK_ILLEGAL.search(deck_name_raw.replace("::","")):
            st.warning("⚠️ Illegal characters removed from deck name.")
        if len(deck_parts) > 1:
            st.caption("📂 Hierarchy: " + " → ".join(deck_parts))
        if deck_col2.button("🎲 New Deck ID"):
            st.session_state.deck_id = random.randrange(1 << 30, 1 << 31)
        deck_col2.caption(f"ID: {st.session_state.deck_id}")

        requests_left = max(0, 20 - st.session_state.rpd_count)
        _n_words      = len(subset)
        _r_left       = max(1, requests_left)
        _recommended  = min(15, max(1, math.ceil(_n_words / max(1, int(_r_left * 0.6)))))
        raw_batch     = st.slider("⚡ Batch Size (Words per Request)", 1, 15, _recommended)
        if raw_batch == _recommended:
            st.caption(f"✅ Using recommended batch size **{_recommended}** "
                       f"({math.ceil(_n_words/_recommended)} request(s) needed).")
        else:
            st.caption(f"⚙️ Custom batch size: **{raw_batch}** (recommended was {_recommended}).")
        max_safe   = max(1, math.ceil(_n_words/max(1,requests_left))) if requests_left > 0 else 1
        batch_size = min(raw_batch, max_safe)
        st.session_state.last_batch_size = batch_size
        if batch_size != raw_batch:
            st.caption(f"⚠️ Capped to **{batch_size}** by quota limit.")

        include_audio          = st.checkbox("🔊 Generate Audio Files",                     value=True)
        include_slow_audio     = st.checkbox("🐢 Also slow pronunciation audio",             value=False)
        include_sentence_audio = st.checkbox("🔊 Also audio for example sentences",          value=False)
        include_reversed       = st.checkbox("🔄 Include Reversed Cards (Translation→Word)", value=False)
        include_images         = st.checkbox("🖼️ Embed Unsplash images in deck",             value=False,
                                             disabled=not (UNSPLASH_ACCESS_KEY and st.session_state.use_images))
        st.session_state.include_antonyms = st.checkbox("➖ Include Antonyms in Card Back",
                                                         value=st.session_state.include_antonyms)
        st.session_state.dry_run = st.checkbox("🔬 Dry Run Mode (simulate, no quota)", value=st.session_state.dry_run)

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
                "Export":              st.column_config.CheckboxColumn("Export?",        required=True),
                "⚠️ Prev. Exported":  st.column_config.CheckboxColumn("Prev. Exported?", disabled=True),
            },
            hide_index=True,
            disabled=["vocab","phrase","status","tags","⚠️ Prev. Exported"]
        )
        final_export = edited_export[edited_export['Export'].astype(bool)]
        dup_count    = int(final_export['⚠️ Prev. Exported'].astype(bool).sum()) \
            if '⚠️ Prev. Exported' in final_export.columns else 0
        if dup_count > 0:
            st.warning(f"⚠️ **{dup_count}** selected word(s) were previously exported this session.")

        if not final_export.empty:
            st.write("### Export Preview")
            st.dataframe(final_export[['vocab','phrase']], hide_index=True)
            card_count  = len(final_export)
            per_card_kb = 35.0 if include_audio else 2.5
            if include_reversed:       per_card_kb *= 1.15
            if include_sentence_audio: per_card_kb *= 1.8
            if include_slow_audio:     per_card_kb *= 1.6
            if include_images:         per_card_kb *= 1.4
            est_size_kb = card_count * per_card_kb
            size_label  = f"{est_size_kb/1024:.2f} MB" if est_size_kb > 1024 else f"{est_size_kb:.1f} KB"
            extras = "".join([
                " + reversed"        if include_reversed       else "",
                " + sentence audio"  if include_sentence_audio else "",
                " + slow audio"      if include_slow_audio     else "",
                " + images"          if include_images         else "",
            ])
            st.info(f"📊 **{card_count} cards** • Est. .apkg size: **{size_label}**{extras}")

        if not final_export.empty:
            render_quota_forecast(len(final_export), batch_size, requests_left)

        if not final_export.empty and not st.session_state.dry_run:
            avg_chars  = (sum(len(str(r['vocab']))+len(str(r['phrase']))
                              for _, r in final_export.iterrows()) / max(len(final_export),1))
            est_tpm    = int(len(final_export) * (avg_chars + 650) / 4)
            tpm_remain = 1_000_000 - get_rolling_tpm()
            if est_tpm > tpm_remain * 0.6:
                st.warning(f"⚠️ This generation may use ~{est_tpm:,} tokens "
                           f"({est_tpm/1_000_000*100:.0f}% of remaining hourly TPM budget).")

        quota_key = (st.session_state.rpd_count, len(final_export), batch_size)
        if st.session_state._quota_cache_key != quota_key:
            r_left = max(0, 20 - st.session_state.rpd_count)
            r_req  = math.ceil(len(final_export)/batch_size) if not final_export.empty else 0
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
                        st.session_state.editing_sentence_audio = include_sentence_audio
                        st.session_state.editing_slow_audio     = include_slow_audio
                        st.session_state.editing_images         = include_images
                        st.rerun(scope="app")
                except Exception as e:
                    st.error(f"❌ Generation error: {e} — Status rolled back to 'New'.")
                    if raw_notes:
                        failed = [n.get('VocabRaw','') for n in raw_notes]
                        st.session_state.vocab_df.loc[
                            st.session_state.vocab_df['vocab'].isin(failed), 'status'] = 'New'
                        save_to_github(st.session_state.vocab_df)

        st.divider()
        with st.expander("📊 Session Summary", expanded=False):
            _api_this = max(0, st.session_state.rpd_count - st.session_state.get("session_api_calls_start",0))
            _q_remain = max(0, 20 - st.session_state.rpd_count)
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Words Added",     st.session_state.session_words_added)
            sc2.metric("Cards Generated", st.session_state.session_cards_generated)
            sc3.metric("API Calls Used",  _api_this)
            sc4.metric("Quota Remaining", f"{_q_remain}/20")
            st.caption(f"📡 GitHub writes: **{gh_write_count()}** "
                       f"{'⚠️ near limit' if gh_write_count() >= GH_WRITE_WARN_THRESHOLD else '✅ healthy'} · "
                       f"⏰ Quota resets in **{quota_reset_countdown()}**")

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
                    ts = entry.get("timestamp","")
                    try: ts = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
                    except: pass
                    rows.append({"Timestamp": ts, "Deck": entry.get("deck_name",""),
                                 "Cards": entry.get("card_count",0),
                                 "Vocab": ", ".join(entry.get("vocabs",[])[:10])
                                          + ("…" if len(entry.get("vocabs",[])) > 10 else "")})
                hist_df = pd.DataFrame(rows)
                h_page  = st.number_input("History page", min_value=1,
                                          max_value=max(1, math.ceil(len(hist_df)/10)),
                                          value=1, step=1, key="hist_page")
                h_start = (h_page-1)*10
                st.dataframe(hist_df.iloc[h_start:h_start+10], hide_index=True, use_container_width=True)
                st.caption(f"Showing {min(h_start+10, len(hist_df))} of {len(hist_df)} records.")

    render_tab3()
