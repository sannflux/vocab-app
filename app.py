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
import math

try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

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

# ========================== SECRETS ==========================
try:
    token             = st.secrets["GITHUB_TOKEN"]
    repo_name         = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

# ========================== B8: MODULE-LEVEL PRE-COMPILED REGEX ==========================
# All patterns compiled ONCE at import time — zero recompilation cost per call.
_RE_SPACES         = re.compile(r"\s+")
_RE_SENT_SPLIT     = re.compile(r'(?<=[.!?])\s+')
_RE_JSON_FENCE_S   = re.compile(r"^```(?:json)?\s*")
_RE_JSON_FENCE_E   = re.compile(r"\s*```$")
_RE_JSON_ARRAY     = re.compile(r'\[.*\]', re.DOTALL)
_RE_CLEAN_TEXT     = re.compile(r'[^\w\s\-\']')
_RE_CLEAN_FNAME    = re.compile(r'[^a-zA-Z0-9]')
_RE_AUDIO_CLEAN    = re.compile(r'[^a-zA-Z0-9\s\-\']')

# Grammar rules: list of (compiled_pattern, replacement) — built once.
_GRAMMAR_RULES = [
    (re.compile(r"\bto doing\b",     re.IGNORECASE), "to do"),
    (re.compile(r"\bfor helps\b",    re.IGNORECASE), "to help"),
    (re.compile(r"\bis use to\b",    re.IGNORECASE), "is used to"),
    (re.compile(r"\bhelp for to\b",  re.IGNORECASE), "help to"),
    (re.compile(r"\bfor to\b",       re.IGNORECASE), "to"),
    (re.compile(r"\bcan able to\b",  re.IGNORECASE), "can"),
    (re.compile(r"\bI am agree\b",   re.IGNORECASE), "I agree"),
    (re.compile(r"\bdiscuss about\b",re.IGNORECASE), "discuss"),
    (re.compile(r"\breturn back\b",  re.IGNORECASE), "return"),
]

# ========================== C12: BACKGROUND GITHUB EXECUTOR ==========================
# @st.cache_resource ensures ONE executor survives across all Streamlit reruns.
# GitHub writes are submitted as fire-and-forget tasks — never block the hot path.
@st.cache_resource
def _get_gh_executor():
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="gh_bg"
    )

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
def load_usage() -> int:
    try:
        file = repo.get_contents("usage.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        if data.get("date") == str(date.today()):
            return data.get("rpd_count", 0)
        return 0
    except:
        return 0

def _bg_save_usage(count: int):
    """B7: Runs in background thread. Never called on the main thread."""
    data = json.dumps({"date": str(date.today()), "rpd_count": count})
    try:
        file = repo.get_contents("usage.json")
        repo.update_file(file.path, "Update API usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage.json", "Init API usage", data)

def save_usage(count: int):
    """B7: Non-blocking — submits to background executor instantly."""
    _get_gh_executor().submit(_bg_save_usage, count)

def load_minute_usage() -> list:
    try:
        file = repo.get_contents("usage_minute.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return [datetime.fromisoformat(ts) for ts in data.get("timestamps", [])]
    except:
        return []

def _bg_save_minute_usage(timestamps: list):
    """B6: Runs in background thread."""
    ts_list = [ts.isoformat() for ts in timestamps]
    data    = json.dumps({"timestamps": ts_list})
    try:
        file = repo.get_contents("usage_minute.json")
        repo.update_file(file.path, "Update minute usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage_minute.json", "Init minute usage", data)

def save_minute_usage(timestamps: list):
    """B6: Non-blocking — submits to background executor instantly."""
    _get_gh_executor().submit(_bg_save_minute_usage, list(timestamps))

# ========================== A5 + C11: SMART RPM ENFORCEMENT ==========================
def enforce_rpm() -> float:
    """
    A5: Single st.empty() slot updated in-place — no accumulating st.warning() elements.
    C11: Returns elapsed wait time (seconds) for C15 profiling.
    B6: save_minute_usage fires non-blocking after enforcement.
    """
    t0  = time.perf_counter()
    now = datetime.now()

    # Prune timestamps older than 60s
    st.session_state.rpm_timestamps = [
        ts for ts in st.session_state.rpm_timestamps
        if (now - ts).total_seconds() < 60
    ]

    if len(st.session_state.rpm_timestamps) >= 5:
        sleep_total = 12
        _slot = st.empty()   # A5: one slot, updated in-place — zero DOM accumulation
        for remaining in range(sleep_total, 0, -1):
            _slot.warning(f"⏳ RPM limit (5/min). Resuming in **{remaining}s**...")
            time.sleep(1)
        _slot.empty()

    st.session_state.rpm_timestamps.append(now)
    save_minute_usage(st.session_state.rpm_timestamps)   # B6: non-blocking

    return time.perf_counter() - t0   # C15: timing

# ========================== GEMINI (🔒 model stasis preserved) ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
    except Exception as e:
        st.error(f"❌ Gemini key error: {e}")
        return None

# ========================== B8: CLEANING FUNCTIONS (pre-compiled patterns) ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if s and s[-1] in ".!?" else (s + "." if s else "")

def normalize_spaces(text: str) -> str:
    return _RE_SPACES.sub(" ", str(text)).strip() if text else ""

def clean_grammar(text: str) -> str:
    if not isinstance(text, str): return text
    for pat, repl in _GRAMMAR_RULES:   # B8: pre-compiled — no re.compile on every call
        text = pat.sub(repl, text)
    return text

def cap_each_sentence(text: str) -> str:
    if not isinstance(text, str): return text
    return " ".join(
        cap_first(s) for s in _RE_SENT_SPLIT.split(text) if s.strip()   # B8
    )

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    pat = re.compile(r'\b' + re.escape(vocab) + r'\b', re.IGNORECASE)
    return pat.sub(f'<b><u>{vocab}</u></b>', text)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    pat = re.compile(r'\b' + re.escape(vocab.lower()) + r'\b', re.IGNORECASE)
    return pat.sub(vocab, phrase)

def robust_json_parse(text: str):
    text = _RE_JSON_FENCE_S.sub("", text.strip())   # B8
    text = _RE_JSON_FENCE_E.sub("", text)
    try: return json.loads(text)
    except: pass
    match = _RE_JSON_ARRAY.search(text)             # B8
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe = text.replace('"', '\\"').replace("'", "\\'")
    st.components.v1.html(
        f"""<script>if('speechSynthesis'in window){{
        var u=new SpeechSynthesisUtterance("{safe}");
        u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>""",
        height=0
    )

# ========================== B10: SINGLE-PASS FIELD CLEANER ==========================
def _clean_field(text: str) -> str:
    """B10: Chains normalize → grammar → cap_sentences → trailing_dot in one call."""
    return ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(text))))

# ========================== C13 + C14: BATCH GENERATOR ==========================
def generate_anki_card_data_batched(
    vocab_phrase_list: list,
    batch_size: int = 6,
    dry_run: bool   = False
) -> list:
    # A1: Read model settings from session_state — safe inside fragments
    TARGET_LANG = st.session_state.get("target_lang", "Indonesian")
    model_name  = st.session_state.get("gemini_model_name", "gemini-2.5-flash-lite")
    model       = get_gemini_model(st.session_state.gemini_key, model_name)
    if not model: return []

    # C14: Per-word deduplication — skip anything already in the session word cache
    word_cache     = st.session_state.get("word_cache", {})
    cached_results = [
        word_cache[vp[0].strip().lower()]
        for vp in vocab_phrase_list
        if vp[0].strip().lower() in word_cache
    ]
    deduped_list = [
        vp for vp in vocab_phrase_list
        if vp[0].strip().lower() not in word_cache
    ]

    if cached_results:
        st.info(f"♻️ {len(cached_results)} word(s) served from cache — **zero quota used**.")

    if not deduped_list:
        return cached_results   # C14: entire batch was cached, no API call needed

    all_new_data = []
    batches      = [deduped_list[i:i + batch_size] for i in range(0, len(deduped_list), batch_size)]
    timings      = []   # C15: per-batch profiling

    with st.status("🤖 Processing AI Batches (RPM Throttled)...", expanded=True) as status_log:
        progress_bar = st.progress(0)

        for idx, batch in enumerate(batches):
            if st.session_state.rpd_count >= 20:
                st.warning("🛑 Daily AI Limit (20 requests) reached. Try again tomorrow.")
                break

            # A5 + C11: non-multiplying countdown, returns elapsed for C15
            t_rpm = enforce_rpm()

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            vocab_words = [v[0] for v in batch]

            # C13: Compressed to 2 most-distinct few-shots (~300 fewer tokens vs original 4)
            prompt = f"""You are an expert educational lexicographer. Think step-by-step:
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

EXAMPLES:
[
  {{"vocab":"serendipity","phrase":"We found the perfect cafe by pure serendipity.","translation":"kebetulan","part_of_speech":"Noun","pronunciation_ipa":"/ˌsɛrənˈdɪpɪti/","definition_english":"The occurrence and development of events by chance in a happy or beneficial way.","example_sentences":["It was pure serendipity that we met."],"synonyms_antonyms":{{"synonyms":["chance","luck"],"antonyms":["misfortune"]}},"etymology":"Coined by Horace Walpole in 1754."}},
  {{"vocab":"run","phrase":"*He decided to run for office","translation":"mencalonkan diri","part_of_speech":"Verb","pronunciation_ipa":"/rʌn/","definition_english":"To compete in an election.","example_sentences":["She will run for president."],"synonyms_antonyms":{{"synonyms":["campaign"],"antonyms":["withdraw"]}},"etymology":"Old English rinnan."}}
]

BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

            success     = False
            t_api_start = time.perf_counter()   # C15

            if dry_run:
                st.info(f"🔬 Dry-run: `{', '.join(vocab_words)}`")
                mock = [
                    {
                        "vocab": v[0], "phrase": v[1],
                        "translation": "mock-" + v[0], "part_of_speech": "Noun",
                        "pronunciation_ipa": "/mock/",
                        "definition_english": "Simulated definition.",
                        "example_sentences": ["Mock example sentence."],
                        "synonyms_antonyms": {"synonyms": ["mock"], "antonyms": []},
                        "etymology": "Simulated."
                    }
                    for v in batch
                ]
                all_new_data.extend(mock)
                for card in mock:
                    word_cache[card['vocab'].strip().lower()] = card   # C14
                success = True
            else:
                for attempt in range(3):
                    try:
                        response = model.generate_content(prompt)
                        st.session_state.rpd_count += 1
                        save_usage(st.session_state.rpd_count)   # B7: non-blocking

                        parsed = robust_json_parse(response.text)
                        if isinstance(parsed, list) and len(parsed) == len(batch_dicts):
                            all_new_data.extend(parsed)
                            for card in parsed:
                                word_cache[card['vocab'].strip().lower()] = card   # C14
                            st.markdown(f"✅ **Batch {idx + 1}**: `{', '.join(vocab_words)}`")
                            success = True
                            break
                    except Exception as e:
                        if "429" in str(e):
                            backoff = 20 + (2 ** attempt) + random.uniform(0, 1)
                            _slot = st.empty()   # C11: single in-place slot for 429 backoff
                            for r in range(int(backoff), 0, -1):
                                _slot.warning(
                                    f"⚠️ 429 Rate Limit. Retrying in **{r}s**... "
                                    f"(attempt {attempt + 1}/3)"
                                )
                                time.sleep(1)
                            _slot.empty()
                        else:
                            time.sleep(2)

            # C15: Record per-batch timing
            t_api_elapsed = time.perf_counter() - t_api_start
            timings.append({
                "batch":       idx + 1,
                "words":       ", ".join(vocab_words),
                "rpm_wait_s":  round(t_rpm, 3),
                "gemini_s":    round(t_api_elapsed, 3),
                "cached":      False,
            })

            if not success and not dry_run:
                st.error(f"❌ **Failed**: `{', '.join(vocab_words)}` — skipping to preserve quota")

            progress_bar.progress((idx + 1) / len(batches))

        total = len(all_new_data) + len(cached_results)
        status_log.update(
            label=f"✅ AI Complete! ({total} items | {len(cached_results)} cached)",
            state="complete", expanded=False
        )

    # Persist updated word cache
    st.session_state.word_cache = word_cache

    # C15: Timing report in collapsible expander
    if timings:
        with st.expander("⏱️ Batch Performance Timings", expanded=False):
            st.dataframe(pd.DataFrame(timings), hide_index=True)

    return cached_results + all_new_data

# ========================== B9 + B10: PROCESS ANKI DATA ==========================
def process_anki_data(
    df_subset: pd.DataFrame,
    batch_size: int  = 6,
    dry_run: bool    = False
) -> list:
    t0 = time.perf_counter()   # C15

    # B9: Vectorized hash — ~50× faster than sha256(str(df.to_dict()))
    cache_key = str(pd.util.hash_pandas_object(df_subset).sum())
    cached    = st.session_state.get("processed_cache", {})
    if (cached.get("key") == cache_key
            and (datetime.now() - cached.get("time", datetime.min)).total_seconds() < 300):
        st.info("♻️ Using cached processed notes — no re-generation needed.")
        return cached["notes"]

    df_clean          = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_clean[['vocab', 'phrase']].values.tolist()
    all_card_data     = generate_anki_card_data_batched(
        vocab_phrase_list, batch_size=batch_size, dry_run=dry_run
    )

    processed_notes = []
    for card_data in all_card_data:
        required = ["vocab", "translation", "part_of_speech"]
        if not all(k in card_data and card_data[k] for k in required):
            st.error(f"⚠️ Missing required fields for `{card_data.get('vocab','?')}` — skipping")
            continue

        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue
        vocab_cap = cap_first(vocab_raw)

        # B10: _clean_field() single-pass pipeline for every text field
        phrase      = fix_vocab_casing(_clean_field(card_data.get("phrase", "")), vocab_raw)
        formatted   = highlight_vocab(phrase, vocab_raw) if phrase else ""
        translation = _clean_field(card_data.get("translation", "?"))
        pos         = str(card_data.get("part_of_speech", "")).title()
        ipa         = card_data.get("pronunciation_ipa", "")
        eng_def     = _clean_field(card_data.get("definition_english", ""))
        examples    = [
            _clean_field(e)
            for e in (card_data.get("example_sentences") or [])[:3]
        ]
        ex_field    = (
            "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>"
            if examples else ""
        )
        syn_ant     = card_data.get("synonyms_antonyms") or {}
        synonyms    = ensure_trailing_dot(", ".join(
            cap_first(s) for s in (syn_ant.get("synonyms") or [])[:5]
        ))
        antonyms    = ensure_trailing_dot(", ".join(
            cap_first(a) for a in (syn_ant.get("antonyms") or [])[:5]
        ))
        etymology   = normalize_spaces(card_data.get("etymology", ""))

        text_field = (
            f"{formatted}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
            if formatted else
            f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        )
        pron_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"

        processed_notes.append({
            "VocabRaw":     vocab_raw,
            "Text":         text_field,
            "Pronunciation":pron_field,
            "Definition":   eng_def,
            "Examples":     ex_field,
            "Synonyms":     synonyms,
            "Antonyms":     antonyms,
            "Etymology":    etymology,
            "Tags":         [],
        })

    st.session_state.processed_cache = {
        "key": cache_key, "notes": processed_notes, "time": datetime.now()
    }
    st.caption(f"⏱️ `process_anki_data`: {time.perf_counter() - t0:.3f}s — {len(processed_notes)} notes")  # C15
    return processed_notes

# ========================== D17: AUDIO HELPER ==========================
def generate_audio_file(args: tuple):
    """
    D17: Accepts (vocab, temp_dir) tuple for clean executor.map usage.
    Regex pre-computation happens in the caller before the executor loop.
    """
    vocab, temp_dir = args
    try:
        clean_vocab  = _RE_AUDIO_CLEAN.sub('', vocab).strip()   # B8
        clean_fname  = _RE_CLEAN_FNAME.sub('', clean_vocab) + ".mp3"   # B8
        file_path    = os.path.join(temp_dir, clean_fname)
        if clean_vocab:
            gTTS(text=clean_vocab, lang='en', slow=False).save(file_path)
            return vocab, clean_fname, file_path
    except Exception as e:
        print(f"Audio error for {vocab}: {e}")
    return vocab, None, None

# ========================== A4: CYBERPUNK CSS (module-level constant — evaluated once) ==========================
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
@media (max-width: 480px) {{
    .card {{ font-size: 16px; padding: 15px; }}
    .vellum-focus-container {{ padding: 15px; }}
}}
"""

# ========================== GENANKI LOGIC ==========================
def create_anki_package(
    notes_data:      list,
    deck_name:       str,
    generate_audio:  bool = True,
    deck_id:         int  = 2059400110,
    include_antonyms:bool = True
) -> io.BytesIO:
    t0 = time.perf_counter()   # C15

    front_html = """<div class="vellum-focus-container front">
<div class="prompt-text">{{cloze:Text}}</div></div>"""

    back_html = """<div class="vellum-focus-container back">
<div class="prompt-text solved-text">{{cloze:Text}}</div></div>
<div class="vellum-detail-container">
{{#Definition}}<div class="vellum-section">
<div class="section-header">📜 DEFINITION</div>
<div class="content">{{Definition}}</div></div>{{/Definition}}
{{#Pronunciation}}<div class="vellum-section">
<div class="section-header">🗣️ PRONUNCIATION</div>
<div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}
{{#Examples}}<div class="vellum-section">
<div class="section-header">🖋️ EXAMPLES</div>
<div class="content">{{Examples}}</div></div>{{/Examples}}
{{#Synonyms}}<div class="vellum-section">
<div class="section-header">➕ SYNONYMS</div>
<div class="content">{{Synonyms}}</div></div>{{/Synonyms}}"""

    if include_antonyms:
        back_html += """{{#Antonyms}}<div class="vellum-section">
<div class="section-header">➖ ANTONYMS</div>
<div class="content">{{Antonyms}}</div></div>{{/Antonyms}}"""

    back_html += """{{#Etymology}}<div class="vellum-section">
<div class="section-header">🏛️ ETYMOLOGY</div>
<div class="content">{{Etymology}}</div></div>{{/Etymology}}
<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""

    model_id = st.session_state.get("model_id", 1607392319)
    my_model = genanki.Model(
        model_id, 'Cyberpunk Vocab Model',
        fields=[
            {'name': 'Text'},        {'name': 'Pronunciation'},
            {'name': 'Definition'},  {'name': 'Examples'},
            {'name': 'Synonyms'},    {'name': 'Antonyms'},
            {'name': 'Etymology'},   {'name': 'Audio'},
        ],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=CYBERPUNK_CSS,   # A4: module-level constant — no re-evaluation
        model_type=genanki.Model.CLOZE
    )
    my_deck    = genanki.Deck(deck_id, deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}

        if generate_audio:
            t_audio       = time.perf_counter()
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}

            # D17: Build (vocab, temp_dir) tuples ONCE before the executor loop.
            # All filename regex runs inside generate_audio_file using pre-compiled patterns.
            args_list = [(v, temp_dir) for v in unique_vocabs]

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exc:
                for vk, fn, fp in exc.map(generate_audio_file, args_list):
                    if fn:
                        media_files.append(fp)
                        audio_map[vk] = f"[sound:{fn}]"

            st.caption(   # C15
                f"⏱️ Audio: {time.perf_counter() - t_audio:.2f}s "
                f"for {len(unique_vocabs)} words"
            )

        for note_data in notes_data:
            guid_input = note_data['VocabRaw'] + deck_name
            vocab_hash = str(
                int(hashlib.sha256(guid_input.encode('utf-8')).hexdigest(), 16) % (10 ** 10)
            )
            my_deck.add_note(genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'],        note_data['Pronunciation'],
                    note_data['Definition'],  note_data['Examples'],
                    note_data['Synonyms'],    note_data['Antonyms'],
                    note_data['Etymology'],   audio_map.get(note_data['VocabRaw'], ""),
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

    st.caption(f"⏱️ `create_anki_package` total: {time.perf_counter() - t0:.2f}s")   # C15
    return buffer

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data() -> pd.DataFrame:
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(
            io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str
        )
        df['phrase'] = df['phrase'].fillna("")
        df['status'] = df.get('status', 'New')
        df['tags']   = df.get('tags', '')
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags'])
        st.stop()
    except:
        st.stop()

def save_to_github(dataframe: pd.DataFrame) -> bool:
    t0 = time.perf_counter()   # C15

    # D16: Single vectorized boolean mask — replaces multi-step chain
    mask      = dataframe['vocab'].astype(str).str.strip().str.len() > 0
    dataframe = dataframe[mask].drop_duplicates(subset=['vocab'], keep='last')
    csv_data  = dataframe.to_csv(index=False)

    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial commit", csv_data)

    load_data.clear()
    st.caption(f"⏱️ GitHub save: {time.perf_counter() - t0:.2f}s")   # C15
    return True

# ========================== D18: SESSION STATE INIT (setdefault — single pass) ==========================
st.session_state.setdefault("gemini_key",        DEFAULT_GEMINI_KEY)
st.session_state.setdefault("vocab_df",          load_data().copy())
st.session_state.setdefault("rpd_count",         load_usage())
st.session_state.setdefault("rpm_timestamps",    load_minute_usage())
st.session_state.setdefault("deck_id",           2059400110)
st.session_state.setdefault("bulk_preview_df",   None)
st.session_state.setdefault("apkg_buffer",       None)
st.session_state.setdefault("processed_vocabs",  [])
st.session_state.setdefault("model_id",          1607392319)
st.session_state.setdefault("include_antonyms",  True)
st.session_state.setdefault("dry_run",           False)
st.session_state.setdefault("processed_cache",   {})
st.session_state.setdefault("word_cache",        {})   # C14: per-word Gemini result cache
st.session_state.setdefault("input_phrase",      "")
st.session_state.setdefault("input_vocab",       "")
st.session_state.setdefault("_quota_cache_key",  None)   # D20
st.session_state.setdefault("_quota_cache",      (20, 0))
st.session_state.setdefault("target_lang",       "Indonesian")
st.session_state.setdefault("gemini_model_name", "gemini-2.5-flash-lite")

# ========================== CALLBACKS ==========================
def mark_as_done_callback():
    if st.session_state.processed_vocabs:
        st.session_state.vocab_df.loc[
            st.session_state.vocab_df['vocab'].isin(st.session_state.processed_vocabs),
            'status'
        ] = 'Done'
        save_to_github(st.session_state.vocab_df)
    st.session_state.apkg_buffer      = None
    st.session_state.processed_vocabs = []

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
            new_row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "tags": ""}])
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
    col1, col2  = st.columns(2)
    col1.metric("📖 Total", total_words)
    col2.metric("✨ New",   new_words)
    st.metric("🤖 Daily AI Usage", f"{st.session_state.rpd_count}/20 Requests")

    rpm_live = len([
        ts for ts in st.session_state.rpm_timestamps
        if (datetime.now() - ts).total_seconds() < 60
    ])
    st.progress(rpm_live / 5,                       text=f"RPM Live: {rpm_live}/5 (last 60s)")
    st.progress(st.session_state.rpd_count / 20,    text=f"RPD: {st.session_state.rpd_count}/20")

    st.divider()

    # key= stores directly in session_state — safe to read inside @st.fragment functions (A1/A2)
    st.selectbox(
        "🎯 Definition Language",
        ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"],
        index=0, key="target_lang"
    )
    st.selectbox(
        "🤖 AI Model",
        ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"],
        index=0, key="gemini_model_name"
    )

    st.divider()

    if st.button("🔄 Regenerate Note Type Model ID"):
        st.session_state.model_id = random.randrange(1 << 30, 1 << 31)
        st.success(f"New Model ID: {st.session_state.model_id}")
    st.caption(f"Current Model ID: {st.session_state.model_id}")

    # C14: Manual cache clear
    if st.button("🗑️ Clear Word Cache"):
        st.session_state.word_cache = {}
        st.session_state.processed_cache = {}
        st.toast("🗑️ Word cache cleared.")

    if not st.session_state.vocab_df.empty:
        st.download_button(
            "💾 Backup Database (CSV)",
            st.session_state.vocab_df.to_csv(index=False).encode('utf-8'),
            f"vocab_backup_{date.today()}.csv",
            "text/csv"
        )

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

# ──────────────────────────── TAB 1 ────────────────────────────
with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")

    if add_mode == "Single":
        p_raw      = st.text_input("🔤 Phrase", placeholder="Paste your sentence here...", key="input_phrase")
        v_selected = ""

        if p_raw and p_raw not in ["1", "*"]:
            clean_text   = _RE_CLEAN_TEXT.sub('', p_raw)   # B8: pre-compiled
            unique_words = list(dict.fromkeys([w.lower() for w in clean_text.split() if w.strip()]))
            if unique_words:
                st.caption("Click words below to extract them as vocabulary:")
                try:
                    selected_pills = st.pills(
                        "Select Vocab", unique_words,
                        selection_mode="multi", label_visibility="collapsed"
                    )
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
        if (v_check
                and not st.session_state.vocab_df.empty
                and (st.session_state.vocab_df['vocab'] == v_check).any()):
            st.warning(
                f"⚠️ '{v_check}' already exists. Saving will overwrite its phrase and reset to 'New'."
            )

        st.button(
            "💾 Save to Cloud", type="primary",
            use_container_width=True, on_click=save_single_word_callback
        )

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
                    new_rows.append({"vocab": bv, "phrase": bp, "status": "New", "tags": ""})
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
                st.rerun()   # A3: no sleep() before rerun

# ──────────────────────────── TAB 2 (A2: st.fragment isolates editor reruns) ────────────────────────────
with tab2:
    @st.fragment
    def render_tab2():
        """
        A2: Wrapping in @st.fragment means search, pagination, and data_editor
        interactions rerun ONLY this function — the sidebar quota bars and Tab 1/3
        are never touched on editor interactions.
        """
        if st.session_state.vocab_df.empty:
            st.info("Add words first!")
            return

        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        search     = st.text_input("🔎 Search...", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search:
            display_df = display_df[display_df['vocab'].str.contains(search, case=False)]

        page_size = 50
        page      = st.number_input("Page", min_value=1, value=1, step=1)
        start     = (page - 1) * page_size
        paginated = display_df.iloc[start:start + page_size]

        edited = st.data_editor(
            paginated, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status", options=["New", "Done"], required=True
                )
            }
        )

        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            st.session_state.vocab_df = edited
            save_to_github(st.session_state.vocab_df)
            st.toast("✅ Cloud updated!")
            st.rerun(scope="app")   # A3: no sleep(); full rerun to sync sidebar metrics

    render_tab2()

# ──────────────────────────── TAB 3 (A1: st.fragment isolates generation UI) ────────────────────────────
with tab3:
    @st.fragment
    def render_tab3():
        """
        A1: Wrapping in @st.fragment means deck name edits, slider drags, checkbox
        toggles, and the data_editor in Tab 3 rerun ONLY this function — Tab 1/2
        and the sidebar quota bars are never re-rendered on these interactions.
        """
        # ── Download state (shown at top after generation) ──
        if st.session_state.apkg_buffer is not None:
            st.success("✅ Deck generated! Download below to update your database.")
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
                st.rerun(scope="app")
            return

        if st.session_state.vocab_df.empty:
            st.info("Add words first!")
            return

        subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
        if subset.empty:
            st.warning("⚠️ No 'New' words to export! All words are marked 'Done'.")
            return

        st.subheader("📇 Generate Cyberpunk Anki Deck")

        deck_col1, deck_col2 = st.columns([3, 1])
        deck_name_input = deck_col1.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        if deck_col2.button("🎲 New Deck ID"):
            st.session_state.deck_id = random.randrange(1 << 30, 1 << 31)
        deck_col2.caption(f"ID: {st.session_state.deck_id}")

        # Batch size with auto-adjustment
        requests_left = max(0, 20 - st.session_state.rpd_count)
        raw_batch     = st.slider("⚡ Batch Size (Words per Request)", 1, 15, 10)
        max_safe      = (
            max(1, math.ceil(len(subset) / max(1, requests_left)))
            if requests_left > 0 else 1
        )
        batch_size = min(raw_batch, max_safe)
        st.caption(f"✅ Effective batch size: **{batch_size}** (quota-adjusted from {raw_batch})")

        include_audio                     = st.checkbox("🔊 Generate Audio Files",            value=True)
        st.session_state.include_antonyms = st.checkbox("➖ Include Antonyms in Card Back",    value=st.session_state.include_antonyms)
        st.session_state.dry_run          = st.checkbox("🔬 Dry Run Mode (simulate, no quota)", value=st.session_state.dry_run)

        # Export selector
        st.write("**Select words to export:**")
        subset['Export'] = True
        edited_export = st.data_editor(
            subset,
            column_config={
                "Export": st.column_config.CheckboxColumn("Export?", required=True)
            },
            hide_index=True,
            disabled=["vocab", "phrase", "status", "tags"]
        )
        final_export = edited_export[edited_export['Export'] == True]

        # Preview table + size estimate
        if not final_export.empty:
            st.write("### Export Preview")
            st.dataframe(final_export[['vocab', 'phrase']], hide_index=True)
            card_count  = len(final_export)
            est_size_kb = card_count * 2.5
            st.info(f"📊 **{card_count} cards** • Estimated .apkg size: **{est_size_kb:.1f} KB**")

        # D20: Memoized quota calculation — recomputes only when inputs actually change
        quota_key = (st.session_state.rpd_count, len(final_export), batch_size)
        if st.session_state._quota_cache_key != quota_key:
            r_left = max(0, 20 - st.session_state.rpd_count)
            r_req  = math.ceil(len(final_export) / batch_size) if not final_export.empty else 0
            st.session_state._quota_cache     = (r_left, r_req)
            st.session_state._quota_cache_key = quota_key
        requests_left, required_requests = st.session_state._quota_cache

        st.info(
            f"💡 You have **{requests_left}** API requests left today. "
            f"This batch requires **{required_requests}** request(s)."
        )

        if final_export.empty:
            st.warning("Select at least one word to export.")
        elif required_requests > requests_left and not st.session_state.dry_run:
            st.error("🛑 Exceeds Daily Limit! Reduce your selection or increase batch size.")
        else:
            if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
                raw_notes = []
                try:
                    raw_notes = process_anki_data(
                        final_export,
                        batch_size=batch_size,
                        dry_run=st.session_state.dry_run
                    )
                    if raw_notes:
                        apkg = create_anki_package(
                            raw_notes, deck_name_input,
                            generate_audio=include_audio,
                            deck_id=st.session_state.deck_id,
                            include_antonyms=st.session_state.include_antonyms
                        )
                        st.session_state.apkg_buffer      = apkg.getvalue()
                        st.session_state.processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                        st.rerun(scope="app")   # A3: no sleep; full rerun to show download button
                except Exception as e:
                    st.error(f"❌ Generation error: {e} — Status rolled back to 'New'.")
                    if raw_notes:
                        failed = [n.get('VocabRaw', '') for n in raw_notes]
                        st.session_state.vocab_df.loc[
                            st.session_state.vocab_df['vocab'].isin(failed), 'status'
                        ] = 'New'
                        save_to_github(st.session_state.vocab_df)

    render_tab3()
