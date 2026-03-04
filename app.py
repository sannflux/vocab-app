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
import threading

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["REPO_NAME"]
DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

# Dynamic API key (new feature)
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.last_key = None

# GitHub connection
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# Dynamic Gemini model
def get_gemini_model():
    key = st.session_state.gemini_key
    if st.session_state.model is None or st.session_state.last_key != key:
        try:
            genai.configure(api_key=key)
            st.session_state.model = genai.GenerativeModel(
                "gemini-2.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            st.session_state.last_key = key
        except Exception as e:
            st.error(f"❌ Gemini key error: {e}")
            return None
    return st.session_state.model

USER_NATIVE_LANGUAGE = "Indonesian"

# ========================== EXACT COLAB CLEANING FUNCTIONS ==========================
def cap_first(s):
    if not isinstance(s, str) or not s: return s
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s

def ensure_trailing_dot(s):
    if not isinstance(s, str) or not s: return s
    s = s.strip()
    return s if s.endswith('.') else s + '.'

def normalize_spaces(text):
    if not isinstance(text, str): return text
    return re.sub(r"\s{2,}", " ", text).strip()

def clean_grammar(text):
    if not isinstance(text, str): return text
    rules = [(r"\bto doing\b", "to do"), (r"\bfor helps\b", "to help"),
             (r"\bis use to\b", "is used to"), (r"\bhelp for to\b", "help to"),
             (r"\bfor to\b", "to"), (r"\bcan able to\b", "can")]
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text

def cap_each_sentence(text):
    if not isinstance(text, str): return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join([s[0].upper() + s[1:] if len(s) > 1 and s.strip() else s.upper() for s in sentences if s.strip()])

def highlight_vocab(text, vocab_raw):
    if not text: return text
    pattern = r'\b' + re.escape(vocab_raw) + r'\b'
    return re.sub(pattern, f'<b><u>{vocab_raw}</u></b>', text, flags=re.IGNORECASE)

def fix_vocab_casing_in_phrase(phrase, vocab_raw):  # ← NEW fix
    if not phrase or not vocab_raw: return phrase
    pattern = r'\b' + re.escape(vocab_raw.lower()) + r'\b'
    return re.sub(pattern, vocab_raw, phrase, flags=re.IGNORECASE)

# ========================== BATCH GENERATOR (strict plain text) ==========================
def run_with_timeout(func, args=(), kwargs=None, timeout=40):
    if kwargs is None: kwargs = {}
    result = {"value": None}
    def target():
        try: result["value"] = func(*args, **kwargs)
        except Exception as e: result["value"] = e
    t = threading.Thread(target=target)
    t.start()
    t.join(timeout)
    return result["value"] if not t.is_alive() else None

def generate_anki_card_data_batched(vocab_phrase_list, native_lang=USER_NATIVE_LANGUAGE, batch_size=5):
    model = get_gemini_model()
    if not model: return []
    all_card_data = []
    for i in range(0, len(vocab_phrase_list), batch_size):
        batch = vocab_phrase_list[i:i+batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES: Copy ALL fields exactly. If phrase empty: generate ONE simple sentence (max 12 words). EXACT vocab unchanged.
NEVER use markdown, asterisks, bold, italic, or any formatting. Plain text only.
OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{native_lang} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "Plain text only."}}]
BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""
        for attempt in range(4):
            response = run_with_timeout(model.generate_content, args=(prompt,), timeout=40)
            if response is None: time.sleep(2); continue
            try:
                raw = response.text.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    break
            except: time.sleep(2); continue
            if attempt == 3:
                for item in batch_dicts:
                    v = item["vocab"]
                    all_card_data.append({"vocab":v,"phrase":item["phrase"] or f"{v}.","translation":"?","part_of_speech":"","pronunciation_ipa":"","definition_english":"","example_sentences":[],"synonyms_antonyms":{"synonyms":[],"antonyms":[]},"etymology":""})
        time.sleep(1)
    return all_card_data

def generate_anki_notes(df):
    vocab_phrase_list = df[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list)
    anki_notes = []
    for card_data in all_card_data:
        vocab_raw = (card_data.get("vocab", "") or "").strip()
        vocab_cap = cap_first(vocab_raw)
        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing_in_phrase(phrase, vocab_raw)  # fixes capital issue
        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""
        translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("translation", "?"))))
        pos = card_data.get("part_of_speech", "").title()
        ipa = card_data.get("pronunciation_ipa", "")
        eng_def = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))
        examples = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e)))) for e in (card_data.get("example_sentences", []) or [])]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""
        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ensure_trailing_dot(", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])]))
        antonyms_field = ensure_trailing_dot(", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])]))
        etymology = normalize_spaces(card_data.get("etymology", ""))
        text_field = f"{formatted_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>" if formatted_phrase else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"
        anki_notes.append({"Text": text_field, "Pronunciation": pronunciation_field, "Definition": eng_def,
                           "Examples": examples_field, "Synonyms": synonyms_field, "Antonyms": antonyms_field, "Etymology": etymology})
    return pd.DataFrame(anki_notes)

# ========================== LOAD / SAVE / SPEECH / WOTD ==========================
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        return df.sort_values(by="vocab", ignore_index=True)
    except:
        return pd.DataFrame(columns=['vocab', 'phrase'])

def save_to_github(dataframe):
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    return True

df = load_data()

def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

wotd_vocab = wotd_phrase = None
if not df.empty:
    today_str = date.today().isoformat()
    random.seed(today_str)
    row = df.sample(n=1).iloc[0]
    wotd_vocab = row["vocab"]
    wotd_phrase = row.get("phrase", "")

# ========================== SIDEBAR (API KEY CHANGER) ==========================
with st.sidebar:
    st.header("🌟 Word of the Day")
    if wotd_vocab:
        st.subheader(wotd_vocab.upper())
        if wotd_phrase.strip(): st.caption(wotd_phrase)
    st.divider()
    st.subheader("🔑 Gemini API Key")
    alt_key = st.text_input("Alternative key (when free tier exhausted)", type="password", value="", placeholder="AIzaSy...")
    if alt_key and alt_key != st.session_state.gemini_key:
        st.session_state.gemini_key = alt_key
        st.success("✅ Switched to new key!")
        st.rerun()
    st.divider()
    lang_options = {"🇬🇧 English (US)": "en-US", "🇮🇩 Indonesian": "id-ID", "🇯🇵 Japanese": "ja-JP"}
    selected_lang_name = st.selectbox("🎙️ Speech Language", list(lang_options.keys()), index=0)
    speech_lang = lang_options[selected_lang_name]
    if wotd_vocab:
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("🔊 Speak Vocab", key="wotd_v"): speak_word(wotd_vocab, speech_lang)
        with c2: 
            if wotd_phrase.strip() and st.button("🔊 Speak Phrase", key="wotd_p"): speak_word(wotd_phrase, speech_lang)

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add New Word", "✏️ Edit / Delete", "📇 Generate Anki (AI)"])

with tab1:
    st.subheader("Add a new vocabulary word")
    with st.form("add_form", clear_on_submit=True):
        v = st.text_input("📝 Vocab (required)", placeholder="e.g. serendipity").lower().strip()
        p_raw = st.text_input("🔤 Phrase / Example (type 1 to skip)", placeholder="I found it by serendipity!").strip()
        if st.form_submit_button("💾 Save to Cloud", use_container_width=True):
            if v and v not in df['vocab'].values:
                p = "" if p_raw.upper() == "1" else p_raw.capitalize()
                updated = pd.concat([df, pd.DataFrame([{"vocab": v, "phrase": p}])], ignore_index=True)
                if save_to_github(updated): st.success(f"✅ '{v}' added!"); st.rerun()

with tab2:
    if df.empty:
        st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        search = st.text_input("🔎 Search...", "").lower().strip()
        display_df = df[df['vocab'].str.contains(search, case=False)] if search else df
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True)
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("💾 Save Changes to Cloud", type="primary", use_container_width=True):
                if save_to_github(edited.sort_values(by="vocab", ignore_index=True)):
                    st.success("✅ Cloud updated!"); st.rerun()
        with col2:
            csv = edited.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "my_vocabulary.csv", "text/csv", use_container_width=True)
        st.divider()
        st.subheader("🔊 Quick Audio Practice")
        quick_word = st.selectbox("Choose word:", sorted(df["vocab"].tolist()))
        quick_phrase = df[df["vocab"]==quick_word]["phrase"].iloc[0]
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("🔊 Speak Vocab", key="q_v"): speak_word(quick_word, speech_lang)
        with c2: 
            if quick_phrase.strip() and st.button("🔊 Speak Phrase", key="q_p"): speak_word(quick_phrase, speech_lang)

with tab3:
    st.subheader("📇 Generate Anki (AI) — 100% identical to Colab")
    if df.empty:
        st.info("Add words first!")
    else:
        if st.button("🚀 Generate Anki Cards with Gemini AI", type="primary", use_container_width=True):
            with st.spinner("🧠 Generating (Synonyms capitalized + sentence casing fixed)..."):
                anki_df = generate_anki_notes(df)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"anki_cards_export_{timestamp}.csv"
                csv_bytes = anki_df.to_csv(index=False, header=False, encoding="utf-8-sig").encode()
                st.success(f"🎉 {len(anki_df)} cards ready!")
                st.download_button("📥 Download Anki CSV", csv_bytes, filename, "text/csv", use_container_width=True)
                st.dataframe(anki_df.head(3), use_container_width=True)

st.caption("✅ 100% identical to Colab + API key changer + better sentence capitalization")
