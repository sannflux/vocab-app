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

# IMPORTS FOR AUDIO & ANKI PACKAGE
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

# --- IDENTIFIED MODELS (RULE #1 OATH) ---
# I have identified the model as gemini-2.5-flash-lite and gemini-2.0-flash-exp. 
# I swear an oath that this model string, core features, and previously integrated refinements 
# will remain 100% unaltered.

# --- CSS VARIABLES ---
THEME_COLOR = "#00ff41"
THEME_GLOW = "rgba(0, 255, 65, 0.4)"
BG_COLOR = "#111111"
BG_STRIPE = "#181818"
TEXT_COLOR = "#aaffaa"

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

# --- SECRETS MANAGEMENT ---
try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

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

# ========================== API KEY ROTATION & QUOTA ==========================
def get_key_hash(api_key):
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]

def load_usage_all_keys():
    try:
        file = repo.get_contents("usage_v2.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        if data.get("date") == str(date.today()):
            return data.get("keys", {})
        return {}
    except:
        return {}

def save_usage_per_key(key_hash, count):
    current_usage = load_usage_all_keys()
    current_usage[key_hash] = count
    data = json.dumps({"date": str(date.today()), "keys": current_usage})
    try:
        file = repo.get_contents("usage_v2.json")
        repo.update_file(file.path, "Update API usage V2", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage_v2.json", "Init API usage V2", data)

if "usage_dict" not in st.session_state:
    st.session_state.usage_dict = load_usage_all_keys()
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# ========================== GEMINI SETUP ==========================
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
    except Exception as e:
        st.error(f"❌ Gemini configuration error: {e}")
        return None

# ========================== CLEANING & TRUNCATION ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if s and s[-1] in ".!?" else (s + "." if s else "")

def truncate_context(text: str, limit: int = 200) -> str:
    if len(text) <= limit: return text
    return text[:limit-3].strip() + "..."

def clean_grammar(text: str) -> str:
    if not isinstance(text, str): return text
    rules = [(r"\bto doing\b", "to do"), (r"\bfor helps\b", "to help")]
    for pat, repl in rules:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def robust_json_parse(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try: return json.loads(text)
    except:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return None

# ========================== ASYNC BATCH GENERATOR (Idea 6: Multi-Cloze) ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    all_keys = [DEFAULT_GEMINI_KEY]
    if st.session_state.get("backup_keys"):
        extra = [k.strip() for k in st.session_state.backup_keys.split(",") if k.strip()]
        all_keys.extend(extra)

    all_card_data = []
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    current_key_idx = 0
    
    with st.status("🤖 AI Processing Multi-Cloze Data...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        
        for idx, batch in enumerate(batches):
            key_found = False
            while current_key_idx < len(all_keys):
                active_key = all_keys[current_key_idx]
                k_hash = get_key_hash(active_key)
                if st.session_state.usage_dict.get(k_hash, 0) < 20:
                    key_found = True
                    break
                else:
                    current_key_idx += 1

            if not key_found: break

            if idx > 0: time.sleep(13)

            batch_dicts = [{"vocab": v[0], "phrase": truncate_context(v[1])} for v in batch]
            
            # Updated Prompt for Idea 6: Multi-Cloze
            prompt = f"""Output ONLY a JSON array. 
            RULES: 
            1. 'translation' is ONLY the {TARGET_LANG} word.
            2. 'cloze_text' is a combination string: 
               Format: "{{{{c2::vocab_word}}}}: {{{{c1::translation_word}}}}"
               If a phrase exists, include it: "<b><u>{{{{c2::vocab_word}}}}</u></b> in context phrase.<br><br>{{{{c2::vocab_word}}}}: <b>{{{{c1::translation}}}}</b>"
            FORMAT: [{{"vocab": "...", "cloze_text": "...", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}]
            INPUT: {json.dumps(batch_dicts)}"""

            model = get_gemini_model(active_key, GEMINI_MODEL)
            try:
                response = model.generate_content(prompt)
                st.session_state.total_tokens += response.usage_metadata.total_token_count
                k_hash = get_key_hash(active_key)
                st.session_state.usage_dict[k_hash] = st.session_state.usage_dict.get(k_hash, 0) + 1
                save_usage_per_key(k_hash, st.session_state.usage_dict[k_hash])
                
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list): all_card_data.extend(parsed)
            except: pass
            progress_bar.progress((idx + 1) / len(batches))

    return all_card_data

def process_anki_data(df_subset, batch_size=6):
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []

    for card_data in all_card_data:
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue
        
        # Note: Idea 6 uses cloze_text provided by AI for Text field
        processed_notes.append({
            "VocabRaw": vocab_raw, 
            "Text": card_data.get("cloze_text", ""), 
            "Pronunciation": f"<b>[{card_data.get('part_of_speech', '')}]</b> {card_data.get('pronunciation_ipa', '')}",
            "Definition": card_data.get("definition_english", ""), 
            "Examples": "<ul>" + "".join(f"<li>{e}</li>" for e in card_data.get("example_sentences", [])) + "</ul>",
            "Synonyms": ", ".join(card_data.get("synonyms_antonyms", {}).get("synonyms", [])),
            "Antonyms": ", ".join(card_data.get("synonyms_antonyms", {}).get("antonyms", [])),
            "Etymology": card_data.get("etymology", ""), 
            "Tags": []
        })
    return processed_notes

# ========================== GENANKI & STYLES ==========================
CYBERPUNK_CSS = f".card {{ font-family: 'Consolas'; color: {THEME_COLOR}; background-color: {BG_COLOR}; padding: 20px; }} .vellum-focus-container {{ border: 2px solid {THEME_COLOR}; padding: 20px; text-align: center; }} .prompt-text {{ font-size: 1.5em; color: white; }} .cloze {{ color: {BG_COLOR}; background: {THEME_COLOR}; }}"

def create_anki_package(notes_data, deck_name, generate_audio=True):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""
    
    my_model = genanki.Model(1607392319, 'Multi-Cloze Model', fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            for v in {n['VocabRaw'] for n in notes_data}:
                clean_v = re.sub(r'[^a-zA-Z0-9]', '', v) + ".mp3"
                fp = os.path.join(temp_dir, clean_v)
                try: 
                    gTTS(text=v, lang='en').save(fp)
                    media_files.append(fp); audio_map[v] = f"[sound:{clean_v}]"
                except: pass
        
        for n in notes_data:
            guid = str(int(hashlib.sha256(n['VocabRaw'].encode()).hexdigest(), 16) % (10**10))
            my_deck.add_note(genanki.Note(model=my_model, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")], guid=guid))
            
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buf = io.BytesIO()
        out = os.path.join(temp_dir, 'out.apkg')
        my_package.write_to_file(out)
        with open(out, "rb") as f: buf.write(f.read())
        buf.seek(0)
    return buf

# ========================== GITHUB CSV LOAD/SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file = repo.get_contents("vocabulary.csv")
        return pd.read_csv(io.StringIO(file.decoded_content.decode('utf-8')), dtype=str).fillna("")
    except: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_to_github(df):
    csv = df.drop_duplicates(subset=['vocab']).to_csv(index=False)
    try:
        f = repo.get_contents("vocabulary.csv")
        repo.update_file(f.path, "Update", csv, f.sha)
    except: repo.create_file("vocabulary.csv", "Init", csv)
    load_data.clear()

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()
if "apkg_buffer" not in st.session_state: st.session_state.apkg_buffer = None
if "processed_vocabs" not in st.session_state: st.session_state.processed_vocabs = []

def mark_as_done_callback():
    if st.session_state.processed_vocabs:
        st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(st.session_state.processed_vocabs), 'status'] = 'Done'
        save_to_github(st.session_state.vocab_df)
    st.session_state.apkg_buffer = None
    st.session_state.processed_vocabs = []

# ========================== UI ==========================
with st.sidebar:
    st.header("⚙️ API Control")
    st.metric("🪙 Tokens Used", f"{st.session_state.total_tokens:,}")
    st.text_area("Backup Keys", key="backup_keys", placeholder="AIzaSy...")
    TARGET_LANG = st.selectbox("🎯 Target Language", ["Indonesian", "Spanish", "French", "Japanese"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)

t1, t2, t3 = st.tabs(["➕ Add", "✏️ Review", "📇 Generate"])

with t1:
    v_in = st.text_input("Vocab")
    p_in = st.text_area("Context Phrase")
    if st.button("💾 Save"):
        if v_in:
            new_row = pd.DataFrame([{"vocab": v_in.lower().strip(), "phrase": p_in.strip(), "status": "New"}])
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
            save_to_github(st.session_state.vocab_df)
            st.toast("Saved!")

with t2:
    st.dataframe(st.session_state.vocab_df, use_container_width=True)

with t3:
    if st.session_state.apkg_buffer:
        st.download_button("📥 Download Anki (Multi-Cloze)", data=st.session_state.apkg_buffer, file_name="deck.apkg", on_click=mark_as_done_callback, use_container_width=True)
    else:
        new_subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']
        if not new_subset.empty and st.button("🚀 Generate Multi-Cloze Deck", type="primary"):
            notes = process_anki_data(new_subset)
            if notes:
                buf = create_anki_package(notes, "Multi-Cloze Deck")
                st.session_state.apkg_buffer = buf.getvalue()
                st.session_state.processed_vocabs = [n['VocabRaw'] for n in notes]
                st.rerun()
