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

# --- SECRETS MANAGEMENT ---
try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

if "deck_ready" not in st.session_state:
    st.session_state.deck_ready = False
    st.session_state.apkg_buffer = None
    st.session_state.apkg_filename = ""

# Initialization for persistence
if "p_input_text" not in st.session_state:
    st.session_state.p_input_text = ""

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_LANG = st.selectbox("🎯 Definition Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    st.divider()
    st.header("🔑 Gemini API Key")
    alt_key = st.text_input("Alternative key", type="password", value="", placeholder="AIzaSy...")
    if alt_key and alt_key != st.session_state.gemini_key:
        st.session_state.gemini_key = alt_key
        st.success("✅ Switched!")
        st.rerun()

# ========================== GITHUB CONNECT ==========================
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except Exception as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== GEMINI ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json", "temperature": 0.1})
    except Exception as e:
        st.error(f"❌ Gemini key error: {e}")
        return None

# ========================== CLEANING FUNCTIONS ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    if not s: return ""
    return s if s[-1] in ".!?" else s + "."

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() if text else ""

def clean_grammar(text: str) -> str:
    if not isinstance(text, str): return text
    rules = [(r"\bto doing\b", "to do"), (r"\bfor helps\b", "to help"), (r"\bis use to\b", "is used to"), (r"\bhelp for to\b", "help to"), (r"\bfor to\b", "to"), (r"\bcan able to\b", "can")]
    for pat, repl in rules: text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def cap_each_sentence(text: str) -> str:
    if not isinstance(text, str): return text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join([cap_first(s) for s in sentences if s.strip()])

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    pattern = r'\b' + re.escape(vocab) + r'\b'
    return re.sub(pattern, f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

# ========================== SPEECH & AI BATCH ==========================
def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

def robust_json_parse(text: str):
    try: 
        return json.loads(text)
    except Exception:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except Exception: pass
    return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []
    all_card_data = []
    pb = st.progress(0)
    for i in range(0, len(vocab_phrase_list), batch_size):
        pb.progress(i / len(vocab_phrase_list), text=f"🤖 AI Processing {i}/{len(vocab_phrase_list)}...")
        batch = vocab_phrase_list[i:i + batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        prompt = f"""Output ONLY JSON array. BATCH: {json.dumps(batch_dicts)}. Format: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}]"""
        for attempt in range(5):
            try:
                res = model.generate_content(prompt)
                parsed = robust_json_parse(res.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    break
            except Exception: 
                time.sleep(2**attempt)
    pb.empty()
    return all_card_data

def process_anki_data(df_subset, batch_size=6):
    raw_data = generate_anki_card_data_batched(df_subset[['vocab', 'phrase']].values.tolist(), batch_size)
    processed = []
    for d in raw_data:
        v_raw = str(d.get("vocab", "")).strip().lower()
        ph = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(d.get("phrase", "")))))
        processed.append({
            "VocabRaw": v_raw, 
            "Text": f"{highlight_vocab(ph, v_raw)}<br><br>{cap_first(v_raw)}: <b>{{{{c1::{ensure_trailing_dot(d.get('translation', '?'))}}}}}</b>",
            "Pronunciation": f"<b>[{str(d.get('part_of_speech', '')).title()}]</b> {d.get('pronunciation_ipa', '')}",
            "Definition": ensure_trailing_dot(cap_each_sentence(d.get("definition_english", ""))),
            "Examples": "<ul>" + "".join(f"<li><i>{ensure_trailing_dot(e)}</i></li>" for e in (d.get("example_sentences", []) or [])[:3]) + "</ul>",
            "Synonyms": ", ".join(d.get("synonyms_antonyms", {}).get("synonyms", [])[:5]),
            "Antonyms": ", ".join(d.get("synonyms_antonyms", {}).get("antonyms", [])[:5]),
            "Etymology": d.get("etymology", "")
        })
    return processed

# ========================== GENANKI & GITHUB ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        fname = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        fpath = os.path.join(temp_dir, fname)
        if vocab.strip():
            gTTS(text=vocab, lang='en').save(fpath)
            return vocab, fname, fpath
    except Exception: pass
    return vocab, None, None

def create_anki_package(notes, deck_name, generate_audio=True):
    CYBERPUNK_CSS = ".card { font-family: 'Roboto Mono', monospace; font-size: 18px; color: #00ff41; background-color: #111111; padding: 30px 20px; } .vellum-focus-container { border: 2px solid #00ff41; text-align: center; padding: 20px; } .cloze { color: #111111; background-color: #00ff41; }"
    model = genanki.Model(1607392319, 'Cyber Model', fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], templates=[{'name': 'C1', 'qfmt': '<div class="vellum-focus-container">{{cloze:Text}}</div>', 'afmt': '<div class="vellum-focus-container">{{cloze:Text}}</div><br>{{Definition}}<br>{{Pronunciation}}<br>{{Examples}}<br>{{Audio}}'}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    deck = genanki.Deck(2059400110, deck_name)
    media = []
    with tempfile.TemporaryDirectory() as td:
        amap = {}
        if generate_audio:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                futures = {ex.submit(generate_audio_file, n['VocabRaw'], td): n for n in notes}
                for f in concurrent.futures.as_completed(futures):
                    v, fn, fp = f.result()
                    if fn: media.append(fp); amap[v] = f"[sound:{fn}]"
        for n in notes:
            deck.add_note(genanki.Note(model=model, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], amap.get(n['VocabRaw'], "")]))
        pkg = genanki.Package(deck); pkg.media_files = media
        buf = io.BytesIO(); out = os.path.join(td, 'o.apkg'); pkg.write_to_file(out)
        with open(out, "rb") as f: buf.write(f.read())
        buf.seek(0); return buf

@st.cache_data(ttl=600)
def load_data():
    try:
        content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(content.decoded_content.decode('utf-8')))
    except Exception: 
        df = pd.DataFrame(columns=['vocab', 'phrase', 'status'])
    
    # Enforce schema
    for col in ['vocab', 'phrase', 'status']:
        if col not in df.columns:
            df[col] = 'New' if col == 'status' else ""
    df['phrase'] = df['phrase'].fillna("")
    return df.sort_values(by="vocab", ignore_index=True)

def save_to_github(df):
    df = df[df['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv = df.to_csv(index=False)
    try:
        f = repo.get_contents("vocabulary.csv")
        repo.update_file(f.path, "Update", csv, f.sha)
    except Exception: 
        repo.create_file("vocabulary.csv", "Init", csv)
    load_data.clear(); st.session_state.df = df.copy()
    return True

if "df" not in st.session_state: st.session_state.df = load_data().copy()

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Anki"])

with tab1:
    st.subheader("Add new word")
    # key="p_input_text" natively binds UI to session state
    p_in = st.text_input("📝 Paste Word or Phrase", key="p_input_text", placeholder="e.g. Serendipity OR The cat sat on the mat.")
    
    words = sorted(list(set(re.findall(r'\b\w+\b', p_in.lower())))) if p_in.strip() else []
    
    v_sel = []
    if words:
        v_sel = st.pills("🎯 Tap words to extract:", options=words, selection_mode="multi")

    if st.button("💾 Save to Cloud", type="primary", use_container_width=True):
        p_val = st.session_state.p_input_text.strip()
        if p_val:
            # Check if user pasted a sentence but didn't pick any words
            if len(words) > 1 and not v_sel:
                st.warning("⚠️ Multiple words detected. Please tap the specific words you wish to learn from the list above.")
            else:
                vocabs_to_add = v_sel if v_sel else [p_val.lower()]
                current_df = st.session_state.df.copy()
                phrase_fixed = ensure_trailing_dot(cap_each_sentence(p_val))
                
                for v in vocabs_to_add:
                    v_clean = v.strip().lower()
                    if not current_df.empty and v_clean in current_df['vocab'].values:
                        current_df.loc[current_df['vocab'] == v_clean, 'phrase'] = phrase_fixed
                        current_df.loc[current_df['vocab'] == v_clean, 'status'] = 'New'
                    else:
                        current_df = pd.concat([current_df, pd.DataFrame([{"vocab": v_clean, "phrase": phrase_fixed, "status": "New"}])], ignore_index=True)
                
                if save_to_github(current_df):
                    st.success("✅ Saved!")
                    st.session_state.p_input_text = "" # Clears widget via key binding
                    time.sleep(0.5)
                    st.rerun()
        else:
            st.warning("Input is empty.")

with tab2:
    if not st.session_state.df.empty:
        search = st.text_input("🔎 Search...")
        display_df = st.session_state.df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        
        # key="vocab_editor_main" prevents state ghosting
        edited = st.data_editor(display_df, key="vocab_editor_main", use_container_width=True, hide_index=True)
        
        if st.button("💾 Save Changes"):
            df_final = st.session_state.df.copy().set_index('vocab')
            df_final.update(edited.set_index('vocab'))
            if save_to_github(df_final.reset_index()): st.toast("✅ Updated!")

with tab3:
    if st.session_state.df.empty: st.info("No words.")
    else:
        st.metric("New Words", len(st.session_state.df[st.session_state.df['status'] == 'New']))
        if st.button("🚀 Generate Anki Deck", type="primary", use_container_width=True):
            subset = st.session_state.df[st.session_state.df['status'] == 'New']
            if subset.empty: st.warning("No new words!")
            else:
                raw_n = process_anki_data(subset)
                buf = create_anki_package(raw_n, "VocabDeck")
                st.session_state.apkg_buffer = buf.getvalue()
                st.session_state.apkg_filename = f"Deck_{datetime.now().strftime('%m%d_%H%M')}.apkg"
                st.session_state.deck_ready = True
                subset_vocabs = subset['vocab'].tolist()
                st.session_state.df.loc[st.session_state.df['vocab'].isin(subset_vocabs), 'status'] = 'Done'
                save_to_github(st.session_state.df)
                st.rerun()

        if st.session_state.deck_ready:
            st.download_button("📥 Download .apkg", data=st.session_state.apkg_buffer, file_name=st.session_state.apkg_filename, mime="application/octet-stream", use_container_width=True)
