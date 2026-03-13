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

# --- MOBILE KEYBOARD AUTO-DISMISS HACK ---
# This script listens for the Enter key on standard inputs and forces the keyboard to close (blur)
st.components.v1.html("""
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            // Check if the active element is an input field
            if (doc.activeElement && (doc.activeElement.tagName === 'INPUT')) {
                doc.activeElement.blur(); // Dismiss Android keyboard
            }
        }
    });
    </script>
""", height=0, width=0)

# --- INITIALIZE SESSION STATE ---
if "p_input_field" not in st.session_state: st.session_state.p_input_field = ""
if "v_input_field" not in st.session_state: st.session_state.v_input_field = ""
if "word_picker" not in st.session_state: st.session_state.word_picker = None

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
@st.cache_resource
def get_github_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except GithubException as e:
        st.error(f"❌ GitHub connection failed: {e}")
        return None

repo = get_github_repo()

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
    if s.endswith(','): s = s[:-1].rstrip()
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

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    pattern = r'\b' + re.escape(vocab.lower()) + r'\b'
    return re.sub(pattern, vocab, phrase, flags=re.IGNORECASE)

# ========================== SPEECH ==========================
def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== AI LOGIC ==========================
def robust_json_parse(text: str):
    try: return json.loads(text)
    except: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []
    all_card_data = []
    progress_bar = st.progress(0)
    total_items = len(vocab_phrase_list)

    for i in range(0, total_items, batch_size):
        progress_bar.progress(i / total_items, text=f"🤖 AI {i}/{total_items}...")
        batch = vocab_phrase_list[i:i + batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array. RULES: 1. Copy ALL fields exactly. 2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT. 3. IF 'phrase' is empty: Generate ONE simple sentence. 4. EXACT vocab unchanged. NEVER use markdown. OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}] BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""
        for attempt in range(5):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed); break
            except Exception: time.sleep((2 ** attempt) + 1)
    progress_bar.empty()
    return all_card_data

def process_anki_data(df_subset, batch_size=6):
    df_subset = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []
    for card_data in all_card_data:
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        phrase = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("phrase", "")))))
        phrase = fix_vocab_casing(phrase, vocab_raw)
        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""
        translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("translation", "?"))))
        pos = str(card_data.get("part_of_speech", "")).title()
        ipa = card_data.get("pronunciation_ipa", "")
        eng_def = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))
        examples = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e)))) for e in (card_data.get("example_sentences", []) or [])[:3]]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""
        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ensure_trailing_dot(", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]]))
        antonyms_field = ensure_trailing_dot(", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])[:5]]))
        etymology = normalize_spaces(card_data.get("etymology", ""))
        text_field = f"{formatted_phrase}<br><br>{cap_first(vocab_raw)}: <b>{{{{c1::{translation}}}}}</b>"
        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"
        processed_notes.append({"VocabRaw": vocab_raw, "Text": text_field, "Pronunciation": pronunciation_field, "Definition": eng_def, "Examples": examples_field, "Synonyms": synonyms_field, "Antonyms": antonyms_field, "Etymology": etymology})
    return processed_notes

# ========================== AUDIO & PACKAGE ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        fname = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        fpath = os.path.join(temp_dir, fname)
        if vocab.strip():
            gTTS(text=vocab, lang='en', slow=False).save(fpath)
            return vocab, fname, fpath
    except Exception: pass
    return vocab, None, None

CYBERPUNK_CSS = ".card { font-family: 'Roboto Mono', monospace; font-size: 18px; color: #00ff41; background-color: #111111; padding: 30px 20px; } .vellum-focus-container { background: #0d0d0d; padding: 30px 20px; border: 2px solid #00ff41; text-align: center; } .prompt-text { font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff; } .cloze { color: #111111; background-color: #00ff41; } .solved-text .cloze { color: #ff00ff; background: none; border-bottom: 3px double #00ffff; } .vellum-section { margin: 15px 0; border-bottom: 1px dashed #00ff41; } .section-header { font-weight: 600; color: #00ffff; } .content { color: #aaffaa; }"

def create_anki_package(notes_data, deck_name, generate_audio=True):
    front = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}</div>{{Audio}}"""
    my_model = genanki.Model(1607392319, 'Cyberpunk Cloze', fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], templates=[{'name': 'Card 1', 'qfmt': front, 'afmt': back}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            v_list = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                futures = {ex.submit(generate_audio_file, v, temp_dir): v for v in v_list}
                for f in concurrent.futures.as_completed(futures):
                    vk, fn, fp = f.result()
                    if fn: media_files.append(fp); audio_map[vk] = f"[sound:{fn}]"
        for n in notes_data:
            my_deck.add_note(genanki.Note(model=my_model, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")] ))
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buf = io.BytesIO()
        out_p = os.path.join(temp_dir, 'out.apkg')
        my_package.write_to_file(out_p)
        with open(out_p, "rb") as f: buf.write(f.read())
        buf.seek(0)
        return buf

# ========================== STORAGE ==========================
@st.cache_data(ttl=600)
def load_data():
    if not repo: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])
    try:
        fc = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(fc.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        if 'status' not in df.columns: df['status'] = 'New'
        return df.sort_values(by="vocab", ignore_index=True)
    except: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_to_github(dataframe):
    if not repo: return False
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        f = repo.get_contents("vocabulary.csv")
        repo.update_file(f.path, "Update", csv_data, f.sha)
    except: repo.create_file("vocabulary.csv", "Init", csv_data)
    load_data.clear()
    return True

df = load_data().copy()

# ========================== UI LOGIC ==========================
with st.sidebar:
    st.divider()
    if not df.empty:
        st.download_button("💾 Backup CSV", df.to_csv(index=False).encode('utf-8'), f"backup_{date.today()}.csv", "text/csv")
        st.divider()
        st.header("🌟 Word of the Day")
        random.seed(date.today().isoformat())
        try:
            row = df.sample(n=1).iloc[0]
            st.subheader(row["vocab"].upper())
            if row["phrase"]: st.caption(row["phrase"])
            if st.button("🔊 Pronounce"): speak_word(row["vocab"])
        except: pass

tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Anki"])

with tab1:
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")
    
    if add_mode == "Single":
        # We bind the input to session state key 'p_input_field'
        p_input = st.text_input("🔤 Paste Sentence/Phrase", key="p_input_field", placeholder="Paste sentence here...")
        
        # CALLBACK: Forces the vocab input state to match the clicked pill
        def sync_pill_to_vocab():
            if st.session_state.word_picker:
                st.session_state.v_input_field = st.session_state.word_picker.lower()

        if p_input:
            words = re.findall(r"[\w']+", p_input)
            if words:
                unique_words = list(dict.fromkeys(words))
                # VERSION SAFE: Using st.pills
                try:
                    st.pills("👆 Tap a word to study", options=unique_words, key="word_picker", on_change=sync_pill_to_vocab)
                except AttributeError:
                    st.caption("Tap word to study (Update Streamlit for best experience)")

        # We bind the input to session state key 'v_input_field'
        st.text_input("📝 Vocab to Save", key="v_input_field", placeholder="Enter word manually or pick above")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", type="primary", use_container_width=True):
                # FIX: Read directly from session_state to ensure we capture the pill selection
                current_v_input = st.session_state.v_input_field.lower().strip()
                current_p_input = st.session_state.p_input_field.strip()

                if current_v_input:
                    proc_phrase = ""
                    if current_p_input and current_p_input != "1":
                        proc_phrase = current_p_input if current_p_input.startswith("*") else ensure_trailing_dot(cap_first(current_p_input))
                    
                    if not df.empty and current_v_input in df['vocab'].values:
                        df.loc[df['vocab'] == current_v_input, ['phrase', 'status']] = [proc_phrase, 'New']
                    else:
                        df = pd.concat([df, pd.DataFrame([{"vocab": current_v_input, "phrase": proc_phrase, "status": "New"}])], ignore_index=True)
                    
                    if save_to_github(df):
                        # Clear inputs after success
                        st.session_state.p_input_field = ""
                        st.session_state.v_input_field = ""
                        st.session_state.word_picker = None
                        st.success(f"✅ Saved '{current_v_input}'!")
                        time.sleep(0.5)
                        st.rerun()
        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.p_input_field = ""
                st.session_state.v_input_field = ""
                st.session_state.word_picker = None
                st.rerun()

    else:
        st.info("Format: `vocab, phrase` or `vocab` per line")
        bulk_text = st.text_area("Paste List", height=150)
        if st.button("💾 Process Bulk", type="primary", use_container_width=True):
            new_rows = []
            for line in bulk_text.split('\n'):
                if not line.strip(): continue
                parts = line.split(',', 1)
                bv = parts[0].strip().lower()
                bp = ensure_trailing_dot(cap_first(parts[1].strip())) if len(parts) > 1 else ""
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "status": "New"})
            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows)]).drop_duplicates(subset=['vocab'], keep='last')
                if save_to_github(df): st.success(f"✅ Added {len(new_rows)} words!"); time.sleep(1); st.rerun()

with tab2:
    if df.empty: 
        st.info("Add words first!")
    else:
        st.write("### 🔍 Search & Edit")
        search = st.text_input("Search word...", "").lower().strip()
        filtered_df = df.copy()
        
        if search:
            filtered_df = filtered_df[filtered_df['vocab'].str.contains(search, case=False)]
            
        if not filtered_df.empty:
            vocab_to_edit = st.selectbox("👆 Select word to edit", filtered_df['vocab'].tolist())
            
            if vocab_to_edit:
                row_data = df[df['vocab'] == vocab_to_edit].iloc[0]
                
                with st.form(key=f"edit_form_{vocab_to_edit}"):
                    new_phrase = st.text_input("🔤 Phrase", value=row_data['phrase'])
                    new_status = st.selectbox("Status", ["New", "Done"], index=0 if row_data['status'] == 'New' else 1)
                    
                    if st.form_submit_button("💾 Save Changes", use_container_width=True):
                        df.loc[df['vocab'] == vocab_to_edit, ['phrase', 'status']] = [new_phrase, new_status]
                        if save_to_github(df):
                            st.success("✅ Updated!")
                            time.sleep(0.5)
                            st.rerun()
                            
                if st.button("🗑️ Delete Word", type="secondary", use_container_width=True):
                    df = df[df['vocab'] != vocab_to_edit]
                    if save_to_github(df):
                        st.warning("🗑️ Deleted!")
                        time.sleep(0.5)
                        st.rerun()
        else:
            st.info("No words match your search.")

with tab3:
    if df.empty: st.info("Add words first!")
    else:
        st.metric("New Words Available", len(df[df['status'] == 'New']))
        deck_name_input = st.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        include_audio = st.checkbox("🔊 Audio", value=True)
        process_only_new = st.checkbox("Only process 'New'", value=True)

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            subset = df[df['status'] == 'New'] if process_only_new else df
            if subset.empty: st.warning("No words to process!")
            else:
                raw_notes = process_anki_data(subset, batch_size=5)
                if raw_notes:
                    with st.spinner("Packaging..."):
                        apkg = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio)
                    st.download_button("📥 Download .apkg", apkg, f"Anki_{datetime.now().strftime('%m%d_%H%M')}.apkg", "application/octet-stream", use_container_width=True)
                    if process_only_new:
                        df.loc[df['vocab'].isin(subset['vocab']), 'status'] = 'Done'
                        save_to_github(df)
