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

# ========================== CONSTANTS & CONFIG ==========================
STOP_WORDS = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "at", "from", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their"}

THEMES = {
    "Cyberpunk": """
        .card { font-family: 'Roboto Mono', monospace; color: #00ff41; background-color: #111111; padding: 30px; }
        .vellum-focus-container { background: #0d0d0d; border: 2px solid #00ff41; box-shadow: 0 0 10px #00ff41; text-align: center; padding: 20px; }
        .prompt-text { font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff; }
        .cloze { color: #111111; background-color: #00ff41; padding: 2px 4px; }
        .section-header { color: #00ffff; border-left: 3px solid #00ff41; padding-left: 10px; font-weight: bold; }
    """,
    "Vellum (Classic)": """
        .card { font-family: 'Georgia', serif; color: #2c3e50; background-color: #fdf6e3; padding: 30px; }
        .vellum-focus-container { background: #fffcf5; border: 1px solid #d3c6aa; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); text-align: center; padding: 20px; }
        .prompt-text { font-size: 1.6em; color: #1a1a1a; }
        .cloze { color: #c0392b; font-weight: bold; background: none; }
        .section-header { color: #7f8c8d; border-bottom: 1px solid #d3c6aa; margin-top: 10px; font-variant: small-caps; }
    """,
    "Midnight": """
        .card { font-family: 'Inter', sans-serif; color: #e0e0e0; background-color: #121212; padding: 30px; }
        .vellum-focus-container { background: #1e1e1e; border-radius: 10px; text-align: center; padding: 20px; }
        .prompt-text { font-size: 1.7em; color: #bb86fc; }
        .cloze { color: #03dac6; font-weight: bold; }
        .section-header { color: #cf6679; font-weight: bold; }
    """
}

ACCENTS = {"US (American)": "com", "UK (British)": "co.uk", "AU (Australian)": "com.au"}

st.set_page_config(page_title="Vocab App v2", layout="centered", page_icon="📚")

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

# --- SECRETS MANAGEMENT ---
try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}")
    st.stop()

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

# ========================== GITHUB CONNECT ==========================
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== PERSISTENT METADATA (Idea 1) ==========================
def load_metadata():
    try:
        content = repo.get_contents("metadata.json")
        data = json.loads(content.decoded_content.decode())
        if data.get("date") != str(date.today()):
            return {"date": str(date.today()), "rpd_count": 0}
        return data
    except:
        return {"date": str(date.today()), "rpd_count": 0}

def update_metadata(count):
    data = {"date": str(date.today()), "rpd_count": count}
    try:
        file = repo.get_contents("metadata.json")
        repo.update_file(file.path, "Update Usage", json.dumps(data), file.sha)
    except:
        repo.create_file("metadata.json", "Init Usage", json.dumps(data))

if "metadata" not in st.session_state:
    st.session_state.metadata = load_metadata()

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
        st.error(f"❌ Gemini key error: {e}")
        return None

# ========================== CLEANING & UTILS ==========================
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
    rules = [(r"\bto doing\b", "to do"), (r"\bis use to\b", "is used to"), (r"\bfor to\b", "to")]
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

# ========================== ASYNC BATCH GENERATOR (Idea 3) ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []

    all_card_data = []
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    
    with st.status("🤖 Processing AI Batches...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        
        for idx, batch in enumerate(batches):
            if st.session_state.metadata["rpd_count"] >= 20:
                st.warning("🛑 Daily AI Limit reached.")
                break
            
            if idx > 0: time.sleep(12) # RPM Throttle

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            # STRICTER PROMPT FOR TRANSLATION BUG FIX
            prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES:
1. 'translation': Translate ONLY the 'vocab' word/idiom into {TARGET_LANG}. Do NOT translate the whole phrase.
2. If 'phrase' contains the 'vocab', ensure it is used naturally.
3. If 'phrase' starts with '*', use it only as context for the definition.
OUTPUT SCHEMA: [{{"vocab": "...", "phrase": "...", "translation": "Strict word meaning only", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}]
INPUT: {json.dumps(batch_dicts)}"""

            try:
                response = model.generate_content(prompt)
                st.session_state.metadata["rpd_count"] += 1
                update_metadata(st.session_state.metadata["rpd_count"])
                parsed = json.loads(response.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    st.markdown(f"✅ Processed: `{', '.join([v[0] for v in batch])}`")
            except Exception as e:
                st.error(f"Batch Error: {e}")
            
            progress_bar.progress((idx + 1) / len(batches))
        status_log.update(label="✅ AI Complete!", state="complete")
    
    return all_card_data

def process_anki_data(df_subset, batch_size=6):
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []

    for card_data in all_card_data:
        v_raw = str(card_data.get("vocab", "")).strip().lower()
        if not v_raw: continue
        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = cap_each_sentence(clean_grammar(phrase))
        phrase = fix_vocab_casing(phrase, v_raw)
        
        # Build note fields
        text_field = f"{highlight_vocab(phrase, v_raw)}<br><br>{cap_first(v_raw)}: <b>{{{{c1::{card_data.get('translation', '?')}}}}}</b>"
        pron_field = f"<b>[{card_data.get('part_of_speech', '').title()}]</b> {card_data.get('pronunciation_ipa', '')}"
        
        processed_notes.append({
            "VocabRaw": v_raw, "Text": text_field, "Pronunciation": pron_field,
            "Definition": ensure_trailing_dot(card_data.get("definition_english", "")),
            "Examples": "<ul>" + "".join([f"<li><i>{e}</i></li>" for e in card_data.get("example_sentences", [])[:3]]) + "</ul>",
            "Synonyms": ", ".join(card_data.get("synonyms_antonyms", {}).get("synonyms", [])[:5]),
            "Antonyms": ", ".join(card_data.get("synonyms_antonyms", {}).get("antonyms", [])[:5]),
            "Etymology": card_data.get("etymology", "")
        })
    return processed_notes

# ========================== AUDIO HELPER (Idea 7) ==========================
def generate_audio_file(vocab, temp_dir, accent_tld):
    try:
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        file_path = os.path.join(temp_dir, clean_filename)
        if vocab.strip():
            tts = gTTS(text=vocab, lang='en', tld=accent_tld, slow=False)
            tts.save(file_path)
            return vocab, clean_filename, file_path
    except Exception as e: print(f"Audio error: {e}")
    return vocab, None, None

# ========================== GENANKI LOGIC ==========================
def create_anki_package(notes_data, deck_name, theme_css, generate_audio=True, accent_tld="com"):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""
    
    my_model = genanki.Model(1607392319, 'Custom Vocab Model', 
        fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], 
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}], 
        css=theme_css, model_type=genanki.Model.CLOZE)
    
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(generate_audio_file, v, temp_dir, accent_tld) for v in unique_vocabs]
                for f in concurrent.futures.as_completed(futures):
                    vk, fn, fp = f.result()
                    if fn: media_files.append(fp); audio_map[vk] = f"[sound:{fn}]"
        
        for n in notes_data:
            my_deck.add_note(genanki.Note(model=my_model, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")]))
        
        output_path = os.path.join(temp_dir, 'output.apkg')
        genanki.Package(my_deck, media_files).write_to_file(output_path)
        with open(output_path, "rb") as f: buffer = io.BytesIO(f.read())
    return buffer

# ========================== GITHUB SYNC ==========================
@st.cache_data(ttl=300)
def load_data():
    try:
        content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna(""); df['status'] = df.get('status', 'New')
        return df.sort_values(by="vocab", ignore_index=True)
    except: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_to_github(dataframe):
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, f"Update {len(dataframe)} words", csv_data, file.sha)
    except: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear(); return True

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()
if "pending_queue" not in st.session_state: st.session_state.pending_queue = []

# ========================== SIDEBAR (Ideas 5 & 7) ==========================
with st.sidebar:
    st.header("⚙️ Configuration")
    TARGET_LANG = st.selectbox("🎯 Target Language", ["Indonesian", "Spanish", "French", "German", "Japanese"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 Model", ["gemini-1.5-flash-latest", "gemini-2.5-flash-lite"], index=0)
    SELECTED_THEME = st.selectbox("🎨 Anki Theme", list(THEMES.keys()), index=0)
    SELECTED_ACCENT = st.selectbox("🗣️ Audio Accent", list(ACCENTS.keys()), index=0)
    st.divider()
    st.metric("Daily AI Usage", f"{st.session_state.metadata['rpd_count']}/20")
    if st.button("🔄 Refresh Cloud Data"): load_data.clear(); st.rerun()

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add Words", "✏️ Edit Cloud", "📇 Generate Anki"])

with tab1:
    st.subheader("Add to Pending Queue")
    p_raw = st.text_input("🔤 Context Phrase", placeholder="Paste sentence...", key="input_phrase")
    
    # Idea 8: Smart Extraction
    if p_raw and p_raw not in ["1", "*"]:
        clean_text = re.sub(r'[^\w\s\-\']', '', p_raw)
        words = [w.lower() for w in clean_text.split() if w.lower() not in STOP_WORDS]
        unique_words = list(dict.fromkeys(words))
        if unique_words:
            st.caption("Extract Vocab:")
            selected_pills = st.pills("Select", unique_words, selection_mode="multi", label_visibility="collapsed")
            if selected_pills: st.session_state.input_vocab = " ".join(selected_pills)

    v_input = st.text_input("📝 Vocabulary Word(s)", key="input_vocab").lower().strip()
    
    # Idea 2: Duplicate Check
    if v_input and not st.session_state.vocab_df.empty:
        if v_input in st.session_state.vocab_df['vocab'].values:
            st.warning(f"⚠️ '{v_input}' is already in your Cloud database.")

    if st.button("➕ Add to Queue", use_container_width=True):
        if v_input:
            st.session_state.pending_queue.append({"vocab": v_input, "phrase": p_raw, "status": "New"})
            st.session_state.input_phrase = ""; st.session_state.input_vocab = ""
            st.rerun()

    if st.session_state.pending_queue:
        st.divider()
        st.write(f"📋 **Queue ({len(st.session_state.pending_queue)} words)**")
        st.dataframe(st.session_state.pending_queue, use_container_width=True)
        if st.button("☁️ Save Queue to Cloud", type="primary", use_container_width=True):
            new_df = pd.DataFrame(st.session_state.pending_queue)
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_df]).drop_duplicates(subset=['vocab'], keep='last')
            if save_to_github(st.session_state.vocab_df):
                st.session_state.pending_queue = []
                st.success("✅ Saved to GitHub!"); time.sleep(1); st.rerun()

with tab2:
    if st.session_state.vocab_df.empty: st.info("Cloud is empty.")
    else:
        st.subheader(f"✏️ Cloud Database ({len(st.session_state.vocab_df)})")
        search = st.text_input("🔎 Search Cloud...", "").lower()
        df_to_edit = st.session_state.vocab_df
        if search: df_to_edit = df_to_edit[df_to_edit['vocab'].str.contains(search)]
        
        edited_df = st.data_editor(df_to_edit, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("💾 Sync Changes", type="primary", use_container_width=True):
            st.session_state.vocab_df = edited_df
            save_to_github(edited_df)
            st.toast("✅ Cloud Updated!"); st.rerun()

with tab3:
    st.subheader("📇 Generate Deck")
    subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']
    if subset.empty: st.warning("No 'New' words to process.")
    else:
        st.info(f"Ready to process **{len(subset)}** words using **{SELECTED_THEME}** theme.")
        deck_name = st.text_input("Deck Name", "-English::Vocabulary")
        b_size = st.slider("Batch Size", 1, 10, 5)
        
        if st.button("🚀 Generate & Download", type="primary", use_container_width=True):
            notes = process_anki_data(subset, batch_size=b_size)
            if notes:
                pkg = create_anki_package(notes, deck_name, THEMES[SELECTED_THEME], accent_tld=ACCENTS[SELECTED_ACCENT])
                st.download_button("📥 Download .apkg", pkg, f"Vocab_{date.today()}.apkg", "application/octet-stream", use_container_width=True)
                
                # Mark as Done
                proc_vocabs = [n['VocabRaw'] for n in notes]
                st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(proc_vocabs), 'status'] = 'Done'
                save_to_github(st.session_state.vocab_df)
