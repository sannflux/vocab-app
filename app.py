import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import random
from datetime import date, datetime
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.ai.generativelanguage as glm
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
    st.error("⚠️ Missing libraries! Please run: pip install gTTS genanki")
    st.stop()

# ========================== CONSTANTS & SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

# --- ANKI & APP CONSTANTS ---
ANKI_MODEL_ID = 1607392319
ANKI_DECK_ID_SEED = 2059400110
RPM_DELAY_SECONDS = 12.1 # 5 RPM = 12s per request. Add a small buffer.
RPD_LIMIT = 20

# --- CSS VARIABLES ---
THEME_COLOR = "#00ff41"
THEME_GLOW = "rgba(0, 255, 65, 0.4)"
BG_COLOR = "#111111"
BG_STRIPE = "#181818"
TEXT_COLOR = "#aaffaa"

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
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

# ========================== GITHUB CONNECT & HELPERS ==========================
@st.cache_resource
def get_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except GithubException as e:
        st.error(f"❌ GitHub connection failed: {e}")
        st.stop()

repo = get_repo()

def update_github_file(path, commit_message, content_str):
    try:
        file = repo.get_contents(path)
        repo.update_file(file.path, commit_message, content_str, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file(path, commit_message, content_str)
        else:
            raise

def load_usage():
    try:
        file = repo.get_contents("usage.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return data.get("rpd_count", 0) if data.get("date") == str(date.today()) else 0
    except GithubException: # File not found
        return 0

def save_usage(count):
    data = json.dumps({"date": str(date.today()), "rpd_count": count})
    update_github_file("usage.json", "Update API usage", data)

# ========================== SESSION STATE INIT ==========================
def init_session_state():
    defaults = {
        "rpd_count": load_usage(),
        "vocab_df": load_data().copy(),
        "deck_id": ANKI_DECK_ID_SEED,
        "bulk_preview_df": None,
        "apkg_buffer": None,
        "processed_vocabs": [],
        "input_phrase": "",
        "input_vocab": "",
        "partial_results": [], # D1: Partial Progress Recovery
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ========================== GEMINI SETUP & JSON SCHEMA ==========================
# C2: JSON Schema Enforcement
CARD_SCHEMA = glm.Schema(
    type=glm.Type.OBJECT,
    properties={
        'vocab': glm.Schema(type=glm.Type.STRING),
        'phrase': glm.Schema(type=glm.Type.STRING),
        'translation': glm.Schema(type=glm.Type.STRING),
        'part_of_speech': glm.Schema(type=glm.Type.STRING, enum=["Noun", "Verb", "Adjective", "Adverb", "Pronoun", "Preposition", "Conjunction", "Interjection", "Phrase"]),
        'pronunciation_ipa': glm.Schema(type=glm.Type.STRING),
        'definition_english': glm.Schema(type=glm.Type.STRING),
        'example_sentences': glm.Schema(type=glm.Type.ARRAY, items=glm.Schema(type=glm.Type.STRING)),
        'synonyms_antonyms': glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                'synonyms': glm.Schema(type=glm.Type.ARRAY, items=glm.Schema(type=glm.Type.STRING)),
                'antonyms': glm.Schema(type=glm.Type.ARRAY, items=glm.Schema(type=glm.Type.STRING)),
            }
        ),
        'etymology': glm.Schema(type=glm.Type.STRING),
    }
)

@st.cache_resource
def get_gemini_model(api_key: str, model_name: str, system_prompt: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            # C1: System Instruction Hardening
            system_instruction=system_prompt,
            generation_config={"response_mime_type": "application/json", "response_schema": glm.Tool(function_declarations=[glm.FunctionDeclaration(name="output_card_array", description="Outputs an array of vocabulary cards", parameters=glm.Schema(type=glm.Type.OBJECT, properties={"cards": glm.Schema(type=glm.Type.ARRAY, items=CARD_SCHEMA)}))])},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except Exception as e:
        st.error(f"❌ Gemini key error: {e}")
        return None

# ========================== CLEANING & FORMATTING FUNCTIONS ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if not s or s[-1] in ".!?" else s + "."

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() if text else ""

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    pattern = r'\b' + re.escape(vocab) + r'\b'
    return re.sub(pattern, f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    pattern = r'\b' + re.escape(vocab.lower()) + r'\b'
    return re.sub(pattern, vocab, phrase, flags=re.IGNORECASE)

# ========================== AI BATCH GENERATOR (ENHANCED) ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    system_prompt = f"""You are an expert educational lexicographer. You create structured data for language learners. Output ONLY the function call with the JSON data.
SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.
RULES:
1. For each input item, generate one corresponding JSON object in the output array.
2. The user's 'phrase' is CRITICAL context. Prioritize the definition/nuance of the 'vocab' that best fits this phrase. (C3)
3. If 'phrase' is empty or starts with '*', generate a simple, common-use example sentence (max 12 words).
4. The 'vocab' word itself MUST be copied exactly.
5. The 'translation' field MUST contain ONLY the {TARGET_LANG} word-for-word translation of the 'vocab'. Do not translate the whole phrase.
6. The 'part_of_speech' MUST be one of the allowed enum values."""

    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL, system_prompt)
    if not model: return []

    all_card_data = st.session_state.partial_results.copy() # D1
    processed_inputs = {tuple(item) for d in all_card_data for item in d.get("input_batch_ref", [])}
    
    # Filter out already processed items
    remaining_items = [item for item in vocab_phrase_list if tuple(item) not in processed_inputs]
    
    if not remaining_items:
        st.info("All selected items were already processed in a previous run. Resuming...")
        return all_card_data

    batches = [remaining_items[i:i + batch_size] for i in range(0, len(remaining_items), batch_size)]
    
    with st.status("🤖 Processing AI Batches (RPM Throttled)...", expanded=True) as status_log:
        progress_bar = st.progress(0, text="Initializing...")
        
        for idx, batch in enumerate(batches):
            if st.session_state.rpd_count >= RPD_LIMIT:
                st.warning(f"🛑 Daily AI Limit ({RPD_LIMIT} requests) reached. Cannot process more batches.")
                break
            
            # A1: Hard-Coded Inter-Request Cooldown
            if idx > 0 or st.session_state.partial_results: # Delay if it's not the very first batch of a fresh run
                for remaining in range(int(RPM_DELAY_SECONDS), 0, -1):
                    progress_bar.progress((idx / len(batches)), text=f"⏳ RPM Safeguard Cooldown... ({remaining}s)")
                    time.sleep(1)

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            prompt = f"BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"
            vocab_words = [v[0] for v in batch]
            success = False
            
            for attempt in range(3):
                try:
                    status_log.update(label=f"🤖 Processing Batch {idx + 1}/{len(batches)}: `{', '.join(vocab_words)}`")
                    response = model.generate_content(prompt)
                    
                    st.session_state.rpd_count += 1
                    save_usage(st.session_state.rpd_count)
                    
                    # C2: Direct access, no parsing needed
                    parsed_data = response.candidates[0].content.parts[0].function_call.args['cards']
                    
                    if isinstance(parsed_data, list):
                        # D1: Add reference to input for recovery
                        for card in parsed_data:
                            card["input_batch_ref"] = batch
                        all_card_data.extend(parsed_data)
                        st.session_state.partial_results.extend(parsed_data) # D1
                        st.markdown(f"✅ **Batch {idx+1} OK**: `{', '.join(vocab_words)}`")
                        success = True
                        break
                except Exception as e:
                    st.warning(f"⚠️ Attempt {attempt+1} failed: {e}")
                    if "429" in str(e):
                        backoff = 20 + (2 ** attempt) + random.uniform(0, 1)
                        st.warning(f"   Rate Limit Hit. Backing off for {backoff:.1f}s...")
                        time.sleep(backoff)
                    else:
                        time.sleep(2)
            
            if not success:
                st.error(f"❌ **Failed & Skipped Batch**: `{', '.join(vocab_words)}` (Preserving quota)")
                # A3: Persistent Fail-Safe Log
                failed_log_content = json.dumps({"failed_batch": batch_dicts, "timestamp": str(datetime.now())}, indent=2)
                update_github_file("failed_batches.json", "Log failed batch", failed_log_content)
            
            progress_bar.progress((idx + 1) / len(batches), text=f"Batch {idx + 1}/{len(batches)} Complete")

        status_log.update(label=f"✅ AI Generation Complete! ({len(all_card_data)} items)", state="complete", expanded=False)
    
    st.session_state.partial_results = [] # Clear recovery state after successful full run
    return all_card_data

# ========================== ANKI DATA PROCESSING ==========================
def process_anki_data(df_subset, batch_size=6):
    df_subset = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []

    for card_data in all_card_data:
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue

        vocab_cap = cap_first(vocab_raw)
        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing(phrase, vocab_raw)
        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""
        
        translation = ensure_trailing_dot(normalize_spaces(card_data.get("translation", "?")))
        pos = str(card_data.get("part_of_speech", "")).title()
        ipa = card_data.get("pronunciation_ipa", "")
        eng_def = ensure_trailing_dot(cap_first(normalize_spaces(card_data.get("definition_english", ""))))
        
        examples = [ensure_trailing_dot(cap_first(normalize_spaces(e))) for e in (card_data.get("example_sentences", []) or [])[:3]]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""
        
        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ensure_trailing_dot(", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]]))
        antonyms_field = ensure_trailing_dot(", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])[:5]]))
        etymology = normalize_spaces(card_data.get("etymology", ""))
        
        text_field = f"{formatted_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>" if formatted_phrase else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"
        
        tags = [f"Gen_{date.today().strftime('%Y-%m-%d')}"] # B2: Automatic Tagging Engine
        
        processed_notes.append({
            "VocabRaw": vocab_raw, "Text": text_field, "Pronunciation": pronunciation_field,
            "Definition": eng_def, "Examples": examples_field, "Synonyms": synonyms_field,
            "Antonyms": antonyms_field, "Etymology": etymology, "Tags": tags
        })
    return processed_notes

# ========================== AUDIO & ANKI PACKAGE CREATION ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        clean_vocab = re.sub(r'[^a-zA-Z0-9\s\-\']', '', vocab).strip()
        if not clean_vocab: return vocab, None, None
        
        # Consistent filename hash
        filename_hash = hashlib.sha1(clean_vocab.encode('utf-8')).hexdigest()[:10]
        clean_filename = f"tts_{filename_hash}_{clean_vocab.replace(' ', '_')}.mp3"
        file_path = os.path.join(temp_dir, clean_filename)

        tts = gTTS(text=clean_vocab, lang='en', slow=False)
        tts.save(file_path)
        return vocab, clean_filename, file_path
    except Exception as e:
        print(f"Audio error for {vocab}: {e}")
        return vocab, None, None

CYBERPUNK_CSS = f"""
.card {{ font-family: 'Roboto Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; color: {THEME_COLOR}; background-color: {BG_COLOR}; background-image: repeating-linear-gradient(0deg, {BG_STRIPE}, {BG_STRIPE} 1px, {BG_COLOR} 1px, {BG_COLOR} 20px); padding: 30px 20px; text-align: left; }}
.vellum-focus-container {{ background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px; border: 2px solid {THEME_COLOR}; box-shadow: 0 0 5px {THEME_COLOR}, 0 0 15px {THEME_GLOW}; text-align: center; }}
.prompt-text {{ font-family: 'Electrolize', sans-serif; font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }}
.cloze {{ color: {BG_COLOR}; background-color: {THEME_COLOR}; padding: 2px 4px; }}
.solved-text .cloze {{ color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }}
.vellum-section {{ margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed {THEME_COLOR}; }}
.section-header {{ font-weight: 600; color: #00ffff; border-left: 3px solid {THEME_COLOR}; padding-left: 10px; }}
.content {{ color: {TEXT_COLOR}; padding-left: 13px; }}
@media (max-width: 600px) {{ /* B3: CSS Media Query Support */
    .card {{ font-size: 16px; padding: 15px 10px; }}
    .prompt-text {{ font-size: 1.5em; }}
}}
"""

def create_anki_package(notes_data, deck_name, generate_audio=True, deck_id=ANKI_DECK_ID_SEED):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""
    
    # B1: Static Note Type IDs
    my_model = genanki.Model(
        ANKI_MODEL_ID, 'Cyberpunk Vocab Model v2',
        fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE
    )
    my_deck = genanki.Deck(deck_id, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with st.spinner(f"🔊 Generating {len(unique_vocabs)} audio files..."):
                # D2: Audio Generation Concurrency
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_vocab = {}
                    for v in unique_vocabs:
                        future = executor.submit(generate_audio_file, v, temp_dir)
                        future_to_vocab[future] = v
                        time.sleep(0.05) # Stagger submissions
                    for future in concurrent.futures.as_completed(future_to_vocab):
                        vk, fn, fp = future.result()
                        if fn and fp:
                            media_files.append(fp)
                            audio_map[vk] = f"[sound:{fn}]"
        
        for note_data in notes_data:
            # B4: Guid Hash Collision Check
            guid_seed = f"{note_data['VocabRaw']}_{deck_name}"
            vocab_hash = str(int(hashlib.sha256(guid_seed.encode('utf-8')).hexdigest(), 16) % (10**10))
            
            my_deck.add_note(genanki.Note(
                model=my_model,
                fields=[note_data['Text'], note_data['Pronunciation'], note_data['Definition'], note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'], note_data['Etymology'], audio_map.get(note_data['VocabRaw'], "")],
                tags=note_data['Tags'],
                guid=vocab_hash
            ))
            
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buffer = io.BytesIO()
        output_path = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(output_path)
        with open(output_path, "rb") as f: buffer.write(f.read())
        buffer.seek(0)
    return buffer

# ========================== DATA LOAD/SAVE ==========================
@st.cache_data(ttl=300)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str)
        df['phrase'] = df['phrase'].fillna("")
        df['status'] = df.get('status', 'New')
        df['tags'] = df.get('tags', '')
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags'])
        st.error(f"Failed to load data from GitHub: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        st.stop()

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        update_github_file("vocabulary.csv", "Updated vocab", csv_data)
    except Exception as e:
        st.error(f"Failed to save to GitHub: {e}")
        return False
    load_data.clear()
    return True

# Initialize state before using it
init_session_state()

# ========================== CALLBACKS ==========================
def mark_as_done_callback():
    if "processed_vocabs" in st.session_state and st.session_state.processed_vocabs:
        st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(st.session_state.processed_vocabs), 'status'] = 'Done'
        save_to_github(st.session_state.vocab_df)
    st.session_state.apkg_buffer = None
    st.session_state.processed_vocabs = []
    st.session_state.partial_results = []

def save_single_word_callback():
    v = st.session_state.input_vocab.lower().strip()
    p = st.session_state.input_phrase.strip()
    if not v:
        st.error("⚠️ Vocabulary word cannot be empty.")
        return

    mask = st.session_state.vocab_df['vocab'] == v
    if not st.session_state.vocab_df.empty and mask.any():
        st.session_state.vocab_df.loc[mask, ['phrase', 'status']] = [p, 'New']
    else:
        new_row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "tags": ""}])
        st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
    
    if save_to_github(st.session_state.vocab_df):
        st.toast(f"✅ Saved '{v}'!", icon="🚀")
        st.session_state.input_phrase = ""
        st.session_state.input_vocab = ""
    else:
        st.error("Failed to save. Check GitHub connection.")

# ========================== SIDEBAR DASHBOARD ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    total_words = len(st.session_state.vocab_df)
    new_words = len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'])
    
    col1, col2 = st.columns(2)
    col1.metric("📖 Total", total_words)
    col2.metric("✨ New", new_words)

    st.subheader("🤖 Daily AI Usage")
    st.progress(st.session_state.rpd_count / RPD_LIMIT, text=f"{st.session_state.rpd_count}/{RPD_LIMIT} Requests")
    
    st.divider()
    TARGET_LANG = st.selectbox("🎯 Target Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    # Model selection remains unchanged per oath
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-1.5-flash", "gemini-1.5-pro-latest"], index=0, help="Model strings are preserved as per system directive.")
    st.divider()
    if not st.session_state.vocab_df.empty:
        st.download_button("💾 Backup Database (CSV)", st.session_state.vocab_df.to_csv(index=False).encode('utf-8'), f"vocab_backup_{date.today()}.csv", "text/csv")

# ========================== UI TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    st.subheader("Add New Word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")

    if add_mode == "Single":
        st.text_input("🔤 Phrase (Context)", placeholder="Paste a sentence here...", key="input_phrase")
        st.text_input("📝 Vocabulary", placeholder="e.g., serendipity", key="input_vocab")
        
        v_check = st.session_state.input_vocab.lower().strip()
        if v_check and not st.session_state.vocab_df.empty and (st.session_state.vocab_df['vocab'] == v_check).any():
            st.warning(f"⚠️ '{v_check}' exists. Saving will overwrite its phrase and reset status to 'New'.")
            
        st.button("💾 Save to Cloud", type="primary", use_container_width=True, on_click=save_single_word_callback)

    else: # Bulk mode
        bulk_text = st.text_area("Paste List (one 'word' or 'word, phrase' per line)", height=150, key="bulk_input")
        # D5: Bulk Import Validator
        is_valid_bulk_text = any(re.match(r'^\s*\w+.*', line) for line in bulk_text.split('\n'))
        
        if st.button("Preview Bulk Import", disabled=not is_valid_bulk_text):
            lines = [l.strip() for l in bulk_text.split('\n') if l.strip()]
            new_rows = []
            for line in lines:
                parts = line.split(',', 1)
                bv = parts[0].strip().lower()
                bp = parts[1].strip() if len(parts) > 1 else ""
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "status": "New", "tags": ""})
            if new_rows: st.session_state.bulk_preview_df = pd.DataFrame(new_rows)

        if st.session_state.bulk_preview_df is not None:
            st.write("### Preview:")
            st.dataframe(st.session_state.bulk_preview_df, hide_index=True)
            if st.button("💾 Confirm & Save Bulk", type="primary"):
                st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, st.session_state.bulk_preview_df]).drop_duplicates(subset=['vocab'], keep='last', ignore_index=True)
                if save_to_github(st.session_state.vocab_df):
                    st.toast(f"✅ Added/updated {len(st.session_state.bulk_preview_df)} words!", icon="🎉")
                    st.session_state.bulk_preview_df = None
                    time.sleep(0.5)
                    st.rerun()

with tab2:
    if st.session_state.vocab_df.empty: st.info("Database is empty. Add words in the '➕ Add' tab.")
    else:
        st.subheader(f"✏️ Edit Database ({len(st.session_state.vocab_df)} words)")
        search = st.text_input("🔎 Search...", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"status": st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True)})
        
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            st.session_state.vocab_df = edited
            if save_to_github(st.session_state.vocab_df):
                st.toast("✅ Cloud database updated!")
                time.sleep(0.5)
                st.rerun()

with tab3:
    st.subheader("📇 Generate Anki Deck")
    
    if st.session_state.apkg_buffer is not None:
        st.success("✅ Deck generated successfully!")
        st.download_button("📥 Download .apkg & Mark as Done", data=st.session_state.apkg_buffer, file_name=f"AnkiDeck_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg", mime="application/octet-stream", use_container_width=True, on_click=mark_as_done_callback)
        if st.button("❌ Cancel / Clear"):
            st.session_state.apkg_buffer = None
            st.session_state.processed_vocabs = []
            st.session_state.partial_results = []
            st.rerun()
    else:
        if st.session_state.vocab_df.empty:
            st.info("Add words to your database first.")
        else:
            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
            if subset.empty:
                st.warning("⚠️ No words marked as 'New' available for export.")
            else:
                deck_col1, deck_col2 = st.columns([3, 1])
                deck_name_input = deck_col1.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
                if deck_col2.button("🎲 New Deck ID"): st.session_state.deck_id = random.randrange(1 << 30, 1 << 31)
                deck_col2.caption(f"ID: {st.session_state.deck_id}")

                batch_size = st.slider("⚡ Batch Size (Words per Request)", 1, 10, 6)
                include_audio = st.checkbox("🔊 Generate Audio Files (slower)", value=True)
                
                st.write("**Select words to export:**")
                subset['Export'] = True
                edited_export = st.data_editor(subset, column_config={"Export": st.column_config.CheckboxColumn("Export?", required=True)}, hide_index=True, disabled=["vocab", "phrase", "status", "tags"])
                final_export_subset = edited_export[edited_export['Export'] == True]
                
                required_requests = math.ceil(len(final_export_subset) / batch_size) if not final_export_subset.empty else 0
                requests_left = max(0, RPD_LIMIT - st.session_state.rpd_count)
                
                # A2: Smart Token/RPD Reservation
                st.info(f"You have **{requests_left}** requests left today. This batch requires **{required_requests}** requests.")
                
                can_proceed = (not final_export_subset.empty) and (required_requests <= requests_left)
                
                if st.button("🚀 Generate Deck", type="primary", use_container_width=True, disabled=not can_proceed):
                    raw_notes = process_anki_data(final_export_subset, batch_size=batch_size)
                    if raw_notes:
                        apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio, deck_id=st.session_state.deck_id)
                        st.session_state.apkg_buffer = apkg_buffer.getvalue()
                        st.session_state.processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                        st.rerun()
                
                if final_export_subset.empty: st.warning("Select at least one word to export.")
                elif required_requests > requests_left: st.error("🛑 Exceeds Daily Limit! Reduce your selection or increase batch size.")
