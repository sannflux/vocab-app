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

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 1. Target Language
    TARGET_LANG = st.selectbox(
        "🎯 Definition Language", 
        ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"],
        index=0
    )
    
    # 2. Model Selection (UNCHANGED)
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
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

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

# ========================== CLEANING FUNCTIONS ==========================
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
    rules = [
        (r"\bto doing\b", "to do"), (r"\bfor helps\b", "to help"),
        (r"\bis use to\b", "is used to"), (r"\bhelp for to\b", "help to"),
        (r"\bfor to\b", "to"), (r"\bcan able to\b", "can")
    ]
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

# ========================== SPEECH (Browser) ==========================
def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== BATCH GENERATOR ==========================
def robust_json_parse(text: str):
    try:
        return json.loads(text)
    except:
        pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model:
        return []

    all_card_data = []
    progress_bar = st.progress(0)
    total_items = len(vocab_phrase_list)

    for i in range(0, total_items, batch_size):
        progress_bar.progress(i / total_items, text=f"🤖 AI Processing {i}/{total_items} words...")
        batch = vocab_phrase_list[i:i + batch_size]
        # batch is list of tuples: (vocab, phrase)
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]

        # UPDATE: Improved Prompt with Context Hint Logic
        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES: 
1. Copy ALL fields exactly. 
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT (e.g. phrase='*bird' for vocab='crane'). Use this hint to pick the specific definition, but generate a NEW sentence for the final 'phrase' field.
3. IF 'phrase' is normal text: Define based on that usage.
4. IF 'phrase' is empty: Generate ONE simple sentence (max 12 words) using the most common definition.
5. EXACT vocab unchanged.
NEVER use markdown, asterisks, bold, italic, or any formatting. Plain text only.
OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "Plain text only."}}]
BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

        # UPDATE: Robust Retry with Exponential Backoff
        for attempt in range(5):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    break
            except Exception as e:
                wait_time = (2 ** attempt) + 1
                # Only print to console to keep UI clean, or use toast
                print(f"⚠️ API Error (Attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        else:
             st.error(f"❌ Batch failed after 5 attempts. Skipping batch starting with: {batch[0][0]}")

    progress_bar.progress(1.0, text="✅ AI Generation Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    return all_card_data

def process_anki_data(df_subset, batch_size=6):
    # UPDATE: Filter out garbage data before sending to API
    df_subset = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []

    for card_data in all_card_data:
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        vocab_cap = cap_first(vocab_raw)

        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
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

        text_field = f"{formatted_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>" if formatted_phrase else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"

        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"

        processed_notes.append({
            "VocabRaw": vocab_raw,
            "Text": text_field, 
            "Pronunciation": pronunciation_field, 
            "Definition": eng_def,
            "Examples": examples_field, 
            "Synonyms": synonyms_field, 
            "Antonyms": antonyms_field, 
            "Etymology": etymology
        })
    return processed_notes

# ========================== AUDIO HELPER (Parallel) ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        file_path = os.path.join(temp_dir, clean_filename)
        # Only generate if not empty
        if vocab.strip():
            tts = gTTS(text=vocab, lang='en', slow=False)
            tts.save(file_path)
            return vocab, clean_filename, file_path
    except Exception as e:
        print(f"Audio error for {vocab}: {e}")
    return vocab, None, None

# ========================== GENANKI LOGIC ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True):
    
    # --- CYBERPUNK CSS ---
    cyberpunk_css = """
/* --- Global Settings (Cyberpunk Glitch Theme) --- */
.card { font-family: 'Roboto Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; color: #00ff41; background-color: #111111; background-image: repeating-linear-gradient(0deg, #181818, #181818 1px, #111111 1px, #111111 20px); padding: 30px 20px; text-align: left; }
.nightMode .card { color: #00aaff; background-color: #080808; }
.vellum-focus-container { background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px; border: 2px solid #00ff41; box-shadow: 0 0 5px #00ff41, 0 0 15px rgba(0, 255, 65, 0.4); text-align: center; }
.nightMode .vellum-focus-container { border: 2px solid #00aaff; box-shadow: 0 0 5px #00aaff, 0 0 15px rgba(0, 170, 255, 0.4); }
.prompt-text { font-family: 'Electrolize', sans-serif; font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }
.cloze { color: #111111; background-color: #00ff41; padding: 2px 4px; }
.back .cloze { color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }
.vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed #00ff41; }
.section-header { font-weight: 600; color: #00ffff; border-left: 3px solid #00ff41; padding-left: 10px; }
.content { color: #aaffaa; padding-left: 13px; }
"""

    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div>
<div class="vellum-detail-container">
  {{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}
  {{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}
  {{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}
  {{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}
  {{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}
  <div style='display:none'>{{Audio}}</div>
</div>{{Audio}}"""

    # Model & Deck Setup
    my_model = genanki.Model(
        1607392319, 'Cyberpunk Vocab Model',
        fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=cyberpunk_css,
        model_type=genanki.Model.CLOZE 
    )
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        
        # --- PARALLEL AUDIO GENERATION ---
        if generate_audio:
            progress_bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Deduplicate vocabs to save time
                unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
                future_to_vocab = {executor.submit(generate_audio_file, v, temp_dir): v for v in unique_vocabs}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_vocab)):
                    vocab_key, fname, fpath = future.result()
                    if fname and fpath:
                        media_files.append(fpath)
                        audio_map[vocab_key] = f"[sound:{fname}]"
                    progress_bar.progress((i + 1) / len(unique_vocabs), text=f"🔊 Generating Audio: {vocab_key}...")
            progress_bar.empty()

        # Add Notes to Deck
        for note_data in notes_data:
            audio_tag = audio_map.get(note_data['VocabRaw'], "")
            
            my_note = genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'], note_data['Pronunciation'], note_data['Definition'],
                    note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'],
                    note_data['Etymology'], audio_tag
                ]
            )
            my_deck.add_note(my_note)
        
        # Package
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buffer = io.BytesIO()
        output_path = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(output_path)
        with open(output_path, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        return buffer

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        
        # Ensure 'status' column exists for tracking
        if 'status' not in df.columns:
            df['status'] = 'New'
            
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=['vocab', 'phrase', 'status'])
        else:
            st.error(f"❌ CRITICAL: GitHub Error {e.status}. Cannot load data.")
            st.stop()
    except Exception as e:
        st.error(f"❌ CRITICAL: CSV Error. {e}")
        st.stop()

def save_to_github(dataframe):
    # UPDATE: Remove empty vocabs and drop duplicates to prevent bugs
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=['vocab'], keep='last')
    
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear()
    return True

df = load_data().copy()

# ========================== WORD OF THE DAY ==========================
with st.sidebar:
    st.divider()
    st.header("🌟 Word of the Day")
    if not df.empty:
        today_str = date.today().isoformat()
        random.seed(today_str)
        # Handle cases where sampling might fail on tiny dataframes
        try:
            row = df.sample(n=1).iloc[0]
            wotd_vocab = row["vocab"]
            wotd_phrase = row["phrase"]
            
            st.subheader(wotd_vocab.upper())
            if wotd_phrase: st.caption(wotd_phrase)
            
            if st.button("🔊 Pronounce"):
                speak_word(wotd_vocab)
        except:
            pass
    else:
        st.info("No words yet!")

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    st.subheader("Add new word")
    st.caption("Tip: Start phrase with `*` to give a context hint (e.g. `*bird` for 'crane')")
    with st.form("add_form", clear_on_submit=True):
        v = st.text_input("📝 Vocab", placeholder="e.g. serendipity").lower().strip()
        p_raw = st.text_input("🔤 Phrase", placeholder="I found it by serendipity! (or type '1' to skip)").strip()
        
        exists = False
        if v and not df.empty and v in df['vocab'].values:
            exists = True
            st.warning(f"⚠️ '{v}' already exists.")
            
        submitted = st.form_submit_button("💾 Save to Cloud", use_container_width=True)

    if submitted:
        if v:
            if p_raw == "1":
                p = ""
            elif p_raw.startswith("*"):
                p = p_raw # Keep context hints as is
            else:
                p = p_raw.capitalize()
                
            if exists:
                df.loc[df['vocab'] == v, 'phrase'] = p
                df.loc[df['vocab'] == v, 'status'] = 'New' # Reset status if updated
                if save_to_github(df): 
                    st.success(f"✅ Phrase for '{v}' updated!")
                    time.sleep(1); st.rerun()
            else:
                new_row = {"vocab": v, "phrase": p, "status": "New"}
                updated = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                if save_to_github(updated): 
                    st.success(f"✅ '{v}' added!"); time.sleep(1); st.rerun()

with tab2:
    if df.empty: 
        st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        
        c1, c2 = st.columns([2, 1])
        with c1: search = st.text_input("🔎 Search...", "").lower().strip()
        with c2: 
            filter_new = st.checkbox("Show 'New' only")
        
        display_df = df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        if filter_new: display_df = display_df[display_df['status'] == 'New']
        
        edited = st.data_editor(
            display_df, 
            num_rows="dynamic", 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    help="Track which words are done",
                    width="small",
                    options=["New", "Done"],
                    required=True,
                )
            }
        )
        
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                # We need to merge edited rows back into main DF carefully
                # Simplified strategy: If no filters, replace all. If filtered, update matching indices.
                if search or filter_new:
                    # Update rows in main df based on vocab match
                    for index, row in edited.iterrows():
                        mask = df['vocab'] == row['vocab']
                        df.loc[mask, ['phrase', 'status']] = [row['phrase'], row['status']]
                    final_save = df
                else:
                    final_save = edited
                
                if save_to_github(final_save.sort_values(by="vocab", ignore_index=True)):
                    st.toast("✅ Cloud updated!", icon="🎉"); st.rerun()

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")
    
    if df.empty:
        st.info("Add words first!")
    else:
        col_new, col_all = st.columns(2)
        new_count = len(df[df['status'] == 'New'])
        col_new.metric("New Words", new_count)
        col_all.metric("Total Words", len(df))

        st.divider()
        
        deck_name_input = st.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        
        c1, c2, c3 = st.columns(3)
        with c1: batch_size = st.slider("⚡ Batch Size", 1, 10, 5)
        with c2: include_audio = st.checkbox("🔊 Audio", value=True)
        with c3: process_only_new = st.checkbox("Only process 'New'", value=True)

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            
            # Filter Data
            subset = df[df['status'] == 'New'] if process_only_new else df
            
            if subset.empty:
                st.warning("⚠️ No words to process!")
            else:
                raw_notes = process_anki_data(subset, batch_size=batch_size)
                
                if not raw_notes:
                    st.error("❌ Generation failed. Check API Key.")
                else:
                    with st.spinner("📦 Packaging Deck..."):
                        apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    filename = f"AnkiDeck_{timestamp}.apkg"
                    
                    st.success(f"🎉 {len(raw_notes)} cards ready!")
                    st.download_button("📥 Download .apkg", apkg_buffer, filename, "application/octet-stream", use_container_width=True)
                    
                    # Auto-update status to 'Done'
                    if process_only_new:
                        df.loc[df['vocab'].isin(subset['vocab']), 'status'] = 'Done'
                        save_to_github(df)
                        st.caption("✅ Marked processed words as 'Done' in cloud.")

st.caption("✅ Cyberpunk Anki deck • Phrase + Vocab: {{c1::Translation}}")
