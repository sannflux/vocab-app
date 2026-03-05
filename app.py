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
import os
import tempfile

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

token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["REPO_NAME"]
DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.last_key = None

try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

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

# ========================== CLEANING FUNCTIONS ==========================
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

def fix_vocab_casing_in_phrase(phrase, vocab_raw):
    if not phrase or not vocab_raw: return phrase
    pattern = r'\b' + re.escape(vocab_raw.lower()) + r'\b'
    return re.sub(pattern, vocab_raw, phrase, flags=re.IGNORECASE)

# ========================== BATCH GENERATOR ==========================
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
    
    progress_bar = st.progress(0)
    total_items = len(vocab_phrase_list)
    
    for i in range(0, total_items, batch_size):
        current_progress = min(i / total_items, 1.0)
        progress_bar.progress(current_progress, text=f"🤖 AI Processing {i}/{total_items} words...")
        
        batch = vocab_phrase_list[i:i+batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        
        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES: 
1. Copy ALL fields exactly. 
2. If phrase is provided: Define the vocab word based STRICTLY on how it is used in that phrase.
3. If phrase is empty: generate ONE simple sentence (max 12 words) using the most common definition.
4. EXACT vocab unchanged.
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
        
    progress_bar.progress(1.0, text="AI Generation Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    return all_card_data

def process_anki_data(df, batch_size=5):
    vocab_phrase_list = df[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []
    
    for card_data in all_card_data:
        vocab_raw = (card_data.get("vocab", "") or "").strip()
        vocab_cap = cap_first(vocab_raw)
        
        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing_in_phrase(phrase, vocab_raw)
        
        # RESTORED ORIGINAL LOGIC: Phrase at top, Vocab: {{c1::Translation}}
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
        
        # KEY FIX: Using the format "Phrase<br><br>Vocab: {{c1::Translation}}"
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

# ========================== GENANKI & AUDIO LOGIC ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True):
    
    # --- CYBERPUNK CSS ---
    cyberpunk_css = """
/* --- Global Settings (Cyberpunk Glitch Theme) --- */
.card {
  font-family: 'Roboto Mono', 'Consolas', monospace;
  font-size: 18px;
  line-height: 1.5;
  font-weight: 400;
  color: #00ff41; 
  background-color: #111111; 
  background-image: repeating-linear-gradient(0deg, #181818, #181818 1px, #111111 1px, #111111 20px);
  padding: 30px 20px;
  max-width: 800px;
  margin: 0 auto;
  box-sizing: border-box;
  text-align: left;
}

.nightMode .card {
  color: #00aaff;
  background-color: #080808;
}

/* --- UNIFIED FOCUS CONTAINER --- */
.vellum-focus-container {
  background: #0d0d0d;
  padding: 30px 20px;
  margin: 0 auto 40px;
  max-width: 95%;
  border-radius: 4px; 
  border: 2px solid #00ff41; 
  box-shadow: 0 0 5px #00ff41, 0 0 15px rgba(0, 255, 65, 0.4), 0 4px 8px rgba(0, 0, 0, 0.5);
  text-align: center;
  position: relative;
  overflow: hidden;
}

.nightMode .vellum-focus-container {
  border: 2px solid #00aaff;
  box-shadow: 0 0 5px #00aaff, 0 0 15px rgba(0, 170, 255, 0.4), 0 4px 8px rgba(0, 0, 0, 0.6);
}

.prompt-text {
  font-family: 'Electrolize', 'Arial Narrow', sans-serif;
  font-size: clamp(1.5em, 5vw, 2.0em); /* Adjusted size for Phrase + Vocab */
  font-weight: 900;
  color: #ffffff;
  text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff;
  font-style: normal;
}

.nightMode .prompt-text {
  color: #f0f8ff; 
  text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff;
}

/* --- SPECIFIC STYLING FOR FRONT (Hidden Cloze) --- */
.vellum-focus-container.front .cloze {
  color: #111111;
  background-color: #00ff41;
  border-radius: 2px;
  padding: 2px 4px;
  line-height: 1;
  text-decoration: none;
  font-style: normal;
}

.nightMode .vellum-focus-container.front .cloze {
  background-color: #00aaff;
  color: #0d0d0d;
}

/* --- SPECIFIC STYLING FOR BACK (Solved Cloze) --- */
.vellum-focus-container.back .prompt-text {
  color: #e0e0e0; 
}

.vellum-focus-container.back .cloze {
  color: #ff00ff;
  font-weight: 900;
  background: none;
  padding: 0 3px;
  text-decoration: none;
  border-bottom: 3px double #00ffff;
  text-shadow: 0 0 5px #ff00ff;
  font-style: normal;
}

.nightMode .vellum-focus-container.back .cloze {
  color: #00ffff;
  border-bottom: 3px double #ff00ff;
  text-shadow: 0 0 5px #00ffff;
}

/* --- DETAIL SECTIONS --- */
.vellum-detail-container { padding: 10px 0; }
.vellum-section {
  margin: 15px 0;
  padding: 10px 0;
  border-bottom: 1px dashed #00ff41; 
  padding-left: 5px;
  padding-right: 5px;
}
.nightMode .vellum-section { border-bottom: 1px dashed #00aaff; }

.section-header {
  font-size: 1.1em;
  font-weight: 600;
  margin-bottom: 8px;
  color: #00ffff;
  display: flex;
  align-items: center;
  gap: 8px;
  border-left: 3px solid; 
  padding-left: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.nightMode .section-header { color: #ff00ff; }

.content {
  font-size: 0.95em;
  color: #aaffaa; 
  padding: 0 0 0 13px; 
}
.nightMode .content { color: #99ccff; }

/* Markers */
.vellum-section.definition .section-header { border-left-color: #00ff41; } 
.vellum-section.pronunciation .section-header { border-left-color: #ff00ff; }
.vellum-section.examples .section-header { border-left-color: #ffff00; } 
.vellum-section.synonyms .section-header { border-left-color: #ff00ff; } 
.vellum-section.antonyms .section-header { border-left-color: #ff4100; } 
.vellum-section.etymology .section-header { border-left-color: #00ffff; } 

.pronunciation .content {
  font-family: 'Consolas', monospace;
  font-size: 1.05em;
  font-weight: 500;
  color: #ffff00; 
}
.examples .content {
  color: #77ff77;
  font-style: italic;
  font-size: 0.9em;
}

@media (max-width: 480px) {
  .card { font-size: 16px; padding: 15px 10px; }
  .vellum-focus-container { padding: 20px 10px; max-width: 100%; }
}
"""

    # --- HTML TEMPLATES ---
    front_html = """
<div class="vellum-focus-container front">
  <div class="prompt-text">
    {{cloze:Text}}
  </div>
</div>
"""

    back_html = """
<div class="vellum-focus-container back">
  <div class="prompt-text solved-text">
    {{cloze:Text}}
  </div>
</div>

<div class="vellum-detail-container">
  {{#Definition}}
  <div class="vellum-section definition">
    <div class="section-header">📜 DEFINITION</div>
    <div class="content">{{Definition}}</div>
  </div>
  {{/Definition}}
  
  {{#Pronunciation}}
  <div class="vellum-section pronunciation">
    <div class="section-header">🗣️ PRONUNCIATION</div>
    <div class="content">{{Pronunciation}}</div>
  </div>
  {{/Pronunciation}}

  {{#Examples}}
  <div class="vellum-section examples">
    <div class="section-header">🖋️ EXAMPLES</div>
    <div class="content">{{Examples}}</div>
  </div>
  {{/Examples}}

  {{#Synonyms}}
  <div class="vellum-section synonyms">
    <div class="section-header">➕ SYNONYMS</div>
    <div class="content">{{Synonyms}}</div>
  </div>
  {{/Synonyms}}

  {{#Antonyms}}
  <div class="vellum-section antonyms">
    <div class="section-header">➖ ANTONYMS</div>
    <div class="content">{{Antonyms}}</div>
  </div>
  {{/Antonyms}}

  {{#Etymology}}
  <div class="vellum-section etymology">
    <div class="section-header">🏛️ ETYMOLOGY</div>
    <div class="content">{{Etymology}}</div>
  </div>
  {{/Etymology}}
  
  <div style='display:none'>{{Audio}}</div>
</div>
{{Audio}}
"""

    # 1. Define Anki Model (CLOZE)
    model_id = 1607392325 # Unique ID
    my_model = genanki.Model(
        model_id,
        'Cyberpunk Vocab Model',
        fields=[
            {'name': 'Text'},
            {'name': 'Pronunciation'},
            {'name': 'Definition'},
            {'name': 'Examples'},
            {'name': 'Synonyms'},
            {'name': 'Antonyms'},
            {'name': 'Etymology'},
            {'name': 'Audio'},
        ],
        templates=[{
            'name': 'Cyberpunk Card',
            'qfmt': front_html,
            'afmt': back_html,
        }],
        css=cyberpunk_css,
        model_type=genanki.Model.CLOZE 
    )

    # 2. Define Deck
    deck_id = abs(hash(deck_name)) % (10 ** 10)
    my_deck = genanki.Deck(deck_id, deck_name)
    
    media_files = []
    
    # 3. Create Temp Directory for MP3s
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_bar = st.progress(0)
        total = len(notes_data)
        
        for i, note_data in enumerate(notes_data):
            if generate_audio:
                progress_bar.progress((i / total), text=f"🔊 Generating Audio: {note_data['VocabRaw']}...")
            
            audio_field = ""
            
            if generate_audio and note_data['VocabRaw']:
                try:
                    clean_filename = re.sub(r'[^a-zA-Z0-9]', '', note_data['VocabRaw']) + ".mp3"
                    file_path = os.path.join(temp_dir, clean_filename)
                    tts = gTTS(text=note_data['VocabRaw'], lang='en', slow=False)
                    tts.save(file_path)
                    media_files.append(file_path)
                    audio_field = f"[sound:{clean_filename}]"
                except Exception as e:
                    print(f"Audio error for {note_data['VocabRaw']}: {e}")

            my_note = genanki.Note(
                model=my_model,
                fields=[
                    note_data['Text'],
                    note_data['Pronunciation'],
                    note_data['Definition'],
                    note_data['Examples'],
                    note_data['Synonyms'],
                    note_data['Antonyms'],
                    note_data['Etymology'],
                    audio_field
                ]
            )
            my_deck.add_note(my_note)
        
        progress_bar.progress(1.0, text="📦 Packaging Deck...")
        
        # 4. Create Package
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        
        buffer = io.BytesIO()
        output_path = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(output_path)
        
        with open(output_path, "rb") as f:
            buffer.write(f.read())
            
        buffer.seek(0)
        progress_bar.empty()
        return buffer

# ========================== LOAD / SAVE / SPEECH / WOTD ==========================
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
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
    wotd_phrase = row["phrase"]

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("🌟 Word of the Day")
    if wotd_vocab:
        st.subheader(wotd_vocab.upper())
        if wotd_phrase.strip(): st.caption(wotd_phrase)
    st.divider()
    st.subheader("🔑 Gemini API Key")
    alt_key = st.text_input("Alternative key", type="password", value="", placeholder="AIzaSy...")
    if alt_key and alt_key != st.session_state.gemini_key:
        st.session_state.gemini_key = alt_key
        st.success("✅ Switched!")
        st.rerun()
    st.divider()
    lang_options = {"🇬🇧 English (US)": "en-US", "🇮🇩 Indonesian": "id-ID", "🇯🇵 Japanese": "ja-JP"}
    selected_lang_name = st.selectbox("🎙️ Speech Language", list(lang_options.keys()), index=0)
    speech_lang = lang_options[selected_lang_name]
    if wotd_vocab:
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("🔊 Vocab", key="wotd_v"): speak_word(wotd_vocab, speech_lang)
        with c2: 
            if wotd_phrase.strip() and st.button("🔊 Phrase", key="wotd_p"): speak_word(wotd_phrase, speech_lang)

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Generate Anki (Cyberpunk)"])

with tabs[0]:  # This corresponds to your "➕ Add" Tab
    st.header("Add New Vocabulary")

    # 1. Initialize Session State for inputs if they don't exist
    if "new_vocab" not in st.session_state:
        st.session_state["new_vocab"] = ""
    if "new_phrase" not in st.session_state:
        st.session_state["new_phrase"] = ""

    # 2. Bind the inputs to these session state keys
    # Note: We remove the 'value' logic and rely on 'key'
    vocab_input = st.text_input("Vocab", key="new_vocab")
    phrase_input = st.text_input("Phrase", key="new_phrase")

    # Real-time validation (as per your original doc)
    if vocab_input and vocab_input in df["Vocab"].values:
        st.warning(f"⚠️ '{vocab_input}' already exists!")
        # You might show an 'Update' button here, but let's focus on the 'Save' flow

    # 3. The Save Button Logic
    if st.button("Save to Cloud"):
        if not vocab_input:
            st.error("Please enter a vocabulary word.")
        else:
            # Prepare the new row
            # If phrase is empty, use "1" as per your specific logic
            final_phrase = phrase_input if phrase_input.strip() else "1"
            
            new_data = pd.DataFrame([{"Vocab": vocab_input, "Phrase": final_phrase}])
            updated_df = pd.concat([df, new_data], ignore_index=True)

            try:
                # Save to GitHub
                save_to_github(updated_df)
                
                # Success Message
                st.success(f"✅ Saved: **{vocab_input}**")
                
                # --- THE FIX: CLEAR THE INPUTS ---
                st.session_state["new_vocab"] = ""   # Reset Vocab to blank
                st.session_state["new_phrase"] = ""  # Reset Phrase to blank
                
                # Force the app to reload immediately so you see blank boxes
                time.sleep(1) # Optional: slight pause so you can see the success message
                st.rerun()    
                
            except Exception as e:
                st.error(f"Error saving to GitHub: {e}")
                
with tab2:
    if df.empty: st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        search = st.text_input("🔎 Search...", "").lower().strip()
        display_df = df[df['vocab'].str.contains(search, case=False)] if search else df
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True)
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                if save_to_github(edited.sort_values(by="vocab", ignore_index=True)):
                    st.success("✅ Cloud updated!"); st.rerun()
        with col2:
            csv = edited.to_csv(index=False).encode()
            st.download_button("📥 CSV", csv, "vocab.csv", "text/csv", use_container_width=True)

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck (.apkg)")
    if df.empty:
        st.info("Add words first!")
    else:
        # User input for Deck Name
        deck_name_input = st.text_input("📦 Deck Name", value="My Cyberpunk Vocab")
        
        c1, c2 = st.columns(2)
        with c1:
            batch_size = st.slider("⚡ Batch Size (AI)", 1, 10, 5)
        with c2:
            include_audio = st.checkbox("🔊 Generate Audio", value=True)

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            raw_notes = process_anki_data(df, batch_size=batch_size)
            
            with st.spinner("📦 Packaging Deck & Audio..."):
                apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^a-zA-Z0-9]', '_', deck_name_input)
            filename = f"{safe_filename}_{timestamp}.apkg"
            
            st.success(f"🎉 {len(raw_notes)} cards in '{deck_name_input}' ready!")
            st.download_button("📥 Download .apkg", apkg_buffer, filename, "application/octet-stream", use_container_width=True)
            
            # Show preview of text field to confirm format
            st.dataframe(pd.DataFrame(raw_notes)[['VocabRaw', 'Text']].head(3), use_container_width=True)

st.caption("✅ Restored original format: Phrase + Vocab: {{c1::Translation}}")
