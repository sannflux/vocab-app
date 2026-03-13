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

# Initialization for Tab 1 inputs to allow manual clearing without a form
if "p_input_text" not in st.session_state:
    st.session_state.p_input_text = ""
if "v_selected_words" not in st.session_state:
    st.session_state.v_selected_words = []

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    TARGET_LANG = st.selectbox(
        "🎯 Definition Language", 
        ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"],
        index=0
    )
    
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
    except Exception:
        pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
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
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]

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

        for attempt in range(5):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    break
            except Exception as e:
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
        else:
             st.error(f"❌ Batch failed. Skipping.")

    progress_bar.progress(1.0, text="✅ AI Generation Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    return all_card_data

def process_anki_data(df_subset, batch_size=6):
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
            "VocabRaw": vocab_raw, "Text": text_field, "Pronunciation": pronunciation_field, 
            "Definition": eng_def, "Examples": examples_field, "Synonyms": synonyms_field, 
            "Antonyms": antonyms_field, "Etymology": etymology
        })
    return processed_notes

# ========================== AUDIO HELPER ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        file_path = os.path.join(temp_dir, clean_filename)
        if vocab.strip():
            tts = gTTS(text=vocab, lang='en', slow=False)
            tts.save(file_path)
            return vocab, clean_filename, file_path
    except Exception: pass
    return vocab, None, None

# ========================== CSS / PREVIEW ==========================
CYBERPUNK_CSS = """
.card { font-family: 'Roboto Mono', monospace; font-size: 18px; color: #00ff41; background-color: #111111; padding: 30px 20px; }
.vellum-focus-container { background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px; border: 2px solid #00ff41; box-shadow: 0 0 15px rgba(0, 255, 65, 0.4); text-align: center; }
.prompt-text { font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }
.cloze { color: #111111; background-color: #00ff41; padding: 2px 4px; }
.solved-text .cloze { color: #ff00ff; background: none; border-bottom: 3px double #00ffff; }
.vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed #00ff41; }
.section-header { font-weight: 600; color: #00ffff; border-left: 3px solid #00ff41; padding-left: 10px; }
.content { color: #aaffaa; padding-left: 13px; }
"""

# ========================== GENANKI LOGIC ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True):
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

    my_model = genanki.Model(1607392319, 'Cyberpunk Vocab Model',
        fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_vocab = {executor.submit(generate_audio_file, v, temp_dir): v for v in unique_vocabs}
                for future in concurrent.futures.as_completed(future_to_vocab):
                    vk, fname, fpath = future.result()
                    if fname: 
                        media_files.append(fpath)
                        audio_map[vk] = f"[sound:{fname}]"

        for n in notes_data:
            my_note = genanki.Note(model=my_model, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")] )
            my_deck.add_note(my_note)
        
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buf = io.BytesIO()
        out = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(out)
        with open(out, "rb") as f: buf.write(f.read())
        buf.seek(0)
        return buf

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
    except GithubException:
        df = pd.DataFrame(columns=['vocab', 'phrase', 'status'])
    
    for col in ['vocab', 'phrase', 'status']:
        if col not in df.columns: df[col] = 'New' if col == 'status' else ""
    df['phrase'] = df['phrase'].fillna("")
    return df.sort_values(by="vocab", ignore_index=True)

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException:
        repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear()
    st.session_state.df = dataframe.copy()
    return True

if "df" not in st.session_state:
    st.session_state.df = load_data().copy()

# ========================== WORD OF THE DAY ==========================
with st.sidebar:
    st.divider()
    if not st.session_state.df.empty:
        csv_full = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Backup CSV", csv_full, f"vocab_backup_{date.today()}.csv", "text/csv")
    
    st.divider()
    st.header("🌟 Word of the Day")
    if not st.session_state.df.empty:
        random.seed(date.today().isoformat())
        try:
            row = st.session_state.df.sample(n=1).iloc[0]
            st.subheader(row["vocab"].upper())
            if row["phrase"]: st.caption(row["phrase"])
            if st.button("🔊 Pronounce"): speak_word(row["vocab"])
        except Exception: pass

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Phrase-First", "Bulk"], horizontal=True, label_visibility="collapsed")
    
    if add_mode == "Phrase-First":
        # Removed st.form to allow real-time reactivity for the multiselect
        p_input = st.text_area("📝 Paste Phrase", placeholder="Paste the sentence here...", height=100, key="p_input_text")
        
        words_in_phrase = []
        if p_input.strip():
            # Standard word boundary regex for dynamic extraction
            words_in_phrase = sorted(list(set(re.findall(r'\b\w+\b', p_input.lower()))))
        
        v_selected = st.multiselect("🎯 Select Vocab Word(s)", options=words_in_phrase, key="v_selected_words")
        
        if st.button("💾 Save to Cloud", type="primary", use_container_width=True):
            if p_input and v_selected:
                current_df = st.session_state.df.copy()
                phrase_formatted = p_input.strip().capitalize()
                
                for word in v_selected:
                    word_clean = word.strip().lower()
                    if not current_df.empty and word_clean in current_df['vocab'].values:
                        current_df.loc[current_df['vocab'] == word_clean, 'phrase'] = phrase_formatted
                        current_df.loc[current_df['vocab'] == word_clean, 'status'] = 'New'
                    else:
                        new_row = pd.DataFrame([{"vocab": word_clean, "phrase": phrase_formatted, "status": "New"}])
                        current_df = pd.concat([current_df, new_row], ignore_index=True)
                
                if save_to_github(current_df):
                    st.success(f"✅ Saved {len(v_selected)} words!")
                    # Manually clear session state keys to replicate form clearance
                    st.session_state.p_input_text = ""
                    st.session_state.v_selected_words = []
                    time.sleep(1)
                    st.rerun()
            else:
                st.warning("Please paste a phrase and select at least one word.")

    else:
        st.info("Paste words separated by newlines. Example: `cat, The cat sat.`")
        bulk_text = st.text_area("Paste List", height=150)
        if st.button("💾 Process Bulk List", type="primary"):
            lines = [l.strip() for l in bulk_text.split('\n') if l.strip()]
            new_rows = []
            for line in lines:
                parts = line.split(',', 1)
                bv = parts[0].strip().lower()
                bp = parts[1].strip() if len(parts) > 1 else ""
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "status": "New"})
            if new_rows:
                current_df = st.session_state.df.copy()
                current_df = pd.concat([current_df, pd.DataFrame(new_rows)]).drop_duplicates(subset=['vocab'], keep='last')
                if save_to_github(current_df): 
                    st.success(f"✅ Added {len(new_rows)} words!")
                    time.sleep(1)
                    st.rerun()

with tab2:
    if not st.session_state.df.empty:
        st.subheader(f"✏️ Edit List ({len(st.session_state.df)})")
        c1, c2 = st.columns([2, 1])
        search = c1.text_input("🔎 Search...", "").lower().strip()
        filter_new = c2.checkbox("Show 'New' only")
        
        display_df = st.session_state.df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        if filter_new: display_df = display_df[display_df['status'] == 'New']
        
        # Added explicit key to prevent state loss
        edited = st.data_editor(display_df, key="vocab_editor_main", num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"status": st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True)})
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            current_df = st.session_state.df.copy().set_index('vocab')
            current_df.update(edited.set_index('vocab'))
            if save_to_github(current_df.reset_index().sort_values(by="vocab", ignore_index=True)):
                st.toast("✅ Cloud updated!")
                st.rerun()

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")
    if st.session_state.df.empty: st.info("Add words first!")
    else:
        current_df = st.session_state.df.copy()
        col_new, col_all = st.columns(2)
        col_new.metric("New Words", len(current_df[current_df['status'] == 'New']))
        col_all.metric("Total Words", len(current_df))
        
        deck_name_input = st.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        c1, c2, c3 = st.columns(3)
        batch_size = c1.slider("⚡ Batch Size", 1, 10, 5)
        include_audio = c2.checkbox("🔊 Audio", value=True)
        process_only_new = c3.checkbox("Only 'New'", value=True)

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            subset = current_df[current_df['status'] == 'New'] if process_only_new else current_df
            if subset.empty: st.warning("No words to process!")
            else:
                raw_notes = process_anki_data(subset, batch_size=batch_size)
                if raw_notes:
                    with st.spinner("Packaging Deck..."):
                        apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio)
                    st.session_state.apkg_buffer = apkg_buffer.getvalue()
                    st.session_state.apkg_filename = f"AnkiDeck_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg"
                    st.session_state.deck_ready = True
                    if process_only_new:
                        current_df.loc[current_df['vocab'].isin(subset['vocab']), 'status'] = 'Done'
                        save_to_github(current_df)

        if st.session_state.deck_ready and st.session_state.apkg_buffer:
            st.download_button("📥 Download .apkg", data=st.session_state.apkg_buffer, file_name=st.session_state.apkg_filename, mime="application/octet-stream", use_container_width=True)
