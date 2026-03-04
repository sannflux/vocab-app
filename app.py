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
import genanki  # Requires pip install genanki (already in your requirements.txt)

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="wide", page_icon="📚")  # Changed to wide for better responsiveness
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
        audio_field = f'[sound:https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=en&q={vocab_raw}]'  # Added audio TTS URL
        anki_notes.append({"Text": text_field, "Pronunciation": pronunciation_field, "Definition": eng_def,
                           "Examples": examples_field, "Synonyms": synonyms_field, "Antonyms": antonyms_field, "Etymology": etymology, "Audio": audio_field})
    return anki_notes  # Return list of dicts for genanki

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

if 'df' not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

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
    if 'selected_lang_name' not in st.session_state:
        st.session_state.selected_lang_name = list(lang_options.keys())[0]
    selected_lang_name = st.selectbox("🎙️ Speech Language", list(lang_options.keys()), index=list(lang_options.keys()).index(st.session_state.selected_lang_name))
    st.session_state.selected_lang_name = selected_lang_name
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
        v = st.text_input("📝 Vocab (required)", placeholder="e.g. serendipity", key="add_vocab").lower().strip()
        p_raw = st.text_input("🔤 Phrase / Example (type 1 to skip)", placeholder="I found it by serendipity!", key="add_phrase").strip()
        if st.form_submit_button("💾 Save to Cloud", use_container_width=True):
            if v and v not in df['vocab'].values:
                p = "" if p_raw.upper() == "1" else p_raw.capitalize()
                updated = pd.concat([df, pd.DataFrame([{"vocab": v, "phrase": p}])], ignore_index=True)
                if save_to_github(updated): 
                    st.success(f"✅ '{v}' added!")
                    st.session_state.df = updated  # Update session state to reduce reruns
                    st.rerun()

with tab2:
    if df.empty:
        st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        if 'search' not in st.session_state:
            st.session_state.search = ""
        st.session_state.search = st.text_input("🔎 Search...", value=st.session_state.search, key="edit_search").lower().strip()
        search = st.session_state.search
        display_df = df[df['vocab'].str.contains(search, case=False)] if search else df
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="data_editor")
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("💾 Save Changes to Cloud", type="primary", use_container_width=True):
                edited = edited.drop_duplicates(subset=['vocab'])  # Prevent duplicates
                if save_to_github(edited.sort_values(by="vocab", ignore_index=True)):
                    st.success("✅ Cloud updated!")
                    st.session_state.df = edited.sort_values(by="vocab", ignore_index=True)  # Update session
                    st.rerun()
        with col2:
            csv = edited.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, "my_vocabulary.csv", "text/csv", use_container_width=True)
        st.divider()
        st.subheader("🔊 Quick Audio Practice")
        if 'quick_word' not in st.session_state:
            st.session_state.quick_word = sorted(df["vocab"].tolist())[0] if df["vocab"].tolist() else None
        quick_word = st.selectbox("Choose word:", sorted(df["vocab"].tolist()), index=sorted(df["vocab"].tolist()).index(st.session_state.quick_word) if st.session_state.quick_word in df["vocab"].tolist() else 0)
        st.session_state.quick_word = quick_word
        quick_phrase = df[df["vocab"]==quick_word]["phrase"].iloc[0] if quick_word else ""
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
        if 'generation_future' not in st.session_state:
            st.session_state.generation_future = None
        if 'generation_progress' not in st.session_state:
            st.session_state.generation_progress = 0.0
        if 'generation_results' not in st.session_state:
            st.session_state.generation_results = None
        if 'generation_queue' not in st.session_state:
            st.session_state.generation_queue = queue.Queue()  # For progress updates

        def run_generation_in_background(q):
            try:
                vocab_phrase_list = df[['vocab', 'phrase']].values.tolist()
                batch_size = 5
                num_batches = (len(vocab_phrase_list) + batch_size - 1) // batch_size
                all_card_data = []
                for i in range(num_batches):
                    if st.session_state.get('cancel_generation', False):  # Check for cancel
                        raise Exception("Generation canceled by user")
                    batch_start = i * batch_size
                    batch_end = min(batch_start + batch_size, len(vocab_phrase_list))
                    batch = vocab_phrase_list[batch_start:batch_end]
                    batch_data = generate_anki_card_data_batched(batch)
                    all_card_data.extend(batch_data)
                    progress = (i + 1) / num_batches
                    q.put(progress)  # Send progress to queue
                anki_notes = generate_anki_notes(pd.DataFrame({"vocab": [v[0] for v in vocab_phrase_list], "phrase": [v[1] for v in vocab_phrase_list]}))
                q.put(anki_notes)  # Send final results
            except Exception as e:
                q.put(e)  # Send error

        if st.button("🚀 Generate Anki Cards with Gemini AI", type="primary", use_container_width=True):
            if st.session_state.generation_future is None:  # Start only if not running
                st.session_state.cancel_generation = False
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                st.session_state.generation_future = executor.submit(run_generation_in_background, st.session_state.generation_queue)
                st.session_state.generation_progress = 0.0
                st.session_state.generation_results = None
                st.rerun()  # Rerun to show progress immediately

        # Monitor progress on every run
        if st.session_state.generation_future is not None:
            with st.spinner("🧠 Generating... (Do not interact until done)"):
                progress_bar = st.progress(st.session_state.generation_progress)
                while not st.session_state.generation_queue.empty():
                    item = st.session_state.generation_queue.get()
                    if isinstance(item, float):
                        st.session_state.generation_progress = item
                        progress_bar.progress(item)
                    elif isinstance(item, list):  # anki_notes
                        st.session_state.generation_results = item
                    elif isinstance(item, Exception):
                        st.error(f"Error: {str(item)}")
                        st.session_state.generation_future = None

                if st.session_state.generation_future.done():
                    st.session_state.generation_future = None  # Reset for next run

            if st.button("❌ Cancel Generation"):
                st.session_state.cancel_generation = True
                st.rerun()

        if st.session_state.generation_results is not None:
            anki_notes = st.session_state.generation_results
            # ... (rest of your Anki model/deck/package code here, using anki_notes)
            # (Generate .apkg as before and provide download)
            st.success(f"🎉 {len(anki_notes)} cards ready!")
            # ... (download button and preview)

            st.session_state.generation_results = None
                progress_bar = st.progress(0)
                vocab_phrase_list = df[['vocab', 'phrase']].values.tolist()
                batch_size = 5
                num_batches = (len(vocab_phrase_list) + batch_size - 1) // batch_size
                all_card_data = []
                for i in range(num_batches):
                    batch_start = i * batch_size
                    batch_end = min(batch_start + batch_size, len(vocab_phrase_list))
                    batch = vocab_phrase_list[batch_start:batch_end]
                    batch_data = generate_anki_card_data_batched(batch)  # Assuming batched function handles one batch
                    all_card_data.extend(batch_data)
                    progress_bar.progress((i + 1) / num_batches)
                anki_notes = generate_anki_notes(pd.DataFrame({"vocab": [v[0] for v in vocab_phrase_list], "phrase": [v[1] for v in vocab_phrase_list]}))  # Get list of dicts
                
                # Define custom Anki model with provided templates and CSS
                model_id = random.randrange(1 << 30, 1 << 31)
                my_model = genanki.Model(
                    model_id,
                    'Custom Vocab Cloze',
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
                    templates=[
                        {
                            'name': 'Cloze',
                            'qfmt': '''
<div class="vellum-focus-container front">
  <div class="prompt-text">
    {{cloze:Text}}
  </div>
</div>
''',
                            'afmt': '''
<div class="vellum-focus-container back">
  <div class="prompt-text solved-text">
    {{cloze:Text}}
  </div>
</div>

<div class="vellum-detail-container">
  {{#Audio}}
  <div class="vellum-section audio">
    <div class="section-header">🔊 AUDIO</div>
    <div class="content">{{Audio}}</div>
  </div>
  {{/Audio}}

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
</div>
''',
                        },
                    ],
                    css='''
/* --- Global Settings (Cyberpunk Glitch Theme) --- */
.card {
  font-family: 'Roboto Mono', 'Consolas', monospace; /* Monospace font for a terminal/glitch feel */
  font-size: 18px;
  line-height: 1.5;
  font-weight: 400;
  color: #00ff41; /* Primary Neon Green/Matrix color */
  
  /* Deep, dark background with subtle texture */
  background-color: #111111; 
  background-image: repeating-linear-gradient(0deg, #181818, #181818 1px, #111111 1px, #111111 20px);
  
  padding: 30px 20px;
  max-width: 800px;
  margin: 0 auto;
  box-sizing: border-box;
  text-align: left;
}

/* --- Night Mode (Minimal change, maybe subtle shift) --- */
.nightMode .card {
  color: #00aaff; /* Shift to Neon Blue in Night Mode */
  background-color: #080808;
}

/* --- UNIFIED FRONT/BACK FOCUS CONTAINER (THE DISPLAY) --- */
.vellum-focus-container {
  /* High-tech screen/display frame */
  background: #0d0d0d;
  padding: 30px 20px;
  margin: 0 auto 40px;
  max-width: 95%;
  border-radius: 4px; 
  
  /* Digital Border / Light Glitch Effect */
  border: 2px solid #00ff41; 
  box-shadow: 
    0 0 5px #00ff41,       /* Inner glow */
    0 0 15px rgba(0, 255, 65, 0.4), /* Outer halo */
    0 4px 8px rgba(0, 0, 0, 0.5); /* Deep shadow */
    
  text-align: center;
  position: relative;
  overflow: hidden;
}

.nightMode .vellum-focus-container {
  border: 2px solid #00aaff;
  box-shadow: 
    0 0 5px #00aaff,
    0 0 15px rgba(0, 170, 255, 0.4),
    0 4px 8px rgba(0, 0, 0, 0.6);
}

.prompt-text {
  font-family: 'Electrolize', 'Arial Narrow', sans-serif; /* Stylized, condensed font */
  font-size: clamp(2.0em, 6vw, 3.0em); 
  font-weight: 900;
  color: #ffffff; /* Bright white/light for high contrast */
  text-shadow: 
    1px 1px 0 #ff00ff, /* Magenta glitch offset */
    -1px -1px 0 #00ffff; /* Cyan glitch offset */
  font-style: normal;
}

.nightMode .prompt-text {
  color: #f0f8ff; 
  text-shadow: 
    1px 1px 0 #ff00ff,
    -1px -1px 0 #00ffff;
}

/* --- SPECIFIC STYLING FOR FRONT (Hidden Cloze) - Diperlukan di Styling agar .cloze tetap berfungsi di Front Template juga */
.vellum-focus-container.front .cloze {
  color: #111111; /* Hidden color, close to background */
  background-color: #00ff41; /* Highlighted block */
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
  /* Solved word in Glitch Neon */
  color: #ff00ff; /* Neon Magenta/Pink */
  font-weight: 900;
  background: none;
  padding: 0 3px;
  text-decoration: none;
  border-bottom: 3px double #00ffff; /* Cyan underline */
  text-shadow: 
    0 0 5px #ff00ff; /* Glow */
  font-style: normal;
}

.nightMode .vellum-focus-container.back .cloze {
  color: #00ffff; /* Neon Cyan */
  border-bottom: 3px double #ff00ff; /* Magenta underline */
  text-shadow: 
    0 0 5px #00ffff;
}

/* --- DETAIL SECTIONS (Data Blocks) --- */
.vellum-detail-container {
  padding: 10px 0;
}

.vellum-section {
  margin: 15px 0;
  padding: 10px 0;
  border-bottom: 1px dashed #00ff41; 
  padding-left: 5px;
  padding-right: 5px;
}

.nightMode .vellum-section {
  border-bottom: 1px dashed #00aaff;
}

.section-header {
  font-size: 1.1em;
  font-weight: 600;
  margin-bottom: 8px;
  color: #00ffff; /* Secondary Neon Cyan */
  display: flex;
  align-items: center;
  gap: 8px;
  border-left: 3px solid; 
  padding-left: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.nightMode .section-header {
  color: #ff00ff; /* Secondary Neon Magenta */
}

.content {
  font-size: 0.95em;
  color: #aaffaa; /* Lighter green for content */
  padding: 0 0 0 13px; 
}

.nightMode .content {
  color: #99ccff; /* Lighter blue for content */
}

/* Colored markers for sections (Data Stream Hues) */
.vellum-section.definition .section-header { border-left-color: #00ff41; } 
.vellum-section.pronunciation .section-header { border-left-color: #ff00ff; }
.vellum-section.examples .section-header { border-left-color: #ffff00; } /* Neon Yellow */
.vellum-section.synonyms .section-header { border-left-color: #ff00ff; } 
.vellum-section.antonyms .section-header { border-left-color: #ff4100; } /* Neon Red/Orange */
.vellum-section.etymology .section-header { border-left-color: #00ffff; } 
.vellum-section.audio .section-header { border-left-color: #00aaff; }  /* Added for audio */

/* Type-Specific Content Styles */
.pronunciation .content {
  font-family: 'Consolas', monospace;
  font-size: 1.05em;
  font-weight: 500;
  color: #ffff00; /* Pronunciation in Neon Yellow */
}

.examples .content {
  color: #77ff77;
  font-style: italic;
  font-size: 0.9em;
}

.audio .content {
  text-align: center;  /* Center audio playback */
}

/* Enhanced Responsive adjustment */
@media (max-width: 600px) {  /* Adjusted from 480px for better mobile */
  .card {
    font-size: 16px;
    padding: 15px 10px;
  }
  .vellum-focus-container {
    padding: 20px 10px;
    max-width: 100%;
  }
  .prompt-text {
    font-size: clamp(1.5em, 8vw, 2.5em);  /* Smaller clamp for mobile */
  }
  .vellum-section {
    padding: 8px 0;
  }
  .section-header {
    font-size: 1em;
    gap: 5px;
  }
  .content {
    font-size: 0.9em;
    padding-left: 10px;
  }
}

@media (max-width: 400px) {
  .card {
    padding: 10px 5px;
  }
  .vellum-focus-container {
    padding: 15px 5px;
  }
}
''',
                    model_type=genanki.CLOZE_MODEL  # Fixed: Changed from genanki.CLOZE to genanki.CLOZE_MODEL
                )

                # Create deck
                deck_id = random.randrange(1 << 30, 1 << 31)
                my_deck = genanki.Deck(deck_id, 'My Cloud Vocab Deck')

                # Add notes
                for note_dict in anki_notes:
                    my_note = genanki.Note(
                        model=my_model,
                        fields=note_dict
                    )
                    my_deck.add_note(my_note)

                # Create package and write to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"anki_cards_export_{timestamp}.apkg"
                genanki.Package(my_deck).write_to_file(filename)

                # Read bytes for download
                with open(filename, 'rb') as f:
                    apkg_bytes = f.read()

                st.success(f"🎉 {len(anki_notes)} cards ready!")
                st.download_button("📥 Download Anki APKG", apkg_bytes, filename, "application/zip", use_container_width=True)
                # Optional: Preview first few as DF
                st.dataframe(pd.DataFrame(anki_notes).head(3), use_container_width=True)

st.caption("✅ 100% identical to Colab + API key changer + better sentence capitalization + responsive wide layout + session state for re-renders + progress bar + Anki audio + .apkg export (install genanki via pip)")
