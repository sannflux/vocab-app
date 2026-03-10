import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import random
from datetime import date, datetime, timezone, timedelta
import google.generativeai as genai
import json
import re
import time
import os
import tempfile
import hashlib
import concurrent.futures
import altair as alt

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

# --- TIMEZONE (WIB UTC+7) ---
WIB = timezone(timedelta(hours=7))
def get_wib_now():
    return datetime.now(WIB).strftime("%d-%m-%Y %H:%M")

# --- SESSION STATE INIT ---
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY
if "unsaved_changes" not in st.session_state:
    st.session_state.unsaved_changes = False
if "deck_name" not in st.session_state:
    st.session_state.deck_name = "-English Learning::Vocabulary"
if "deleted_rows_history" not in st.session_state:
    st.session_state.deleted_rows_history = []
if "audio_accent" not in st.session_state:
    st.session_state.audio_accent = "com"
if "audio_speed" not in st.session_state:
    st.session_state.audio_speed = False
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = ""
    
# Form input state management for the Smart Extractor
if "input_phrase" not in st.session_state:
    st.session_state.input_phrase = ""
if "input_vocab" not in st.session_state:
    st.session_state.input_vocab = ""

def clear_add_inputs():
    st.session_state.input_phrase = ""
    st.session_state.input_vocab = ""

# ========================== GITHUB CONNECT ==========================
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== LOAD / SAVE DATA ==========================
@st.cache_data(ttl=600)
def load_data_from_github():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        if 'status' not in df.columns: df['status'] = 'New'
        
        # WIB DATE FIX: Ensure column exists and clean up 'None'
        if 'date_added' not in df.columns:
            df['date_added'] = get_wib_now()
        df['date_added'] = df['date_added'].fillna(get_wib_now()).replace("None", get_wib_now())
        
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404: return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'date_added'])
        else: st.error(f"❌ CRITICAL: GitHub Error {e.status}"); st.stop()
    except Exception as e:
        st.error(f"❌ CRITICAL: CSV Error. {e}"); st.stop()

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, f"Vocab App Update - {get_wib_now()}", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data_from_github.clear()
    return True

# Initialize Session State DataFrame
if "vocab_df" not in st.session_state:
    st.session_state.vocab_df = load_data_from_github().copy()

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_LANG = st.selectbox("🎯 Definition Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    
    # CUSTOM PROMPT INJECTOR
    st.session_state.custom_prompt = st.text_area("🧠 Custom AI Rules (Optional)", value=st.session_state.custom_prompt, placeholder="e.g., Make examples funny, use business context...")
    
    st.divider()
    
    # AUDIO SETTINGS
    st.header("🔊 Audio Settings")
    accent_map = {"US English": "com", "UK English": "co.uk", "Australian English": "com.au", "Indian English": "co.in"}
    selected_accent = st.selectbox("Accent", list(accent_map.keys()), index=0)
    st.session_state.audio_accent = accent_map[selected_accent]
    st.session_state.audio_speed = st.checkbox("Slow Audio Generation", value=st.session_state.audio_speed)
    
    st.divider()
    
    # 🌟 VISUAL PROGRESS DASHBOARD
    st.subheader("📊 Learning Progress")
    if not st.session_state.vocab_df.empty:
        status_counts = st.session_state.vocab_df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        chart = alt.Chart(status_counts).mark_arc(innerRadius=40).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Status", type="nominal", scale=alt.Scale(domain=['Done', 'New'], range=['#00ff41', '#ff00ff'])),
            tooltip=['Status', 'Count']
        ).properties(height=200)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No words yet to track!")

    st.divider()
    
    st.header("🔑 Gemini API Key")
    alt_key = st.text_input("Alternative key", type="password", value="", placeholder="AIzaSy...")
    if alt_key and alt_key != st.session_state.gemini_key:
        st.session_state.gemini_key = alt_key
        st.success("✅ Switched!")
        st.rerun()

    st.divider()
    
    # ☁️ LAZY CLOUD SYNC LOGIC
    if st.session_state.unsaved_changes:
        st.warning("⚠️ You have unsaved changes locally!")
        if st.button("☁️ Sync to GitHub", type="primary", use_container_width=True):
            with st.spinner("Syncing..."):
                if save_to_github(st.session_state.vocab_df):
                    st.session_state.unsaved_changes = False
                    st.success("✅ Synced successfully!")
                    time.sleep(1)
                    st.rerun()
    else:
        st.success("☁️ Cloud is up to date.")
        
    if not st.session_state.vocab_df.empty:
        csv_full = st.session_state.vocab_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Backup DB", csv_full, f"vocab_backup_{date.today()}.csv", "text/csv", use_container_width=True)

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

# ========================== SPEECH & AI HELPERS ==========================
def speak_word(text: str):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js_lang_map = {"com": "en-US", "co.uk": "en-GB", "com.au": "en-AU", "co.in": "en-IN"}
    js_lang = js_lang_map.get(st.session_state.audio_accent, "en-US")
    rate = "0.6" if st.session_state.audio_speed else "0.95"
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{js_lang}";u.rate={rate};window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

@st.cache_data(ttl=3600)
def fetch_ai_definition(vocab, phrase, api_key, model_name, target_lang):
    model = get_gemini_model(api_key, model_name)
    if not model: return "AI Unavailable."
    prompt = f'Provide a 1-sentence {target_lang} translation and short English definition for "{vocab}" in the context of: "{phrase}". JSON format: {{"translation": "...", "definition": "..."}}'
    try:
        res = model.generate_content(prompt).text
        data = json.loads(res)
        return f"**{target_lang}:** {data.get('translation', '')} \n\n **Def:** {data.get('definition', '')}"
    except Exception:
        return "Could not fetch definition."

# ========================== BATCH GENERATOR ==========================
def robust_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except Exception: pass
    return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []

    all_card_data = []
    progress_bar = st.progress(0)
    total_items = len(vocab_phrase_list)
    
    custom_rule = f"\n5. CUSTOM RULE: {st.session_state.custom_prompt}" if st.session_state.custom_prompt else ""

    for i in range(0, total_items, batch_size):
        progress_bar.progress(i / total_items, text=f"🤖 AI Processing {i}/{total_items} words...")
        batch = vocab_phrase_list[i:i + batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]

        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES: 
1. Copy ALL fields exactly. 
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT. Use this hint to pick the specific definition, but generate a NEW sentence for the final 'phrase' field.
3. IF 'phrase' is normal text: Define based on that usage.
4. IF 'phrase' is empty: Generate ONE simple sentence (max 12 words).{custom_rule}
NEVER use markdown formatting. Plain text only.
OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "phrase_translation": "{TARGET_LANG} translation of the entire phrase/sentence", "translation": "{TARGET_LANG} meaning of the vocab word itself", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "Plain text only."}}]
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
             st.error(f"❌ Batch failed. Rescuing {len(all_card_data)} successfully generated words.")
             break

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

        phrase = cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("phrase", ""))))
        phrase = fix_vocab_casing(ensure_trailing_dot(phrase), vocab_raw)
        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""
        
        # New Context Translation
        phrase_translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("phrase_translation", ""))))

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
            "VocabRaw": vocab_raw, "Text": text_field, "PhraseTranslation": phrase_translation,
            "Pronunciation": pronunciation_field, "Definition": eng_def, 
            "Examples": examples_field, "Synonyms": synonyms_field, 
            "Antonyms": antonyms_field, "Etymology": etymology
        })
    return processed_notes

# ========================== AUDIO HELPER ==========================
def generate_audio_file(vocab, temp_dir, accent="com", is_slow=False):
    try:
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        file_path = os.path.join(temp_dir, clean_filename)
        if vocab.strip():
            tts = gTTS(text=vocab, lang='en', tld=accent, slow=is_slow)
            tts.save(file_path)
            return vocab, clean_filename, file_path
    except Exception: pass
    return vocab, None, None

# ========================== CSS THEMES ==========================
THEMES = {
    "Cyberpunk Glitch": """
        .card { font-family: 'Roboto Mono', monospace; font-size: 18px; line-height: 1.5; color: #00ff41; background-color: #111111; padding: 30px 20px; text-align: left; }
        .vellum-focus-container { background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px; border: 2px solid #00ff41; box-shadow: 0 0 5px #00ff41; text-align: center; }
        .prompt-text { font-family: sans-serif; font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }
        .context-translation { font-style: italic; color: #ff00ff; font-size: 0.9em; margin-top: 10px; }
        .cloze { color: #111111; background-color: #00ff41; padding: 2px 4px; }
        .solved-text .cloze { color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }
        .vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed #00ff41; }
        .section-header { font-weight: 600; color: #00ffff; border-left: 3px solid #00ff41; padding-left: 10px; }
        .content { color: #aaffaa; padding-left: 13px; }
    """,
    "Minimalist Light": """
        .card { font-family: 'Inter', sans-serif; font-size: 18px; line-height: 1.6; color: #333333; background-color: #ffffff; padding: 30px 20px; text-align: left; }
        .vellum-focus-container { background: #f9fafb; padding: 30px 20px; margin: 0 auto 40px; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center; }
        .prompt-text { font-size: 1.6em; font-weight: 700; color: #111827; }
        .context-translation { font-style: italic; color: #4b5563; font-size: 0.9em; margin-top: 10px; }
        .cloze { color: #ffffff; background-color: #3b82f6; border-radius: 4px; padding: 2px 6px; }
        .solved-text .cloze { color: #2563eb; background: none; border-bottom: 2px solid #3b82f6; }
        .vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px solid #f3f4f6; }
        .section-header { font-weight: 600; color: #6b7280; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
        .content { color: #4b5563; }
        
        /* --- NIGHT MODE OVERRIDES --- */
        .nightMode .card { background-color: #121212 !important; color: #e0e0e0 !important; }
        .nightMode .vellum-focus-container { background: #1e1e1e !important; border-color: #333 !important; }
        .nightMode .prompt-text { color: #ffffff !important; }
        .nightMode .context-translation { color: #9ca3af !important; }
        .nightMode .solved-text .cloze { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
        .nightMode .vellum-section { border-bottom-color: #333 !important; }
        .nightMode .section-header { color: #9ca3af !important; }
        .nightMode .content { color: #d1d5db !important; }
    """,
    "Dark Academia": """
        .card { font-family: 'Merriweather', serif; font-size: 18px; line-height: 1.6; color: #d4c4a8; background-color: #2c2826; padding: 30px 20px; text-align: left; }
        .vellum-focus-container { background: #231f1d; padding: 30px 20px; margin: 0 auto 40px; border: 1px solid #4a4138; border-radius: 4px; text-align: center; }
        .prompt-text { font-size: 1.8em; font-weight: 700; color: #e8dcc7; }
        .context-translation { font-style: italic; color: #b08d57; font-size: 0.9em; margin-top: 10px; }
        .cloze { color: #2c2826; background-color: #b08d57; padding: 2px 4px; border-radius: 2px; }
        .solved-text .cloze { color: #cba873; background: none; font-style: italic; border-bottom: 1px solid #cba873; }
        .vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed #4a4138; }
        .section-header { font-weight: 700; color: #8e735b; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.1em; border-bottom: 1px solid #4a4138; display: inline-block; padding-bottom: 3px; margin-bottom: 8px; }
        .content { color: #d4c4a8; }
    """
}

# ========================== GENANKI LOGIC ==========================
def create_anki_package(notes_data, deck_name, css_theme, generate_audio=True, max_per_deck=0):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back">
  <div class="prompt-text solved-text">{{cloze:Text}}</div>
  {{#PhraseTranslation}}<div class="context-translation">{{PhraseTranslation}}</div>{{/PhraseTranslation}}
</div>
<div class="vellum-detail-container">
  {{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}
  {{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}
  {{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}
  {{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}
  {{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}
  <div style='display:none'>{{Audio}}</div>
</div>{{Audio}}"""

    my_model = genanki.Model(
        1607392320, 'Custom Vocab Model v2',
        fields=[{'name': 'Text'}, {'name': 'PhraseTranslation'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}],
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}],
        css=css_theme,
        model_type=genanki.Model.CLOZE 
    )
    
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            progress_bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
                future_to_vocab = {executor.submit(generate_audio_file, v, temp_dir, st.session_state.audio_accent, st.session_state.audio_speed): v for v in unique_vocabs}
                for i, future in enumerate(concurrent.futures.as_completed(future_to_vocab)):
                    vocab_key, fname, fpath = future.result()
                    if fname and fpath:
                        media_files.append(fpath)
                        audio_map[vocab_key] = f"[sound:{fname}]"
                    progress_bar.progress((i + 1) / len(unique_vocabs), text=f"🔊 Generating Audio: {vocab_key}...")
            progress_bar.empty()

        decks = []
        base_deck_id = 2059400110
        
        if max_per_deck > 0 and len(notes_data) > max_per_deck:
            chunks = [notes_data[i:i + max_per_deck] for i in range(0, len(notes_data), max_per_deck)]
            for idx, chunk in enumerate(chunks):
                subdeck_name = f"{deck_name}::Part {idx + 1}"
                subdeck = genanki.Deck(base_deck_id + idx, subdeck_name)
                for note_data in chunk:
                    audio_tag = audio_map.get(note_data['VocabRaw'], "")
                    my_note = genanki.Note(
                        model=my_model,
                        fields=[
                            note_data['Text'], note_data['PhraseTranslation'], note_data['Pronunciation'], note_data['Definition'],
                            note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'],
                            note_data['Etymology'], audio_tag
                        ]
                    )
                    subdeck.add_note(my_note)
                decks.append(subdeck)
        else:
            single_deck = genanki.Deck(base_deck_id, deck_name)
            for note_data in notes_data:
                audio_tag = audio_map.get(note_data['VocabRaw'], "")
                my_note = genanki.Note(
                    model=my_model,
                    fields=[
                        note_data['Text'], note_data['PhraseTranslation'], note_data['Pronunciation'], note_data['Definition'],
                        note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'],
                        note_data['Etymology'], audio_tag
                    ]
                )
                single_deck.add_note(my_note)
            decks.append(single_deck)
            
        my_package = genanki.Package(decks)
        my_package.media_files = media_files
        
        buffer = io.BytesIO()
        output_path = os.path.join(temp_dir, 'output.apkg')
        my_package.write_to_file(output_path)
        with open(output_path, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        return buffer

# ========================== WORD OF THE DAY ==========================
st.divider()
st.header("🌟 Word of the Day")
if not st.session_state.vocab_df.empty:
    today_str = date.today().isoformat()
    random.seed(today_str)
    try:
        row = st.session_state.vocab_df.sample(n=1).iloc[0]
        wotd_vocab = row["vocab"]
        wotd_phrase = row["phrase"]
        
        st.subheader(wotd_vocab.upper())
        if wotd_phrase: st.caption(wotd_phrase)
        
        c1, c2 = st.columns([1,4])
        with c1:
            if st.button("🔊 Pronounce"): speak_word(wotd_vocab)
        with c2:
            if st.button("✨ AI Define"):
                with st.spinner("Thinking..."):
                    def_text = fetch_ai_definition(wotd_vocab, wotd_phrase, st.session_state.gemini_key, GEMINI_MODEL, TARGET_LANG)
                    st.info(def_text)
    except Exception as e:
         st.error(f"Error loading WOTD: {e}")
else: 
    st.info("No words yet! Add some below.")

st.divider()

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")
    
    if add_mode == "Single":
        st.info("💡 **Smart Add:** Paste your phrase first! We'll extract the words so you can just click the one you want to learn.")
        
        # 1. User pastes Phrase FIRST
        p_raw = st.text_area("🔤 Phrase", key="input_phrase", placeholder="Paste sentence here...", help="Start with '*' to give AI a context hint instead of a sentence (e.g., '*bird')")
        
        v = ""
        # 2. Automatically generate the dropdown if a phrase exists
        if p_raw and not p_raw.startswith("*"):
            # Extract clean words, keep lowercased, remove duplicates but preserve order
            raw_words = re.findall(r'[^\W\d_]+(?:[-\'][^\W\d_]+)*', p_raw)
            extracted_words = list(dict.fromkeys([w.lower() for w in raw_words]))
            
            if extracted_words:
                v_choice = st.selectbox("📝 Select Vocab Word from Phrase", options=["(Type manually)"] + extracted_words)
                if v_choice == "(Type manually)":
                    v = st.text_input("Type vocab manually", key="input_vocab").lower().strip()
                else:
                    v = v_choice
        else:
            # Fallback if no phrase is typed yet
            v = st.text_input("📝 Vocab", key="input_vocab").lower().strip()
            
        # UI Warning if duplicate
        if v and not st.session_state.vocab_df.empty and v in st.session_state.vocab_df['vocab'].values:
            st.warning(f"⚠️ '{v}' is already in your list! Saving will update its phrase and mark it as 'New'.")

        # 3. Save Button
        if st.button("💾 Save Word", use_container_width=True, type="primary"):
            if v:
                p = "" if p_raw == "1" else p_raw if p_raw.startswith("*") else p_raw.capitalize()
                
                if not st.session_state.vocab_df.empty and v in st.session_state.vocab_df['vocab'].values:
                    st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'] == v, ['phrase', 'status']] = [p, 'New']
                else:
                    new_row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "date_added": get_wib_now()}])
                    st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
                
                st.session_state.unsaved_changes = True
                st.success(f"✅ Saved '{v}'!")
                clear_add_inputs()
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please provide a vocab word.")

    else:
        st.info("Paste words separated by newlines. Automatically cleans bullets and numbers!")
        bulk_text = st.text_area("Paste List", height=150, placeholder="- cat, The cat sat.\n- dog")
        if st.button("💾 Process Bulk List", type="primary"):
            lines = [l.strip() for l in bulk_text.split('\n') if l.strip()]
            new_rows = []
            for line in lines:
                clean_line = re.sub(r'^[\d\.\-\*\s]+', '', line)
                parts = clean_line.split(',', 1)
                bv = parts[0].strip().lower()
                bp = parts[1].strip() if len(parts) > 1 else ""
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "status": "New", "date_added": get_wib_now()})
            
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_df]).drop_duplicates(subset=['vocab'], keep='last').reset_index(drop=True)
                st.session_state.unsaved_changes = True
                st.success(f"✅ Added {len(new_rows)} words! (Locally saved. Don't forget to sync)")

with tab2:
    if st.session_state.vocab_df.empty: st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        
        c_sort, c_undo = st.columns(2)
        with c_sort:
            if st.button("🔤 Sort Alphabetically", use_container_width=True):
                st.session_state.vocab_df = st.session_state.vocab_df.sort_values(by="vocab", ignore_index=True)
                st.session_state.unsaved_changes = True
                st.rerun()
        with c_undo:
            if st.session_state.deleted_rows_history:
                if st.button(f"↩️ Undo Delete ({len(st.session_state.deleted_rows_history[-1])} words)", use_container_width=True):
                    restored_df = st.session_state.deleted_rows_history.pop()
                    st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, restored_df]).drop_duplicates(subset=['vocab'], keep='last').reset_index(drop=True)
                    st.session_state.unsaved_changes = True
                    st.rerun()
            else:
                st.button("↩️ Undo Delete", disabled=True, use_container_width=True)

        st.divider()
        
        c1, c2 = st.columns([2, 1])
        with c1: search = st.text_input("🔎 Search...", "").lower().strip()
        with c2: filter_new = st.checkbox("Show 'New' only")
        
        display_df = st.session_state.vocab_df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        if filter_new: display_df = display_df[display_df['status'] == 'New']
        
        st.caption("💡 Select a row and press `Delete` on your keyboard to remove words.")
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"status": st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True), "date_added": st.column_config.TextColumn("Date Added (WIB)", disabled=True)})
        
        if st.button("💾 Confirm Edits", type="primary", use_container_width=True):
            original_vocabs = set(display_df['vocab'])
            edited_vocabs = set(edited['vocab'])
            deleted_vocabs = original_vocabs - edited_vocabs
            
            if deleted_vocabs:
                deleted_df = st.session_state.vocab_df[st.session_state.vocab_df['vocab'].isin(deleted_vocabs)].copy()
                st.session_state.deleted_rows_history.append(deleted_df)
                st.session_state.vocab_df = st.session_state.vocab_df[~st.session_state.vocab_df['vocab'].isin(deleted_vocabs)]
            
            for index, row in edited.iterrows():
                mask = st.session_state.vocab_df['vocab'] == row['vocab']
                st.session_state.vocab_df.loc[mask, ['phrase', 'status']] = [row['phrase'], row['status']]
            
            st.session_state.unsaved_changes = True
            st.toast("✅ Edits confirmed locally!", icon="🎉")
            time.sleep(1)
            st.rerun()

with tab3:
    st.subheader("📇 Generate Anki Deck")
    
    selected_theme = st.selectbox("🎨 Select Card Theme", list(THEMES.keys()), index=0)
    
    with st.expander("🧪 AI Sandbox (Test 1 Word Live)"):
        test_words = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']['vocab'].tolist()
        if not test_words:
            st.info("Add some 'New' words to test!")
        else:
            c_test1, c_test2 = st.columns([3,1])
            with c_test1: sandbox_word = st.selectbox("Select a word to test how Gemini defines it:", test_words, label_visibility="collapsed")
            with c_test2: test_btn = st.button("✨ Generate Test", use_container_width=True)
            
            if test_btn:
                with st.spinner("Gemini is thinking..."):
                    test_subset = st.session_state.vocab_df[st.session_state.vocab_df['vocab'] == sandbox_word]
                    test_notes = process_anki_data(test_subset, batch_size=1)
                    if test_notes:
                        note = test_notes[0]
                        solved_text = re.sub(r'\{\{c1::(.*?)\}\}', r'<span class="cloze">\1</span>', note['Text'])
                        back_details = ""
                        if note['Definition']: back_details += f"<div class='vellum-section'><div class='section-header'>📜 DEFINITION</div><div class='content'>{note['Definition']}</div></div>"
                        if note['Pronunciation']: back_details += f"<div class='vellum-section'><div class='section-header'>🗣️ PRONUNCIATION</div><div class='content'>{note['Pronunciation']}</div></div>"
                        if note['Examples']: back_details += f"<div class='vellum-section'><div class='section-header'>🖋️ EXAMPLES</div><div class='content'>{note['Examples']}</div></div>"
                        
                        front_preview = f"""<div class="card" style="margin-bottom: 20px;"><div class="vellum-focus-container front"><div class="prompt-text">{note['Text']}</div></div></div>"""
                        back_preview = f"""<div class="card"><div class="vellum-focus-container back"><div class="prompt-text solved-text">{solved_text}</div><div class="context-translation">{note['PhraseTranslation']}</div></div><div class="vellum-detail-container">{back_details}</div></div>"""
                        
                        st.markdown(f"<style>{THEMES[selected_theme]}</style>", unsafe_allow_html=True)
                        st.caption("FRONT OF CARD:")
                        st.markdown(front_preview, unsafe_allow_html=True)
                        st.caption("BACK OF CARD:")
                        st.markdown(back_preview, unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to generate test card.")

    if st.session_state.vocab_df.empty: st.info("Add words first!")
    else:
        col_new, col_all = st.columns(2)
        col_new.metric("New Words", len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']))
        col_all.metric("Total Words", len(st.session_state.vocab_df))
        st.divider()
        
        st.session_state.deck_name = st.text_input("📦 Base Deck Name", value=st.session_state.deck_name)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: batch_size = st.slider("⚡ Batch Size", 1, 10, 5)
        with c2: include_audio = st.checkbox("🔊 Audio", value=True)
        with c3: process_only_new = st.checkbox("Only process 'New'", value=True)
        with c4: split_deck = st.number_input("Split > X cards (0=No)", min_value=0, value=50, step=10, help="Creates sub-decks natively in Anki.")

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'] if process_only_new else st.session_state.vocab_df
            if subset.empty: st.warning("⚠️ No words to process!")
            else:
                raw_notes = process_anki_data(subset, batch_size=batch_size)
                if not raw_notes: st.error("❌ Generation completely failed. Check API Key.")
                else:
                    with st.spinner("📦 Packaging Deck..."):
                        apkg_buffer = create_anki_package(raw_notes, st.session_state.deck_name, THEMES[selected_theme], generate_audio=include_audio, max_per_deck=split_deck)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    st.success(f"🎉 {len(raw_notes)} cards ready!")
                    st.download_button("📥 Download .apkg", apkg_buffer, f"AnkiDeck_{timestamp}.apkg", "application/octet-stream", use_container_width=True)
                    
                    if process_only_new:
                        processed_words = [n['VocabRaw'] for n in raw_notes]
                        st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(processed_words), 'status'] = 'Done'
                        st.session_state.unsaved_changes = True
                        st.caption("✅ Marked successfully processed words as 'Done'. (Don't forget to sync!)")
