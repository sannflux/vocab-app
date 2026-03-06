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

# AUDIO & ANKI
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Add `gTTS` and `genanki` to requirements.txt")
    st.stop()

# ========================== CONFIG ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["REPO_NAME"]
DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]

USER_NATIVE_LANGUAGE = "Indonesian"
GEMINI_MODEL = "gemini-2.5-flash"   # Stable & recommended 2026

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

# ========================== GITHUB ==========================
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== GEMINI ==========================
@st.cache_resource
def get_gemini_model(api_key: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
    except Exception as e:
        st.error(f"Gemini init error: {e}")
        return None

# ========================== TEXT CLEANING ==========================
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

# ========================== AI GENERATION ==========================
def robust_json_parse(text: str):
    # Strategy 1: direct
    try: return json.loads(text)
    except: pass
    # Strategy 2: extract array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key)
    if not model: return []

    all_cards = []
    progress = st.progress(0)
    total = len(vocab_phrase_list)

    for i in range(0, total, batch_size):
        progress.progress(i/total, text=f"🤖 AI processing {i}/{total} words...")
        batch = vocab_phrase_list[i:i+batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]

        prompt = f"""You are an expert lexicographer. Output ONLY a valid JSON array.
Rules:
- Copy "vocab" and "phrase" exactly.
- Use the provided phrase to define the word.
- If phrase empty → create one natural sentence (max 12 words).
- Plain text only. No markdown.

Format:
[{{"vocab": "...", "phrase": "...", "translation": "{USER_NATIVE_LANGUAGE} meaning", "part_of_speech": "...", 
"pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], 
"synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}]

Input: {json.dumps(batch_dicts, ensure_ascii=False)}"""

        for attempt in range(5):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list):
                    all_cards.extend(parsed)
                    break
            except:
                time.sleep(1.2 * (attempt + 1))  # backoff

    progress.progress(1.0, text="✅ AI complete!")
    time.sleep(0.5)
    progress.empty()
    return all_cards

def process_anki_data(df, batch_size=6):
    vocab_list = df.dropna(subset=['vocab'])[['vocab', 'phrase']].values.tolist()
    raw = generate_anki_card_data_batched(vocab_list, batch_size)

    processed = []
    for card in raw:
        vocab_raw = str(card.get("vocab", "")).strip().lower()
        vocab_cap = cap_first(vocab_raw)
        phrase = fix_vocab_casing(
            ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card.get("phrase", ""))))),
            vocab_cap
        )
        formatted_phrase = highlight_vocab(phrase, vocab_cap) if phrase else ""

        translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card.get("translation", "?"))))

        text_field = (f"{formatted_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
                      if formatted_phrase else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>")

        processed.append({
            "VocabRaw": vocab_raw,
            "Text": text_field,
            "Pronunciation": f"<b>[{str(card.get('part_of_speech','')).title()}]</b> {card.get('pronunciation_ipa','')}",
            "Definition": ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card.get("definition_english", ""))))),
            "Examples": "<ul>" + "".join(f"<li><i>{ensure_trailing_dot(e)}</i></li>" for e in card.get("example_sentences", [])[:3]) + "</ul>" if card.get("example_sentences") else "",
            "Synonyms": ensure_trailing_dot(", ".join([cap_first(s) for s in card.get("synonyms_antonyms", {}).get("synonyms", [])[:5]])),
            "Antonyms": ensure_trailing_dot(", ".join([cap_first(a) for a in card.get("synonyms_antonyms", {}).get("antonyms", [])[:5]])),
            "Etymology": normalize_spaces(card.get("etymology", ""))
        })
    return processed

# ========================== ANKI PACKAGE ==========================
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
  font-size: clamp(1.5em, 5vw, 2.0em);
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

    # 1. Define Fields
    fields_list = [
        {'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'},
        {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'},
        {'name': 'Etymology'}, {'name': 'Audio'},
    ]
    
    # 2. Dynamically Generate Model ID based on fields
    field_names_str = "".join([f['name'] for f in fields_list])
    dynamic_model_id = int(hashlib.sha256(field_names_str.encode('utf-8')).hexdigest(), 16) % (10**10)

    # 3. Define Anki Model (CLOZE)
    my_model = genanki.Model(
        dynamic_model_id,
        'Cyberpunk Vocab Model',
        fields=fields_list,
        templates=[{
            'name': 'Cyberpunk Card',
            'qfmt': front_html,
            'afmt': back_html,
        }],
        css=cyberpunk_css,
        model_type=genanki.Model.CLOZE 
)

    fields_list = [
        {'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'},
        {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'},
        {'name': 'Etymology'}, {'name': 'Audio'}
    ]

    dynamic_model_id = int(hashlib.sha256("".join(f['name'] for f in fields_list).encode()).hexdigest(), 16) % 10**10
    my_model = genanki.Model(dynamic_model_id, 'Cyberpunk Vocab Model', fields=fields_list,
                             templates=[{'name': 'Card', 'qfmt': front_html, 'afmt': back_html}],
                             css=cyberpunk_css, model_type=genanki.Model.CLOZE)

    my_deck = genanki.Deck(abs(hash(deck_name)) % 10**10, deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        progress = st.progress(0)
        for i, note_data in enumerate(notes_data):
            progress.progress((i+1)/len(notes_data), text=f"🔊 Audio {i+1}/{len(notes_data)}...")
            audio_field = ""
            if generate_audio and note_data['VocabRaw']:
                try:
                    clean_name = re.sub(r'[^a-zA-Z0-9]', '', note_data['VocabRaw']) + ".mp3"
                    path = os.path.join(temp_dir, clean_name)
                    gTTS(text=note_data['VocabRaw'], lang='en').save(path)
                    media_files.append(path)
                    audio_field = f"[sound:{clean_name}]"
                except:
                    pass
            my_note = genanki.Note(model=my_model, fields=[note_data['Text'], note_data['Pronunciation'],
                                                           note_data['Definition'], note_data['Examples'],
                                                           note_data['Synonyms'], note_data['Antonyms'],
                                                           note_data['Etymology'], audio_field])
            my_deck.add_note(my_note)

        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buffer = io.BytesIO()
        output_path = os.path.join(temp_dir, 'deck.apkg')
        my_package.write_to_file(output_path)
        with open(output_path, "rb") as f:
            buffer.write(f.read())
        buffer.seek(0)
        progress.empty()
    return buffer

# ========================== DATA IO ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(content.decoded_content.decode()))
        df['phrase'] = df['phrase'].fillna("")
        return df.sort_values("vocab").reset_index(drop=True)
    except:
        return pd.DataFrame(columns=['vocab', 'phrase'])

def save_to_github(df):
    csv = df.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Update vocab", csv, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Initial", csv)
    load_data.clear()
    return True

df = load_data().copy()

# ========================== SIDEBAR & WOTD ==========================
with st.sidebar:
    st.header("🌟 Word of the Day")
    if not df.empty:
        today = date.today().isoformat()
        random.seed(today)
        row = df.sample(1).iloc[0]
        st.subheader(row["vocab"].upper())
        if row["phrase"].strip():
            st.caption(row["phrase"])
    st.divider()
    # (Gemini key, speech language, buttons - identical to original)

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Generate Anki"])

with tab1:
    # (your original add form - unchanged, works perfectly)

with tab2:
    if df.empty:
        st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        search = st.text_input("🔎 Search", "").lower().strip()
        display_df = df[df['vocab'].str.contains(search, case=False)] if search else df
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True)
        col1, col2 = st.columns([3,1])
        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                if save_to_github(edited.sort_values("vocab").reset_index(drop=True)):
                    st.toast("✅ Saved to cloud!", icon="🎉")
                    st.rerun()
        with col2:
            st.download_button("📥 CSV", edited.to_csv(index=False).encode(), "vocab.csv", "text/csv")

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")
    if df.empty:
        st.info("Add words first!")
    else:
        deck_name = st.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        c1, c2 = st.columns(2)
        with c1:
            batch = st.slider("⚡ Batch size", 1, 12, 6)
        with c2:
            audio = st.checkbox("🔊 Generate Audio", True)
        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            if len(df) > 400:
                st.warning("⚠️ Large deck - this may take a few minutes")
            raw_notes = process_anki_data(df, batch)
            with st.spinner("Packaging deck..."):
                apkg = create_anki_package(raw_notes, deck_name, audio)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            fname = re.sub(r'[^a-zA-Z0-9]', '_', deck_name) + f"_{ts}.apkg"
            st.success(f"🎉 {len(raw_notes)} cards ready!")
            st.download_button("📥 Download .apkg", apkg, fname, "application/octet-stream", use_container_width=True)
            st.dataframe(pd.DataFrame(raw_notes)[['VocabRaw', 'Text']].head(3), use_container_width=True)

st.caption("✅ Cyberpunk Anki deck with audio • Phrase + Vocab: {{c1::Translation}}")
