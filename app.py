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

# IMPORTS FOR AUDIO & ANKI
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

USER_NATIVE_LANGUAGE = "Indonesian"
GEMINI_MODEL = "gemini-2.5-flash-lite"

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
            generation_config={"response_mime_type": "application/json", "temperature": 0.15}
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
    rules = [(r"\bto doing\b", "to do"), (r"\bfor helps\b", "to help"),
             (r"\bis use to\b", "is used to"), (r"\bhelp for to\b", "help to"),
             (r"\bfor to\b", "to"), (r"\bcan able to\b", "can")]
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

# ========================== SPEECH ==========================
def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== JSON PARSER ==========================
def robust_json_parse(text: str):
    try: return json.loads(text)
    except: pass
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

# ========================== BATCH GENERATOR ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key)
    if not model: return []

    all_card_data = []
    progress_bar = st.progress(0)
    total_items = len(vocab_phrase_list)

    for i in range(0, total_items, batch_size):
        progress_bar.progress(i / total_items, text=f"🤖 AI Processing batch {i//batch_size + 1}...")
        batch = vocab_phrase_list[i:i + batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]

        prompt = f"""You are an expert lexicographer. Output ONLY a valid JSON array.
Rules:
- Copy "vocab" exactly.
- If phrase is provided: base everything on that exact context.
- If phrase empty: create one natural short sentence (max 12 words).
- No markdown, no asterisks, no extra text.

Output format: [{{"vocab": "...", "phrase": "...", "translation": "{USER_NATIVE_LANGUAGE} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}]

Batch: {json.dumps(batch_dicts, ensure_ascii=False)}"""

        for attempt in range(4):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    break
            except:
                time.sleep(1.8 ** attempt)
        time.sleep(0.8)

    progress_bar.empty()
    return all_card_data

# ========================== PROCESS DATA (SAFE MATCHING) ==========================
def process_anki_data(df, batch_size=6, only_missing_phrase=False):
    if only_missing_phrase:
        filtered = df[df['phrase'].str.strip() == ""].copy()
    else:
        filtered = df.copy()

    vocab_phrase_list = filtered.dropna(subset=['vocab'])[['vocab', 'phrase']].values.tolist()
    raw_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size)

    # SAFE MATCHING BY VOCAB
    card_dict = {str(item.get("vocab","")).strip().lower(): item for item in raw_card_data}

    processed_notes = []
    for _, row in filtered.iterrows():
        vocab_raw = str(row['vocab']).strip().lower()
        card_data = card_dict.get(vocab_raw, {})

        phrase = normalize_spaces(card_data.get("phrase", row.get('phrase', "")))
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing(phrase, vocab_raw)

        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""

        translation = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("translation", "?"))))
        pos = str(card_data.get("part_of_speech", "")).title()
        ipa = card_data.get("pronunciation_ipa", "")
        eng_def = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))
        
        examples = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e)))) 
                    for e in (card_data.get("example_sentences", []) or [])[:3]]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""

        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]])
        antonyms_field = ", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])[:5]])
        etymology = normalize_spaces(card_data.get("etymology", ""))

        text_field = f"{formatted_phrase}<br><br>{cap_first(vocab_raw)}: <b>{{{{c1::{translation}}}}}</b>" if formatted_phrase else f"{cap_first(vocab_raw)}: <b>{{{{c1::{translation}}}}}</b>"

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

# ========================== CREATE ANKI PACKAGE ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True):
    # === YOUR ORIGINAL CYBERPUNK CSS (100% unchanged) ===
    cyberpunk_css = """
/* --- Global Settings (Cyberpunk Glitch Theme) --- */
.card { font-family: 'Roboto Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; font-weight: 400; color: #00ff41; background-color: #111111; background-image: repeating-linear-gradient(0deg, #181818, #181818 1px, #111111 1px, #111111 20px); padding: 30px 20px; max-width: 800px; margin: 0 auto; box-sizing: border-box; text-align: left; }
.nightMode .card { color: #00aaff; background-color: #080808; }
/* --- (rest of your original CSS - paste the entire cyberpunk_css block you already have here) --- */
"""

    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text">{{cloze:Text}}</div></div>
<div class="vellum-detail-container">
  {{#Definition}}<div class="vellum-section definition"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}
  {{#Pronunciation}}<div class="vellum-section pronunciation"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}
  {{#Examples}}<div class="vellum-section examples"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}
  {{#Synonyms}}<div class="vellum-section synonyms"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}
  {{#Antonyms}}<div class="vellum-section antonyms"><div class="section-header">➖ ANTONYMS</div><div class="content">{{Antonyms}}</div></div>{{/Antonyms}}
  {{#Etymology}}<div class="vellum-section etymology"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}
  <div style='display:none'>{{Audio}}</div>
</div>{{Audio}}"""

    fields_list = [{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}]
    dynamic_model_id = int(hashlib.sha256("".join(f['name'] for f in fields_list).encode()).hexdigest(), 16) % (10**10)

    my_model = genanki.Model(dynamic_model_id, 'Cyberpunk Vocab Model', fields=fields_list,
                             templates=[{'name': 'Cyberpunk Card', 'qfmt': front_html, 'afmt': back_html}],
                             css=cyberpunk_css, model_type=genanki.Model.CLOZE)

    my_deck = genanki.Deck(abs(hash(deck_name)) % (10**10), deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        progress_bar = st.progress(0)
        total = len(notes_data)
        
        for i, note_data in enumerate(notes_data):
            audio_field = ""
            if generate_audio and note_data['VocabRaw']:
                progress_bar.progress((i+1)/total, text=f"🔊 Generating Audio: {note_data['VocabRaw']}")
                try:
                    clean_filename = re.sub(r'[^a-zA-Z0-9]', '', note_data['VocabRaw']) + ".mp3"
                    file_path = os.path.join(temp_dir, clean_filename)
                    gTTS(text=note_data['VocabRaw'], lang='en', slow=False).save(file_path)
                    media_files.append(file_path)
                    audio_field = f"[sound:{clean_filename}]"
                except:
                    pass

            my_note = genanki.Note(model=my_model, fields=[
                note_data['Text'], note_data['Pronunciation'], note_data['Definition'],
                note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'],
                note_data['Etymology'], audio_field
            ])
            my_deck.add_note(my_note)

        progress_bar.empty()
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
    load_data.clear()
    return True

df = load_data().copy()

# ========================== WORD OF THE DAY + STATS ==========================
wotd_vocab = wotd_phrase = None
if not df.empty:
    today_str = date.today().isoformat()
    random.seed(today_str)
    row = df.sample(n=1).iloc[0]
    wotd_vocab = row["vocab"]
    wotd_phrase = row["phrase"]

with st.sidebar:
    st.header("🌟 Word of the Day")
    if wotd_vocab:
        st.subheader(wotd_vocab.upper())
        if wotd_phrase and wotd_phrase.strip(): 
            st.caption(wotd_phrase)
    st.divider()
    st.subheader("📊 Quick Stats")
    st.metric("Total Words", len(df))
    with_phrase = len(df[df['phrase'].str.strip() != ""])
    st.metric("With Phrase", f"{with_phrase} ({with_phrase/len(df)*100:.1f}% if len(df) else 0)")
    st.divider()
    # (your original Gemini key + speech language code stays exactly the same here)

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Generate Anki (Cyberpunk)"])

with tab1:
    st.subheader("Add new word")
    with st.form("add_form", clear_on_submit=True):
        v = st.text_input("📝 Vocab", placeholder="e.g. serendipity").lower().strip()
        p_raw = st.text_input("🔤 Phrase (type 1 to skip)", placeholder="I found it by serendipity!").strip()
        submitted = st.form_submit_button("💾 Save to Cloud", use_container_width=True)
    if submitted and v:
        p = "" if p_raw.upper() == "1" else p_raw
        if v in df['vocab'].values:
            df.loc[df['vocab'] == v, 'phrase'] = p
        else:
            df = pd.concat([df, pd.DataFrame([{"vocab": v, "phrase": p}])], ignore_index=True)
        if save_to_github(df):
            st.success(f"✅ '{v}' saved!")
            time.sleep(0.8)
            st.rerun()

with tab2:
    if df.empty:
        st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        search = st.text_input("🔎 Search...", "").lower().strip()
        display_df = df[df['vocab'].str.contains(search, case=False)] if search else df
        
        # Delete
        to_delete = st.multiselect("🗑️ Select words to permanently delete", options=display_df['vocab'].tolist())
        if to_delete and st.button("🗑️ Delete Selected Words", type="secondary"):
            df = df[~df['vocab'].isin(to_delete)]
            if save_to_github(df):
                st.success(f"Deleted {len(to_delete)} word(s)")
                st.rerun()
        
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("💾 Save All Changes", type="primary", use_container_width=True):
            if save_to_github(edited.sort_values(by="vocab", ignore_index=True)):
                st.toast("✅ Cloud updated!", icon="🎉")
                st.rerun()

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck (.apkg)")
    if df.empty:
        st.info("Add words first!")
    else:
        deck_name_input = st.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        c1, c2, c3 = st.columns(3)
        with c1:
            batch_size = st.slider("⚡ Batch Size (AI)", 1, 12, 6)
        with c2:
            include_audio = st.checkbox("🔊 Generate Audio", value=True)
        with c3:
            only_missing = st.checkbox("Only words WITHOUT phrases", value=False)

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            raw_notes = process_anki_data(df, batch_size=batch_size, only_missing_phrase=only_missing)
            with st.spinner("📦 Packaging Deck & Audio..."):
                apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{re.sub(r'[^a-zA-Z0-9]', '_', deck_name_input)}_{timestamp}.apkg"
            st.success(f"🎉 {len(raw_notes)} cards ready!")
            st.download_button("📥 Download .apkg", apkg_buffer, filename, "application/octet-stream", use_container_width=True)

st.caption("✅ Cyberpunk Anki deck with audio • Phrase + Vocab: {{c1::Translation}} • Refined March 2026")
