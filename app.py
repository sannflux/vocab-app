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

# --- GLOBAL SESSION STATE INITIALIZATION ---
if "rpd_count" not in st.session_state: st.session_state.rpd_count = 0
if "TARGET_LANG" not in st.session_state: st.session_state.TARGET_LANG = "Indonesian"
if "GEMINI_MODEL" not in st.session_state: st.session_state.GEMINI_MODEL = "gemini-2.5-flash-lite"
if "input_phrase" not in st.session_state: st.session_state.input_phrase = ""
if "input_vocab" not in st.session_state: st.session_state.input_vocab = ""
if "last_v_selected" not in st.session_state: st.session_state.last_v_selected = ""

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

def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== ASYNC BATCH GENERATOR ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, st.session_state.GEMINI_MODEL)
    if not model: return []

    all_card_data = []
    total_items = len(vocab_phrase_list)
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, total_items, batch_size)]

    with st.status("🤖 Processing AI Batches (RPM Throttled)...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        
        for idx, batch in enumerate(batches):
            if st.session_state.rpd_count >= 20:
                st.warning("🛑 Daily AI Limit (20 requests) reached. Please try again tomorrow.")
                break
            
            # RPM (Requests Per Minute) Throttle
            if idx > 0:
                for remaining in range(12, 0, -1):
                    progress_bar.progress(idx / len(batches), text=f"⏳ Throttling for RPM limits... ({remaining}s)")
                    time.sleep(1)

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES:
1. 'vocab': Keep exactly as provided.
2. 'translation': Provide a CONCISE (max 3 words) lexical meaning in {st.session_state.TARGET_LANG} of the 'vocab' ONLY. 
3. IMPORTANT: DO NOT translate the full 'phrase' into the 'translation' field. If 'vocab' is an idiom (e.g. 'piece of cake'), translate the idiom's meaning (e.g. 'Sangat mudah'), NOT the full sentence.
4. 'definition_english': A simple dictionary definition.
5. 'etymology': History of the word. Use asterisks (*) for reconstructed proto-words as per linguistic standards.
6. 'sentence_venting': (Hidden field) You may put the full translation of the 'phrase' here to help your internal logic, but it won't be used.
OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "Lexical meaning only", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "...", "sentence_venting": "Full sentence translation"}}]
BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

            vocab_words = [v[0] for v in batch]
            success = False
            for attempt in range(3):
                try:
                    response = model.generate_content(prompt)
                    st.session_state.rpd_count += 1
                    parsed = robust_json_parse(response.text)
                    if isinstance(parsed, list):
                        all_card_data.extend(parsed)
                        st.markdown(f"✅ **Processed**: `{', '.join(vocab_words)}`")
                        success = True
                        break
                except Exception as e:
                    if "429" in str(e): time.sleep(20)
                    else: time.sleep(2)
            
            if not success:
                st.markdown(f"❌ **Failed**: `{', '.join(vocab_words)}` (Skipping)")
            
            progress_bar.progress((idx + 1) / len(batches))
        status_log.update(label=f"✅ AI Generation Complete!", state="complete", expanded=False)
    
    return all_card_data

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
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing(phrase, vocab_raw)
        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""
        
        # Translation: If AI still provides a sentence, we take the first few words or hope the 'venting' fix worked
        translation = normalize_spaces(card_data.get("translation", "?"))
        
        pos = str(card_data.get("part_of_speech", "")).title()
        ipa = card_data.get("pronunciation_ipa", "")
        eng_def = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))
        examples = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e)))) for e in (card_data.get("example_sentences", []) or [])[:3]]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""
        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]])
        antonyms_field = ", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])[:5]])
        etymology = normalize_spaces(card_data.get("etymology", ""))
        
        text_field = f"{formatted_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>" if formatted_phrase else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"
        processed_notes.append({"VocabRaw": vocab_raw, "Text": text_field, "Pronunciation": pronunciation_field, "Definition": eng_def, "Examples": examples_field, "Synonyms": synonyms_field, "Antonyms": antonyms_field, "Etymology": etymology})
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
    except Exception as e: print(f"Audio error: {e}")
    return vocab, None, None

# ========================== CSS / PREVIEW ==========================
CYBERPUNK_CSS = """
.card { font-family: 'Roboto Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; color: #00ff41; background-color: #111111; background-image: repeating-linear-gradient(0deg, #181818, #181818 1px, #111111 1px, #111111 20px); padding: 30px 20px; text-align: left; }
.vellum-focus-container { background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px; border: 2px solid #00ff41; box-shadow: 0 0 5px #00ff41, 0 0 15px rgba(0, 255, 65, 0.4); text-align: center; }
.prompt-text { font-family: 'Electrolize', sans-serif; font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }
.cloze { color: #111111; background-color: #00ff41; padding: 2px 4px; }
.solved-text .cloze { color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }
.vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed #00ff41; }
.section-header { font-weight: 600; color: #00ffff; border-left: 3px solid #00ff41; padding-left: 10px; }
.content { color: #aaffaa; padding-left: 13px; }
"""

# ========================== GENANKI LOGIC ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""
    my_model = genanki.Model(1607392319, 'Cyberpunk Vocab Model', fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_vocab = {executor.submit(generate_audio_file, v, temp_dir): v for v in unique_vocabs}
                for future in concurrent.futures.as_completed(future_to_vocab):
                    vk, fn, fp = future.result()
                    if fn: media_files.append(fp); audio_map[vk] = f"[sound:{fn}]"
        for note_data in notes_data:
            my_deck.add_note(genanki.Note(model=my_model, fields=[note_data['Text'], note_data['Pronunciation'], note_data['Definition'], note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'], note_data['Etymology'], audio_map.get(note_data['VocabRaw'], "")]))
        my_package = genanki.Package(my_deck); buffer = io.BytesIO(); output_path = os.path.join(temp_dir, 'output.apkg')
        my_package.media_files = media_files; my_package.write_to_file(output_path)
        with open(output_path, "rb") as f: buffer.write(f.read())
        buffer.seek(0)
    return buffer

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna(""); df['status'] = df.get('status', 'New')
        return df.sort_values(by="vocab", ignore_index=True)
    except: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv"); repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear(); return True

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()

# ========================== CALLBACKS ==========================
def save_single_word_callback():
    v = st.session_state.input_vocab.lower().strip()
    p_raw = st.session_state.input_phrase
    if v:
        p = p_raw.strip()
        if p and p != "1" and not p.startswith("*"):
            if p.endswith(","): p = p[:-1] + "."
            elif not p.endswith((".", "!", "?")): p += "."
            p = p.capitalize()
        mask = st.session_state.vocab_df['vocab'] == v
        if not st.session_state.vocab_df.empty and mask.any():
            st.session_state.vocab_df.loc[mask, ['phrase', 'status']] = [p, 'New']
        else:
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, pd.DataFrame([{"vocab": v, "phrase": p, "status": "New"}])], ignore_index=True)
        save_to_github(st.session_state.vocab_df)
        st.session_state.input_phrase = ""; st.session_state.input_vocab = ""
        st.session_state.save_message = f"✅ Saved '{v}'!"
    else: st.session_state.save_error = "⚠️ Enter a vocabulary word."

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    lang_opts = ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"]
    st.session_state.TARGET_LANG = st.selectbox("🎯 Language", lang_opts, index=lang_opts.index(st.session_state.TARGET_LANG))
    model_opts = ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"]
    st.session_state.GEMINI_MODEL = st.selectbox("🤖 Model", model_opts, index=model_opts.index(st.session_state.GEMINI_MODEL))
    st.divider(); st.metric("Daily AI Usage", f"{st.session_state.rpd_count}/20")
    if not st.session_state.vocab_df.empty:
        st.download_button("💾 Backup CSV", st.session_state.vocab_df.to_csv(index=False).encode('utf-8'), f"vocab_backup_{date.today()}.csv", "text/csv")

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Generate"])

with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")
    if "save_message" in st.session_state: st.success(st.session_state.save_message); del st.session_state.save_message
    if "save_error" in st.session_state: st.error(st.session_state.save_error); del st.session_state.save_error
    if add_mode == "Single":
        p_raw = st.text_input("🔤 Phrase", placeholder="Paste sentence...", key="input_phrase")
        v_selected = ""
        if p_raw and p_raw not in ["1", "*"]:
            clean_text = re.sub(r'[^\w\s\-\']', '', p_raw)
            unique_words = list(dict.fromkeys([w.lower() for w in clean_text.split() if w.strip()]))
            if unique_words:
                st.caption("Select Vocab:")
                selected_pills = st.pills("Pills", unique_words, selection_mode="multi", label_visibility="collapsed")
                v_selected = " ".join(selected_pills) if selected_pills else ""
        if v_selected != st.session_state.last_v_selected:
            st.session_state.input_vocab = v_selected; st.session_state.last_v_selected = v_selected
        st.text_input("📝 Vocab", key="input_vocab")
        st.button("💾 Save", type="primary", use_container_width=True, on_click=save_single_word_callback)
    else: 
        bulk_text = st.text_area("Bulk Paste (word, phrase)", height=150)
        if st.button("💾 Process Bulk", type="primary"):
            lines = [l.strip() for l in bulk_text.split('\n') if l.strip()]
            new_rows = [{"vocab": l.split(',', 1)[0].strip().lower(), "phrase": l.split(',', 1)[1].strip() if ',' in l else "", "status": "New"} for l in lines if l]
            if new_rows:
                st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, pd.DataFrame(new_rows)]).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(st.session_state.vocab_df); st.rerun()

with tab2:
    if st.session_state.vocab_df.empty: st.info("Empty!")
    else:
        search = st.text_input("🔎 Search", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        edited = st.data_editor(display_df, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("💾 Save Changes", type="primary"):
            st.session_state.vocab_df = edited; save_to_github(st.session_state.vocab_df); st.rerun()

with tab3:
    st.subheader("Generate Deck")
    if st.session_state.vocab_df.empty: st.info("Empty!")
    else:
        st.info(f"AI Quota: {20 - st.session_state.rpd_count} left.")
        deck_name = st.text_input("📦 Deck Name", "-English::Vocabulary")
        batch_size = st.slider("⚡ Batch Size", 1, 15, 10)
        include_audio = st.checkbox("🔊 Audio", value=True)
        if st.button("🚀 Generate", type="primary", use_container_width=True):
            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']
            if subset.empty: st.warning("No 'New' words!")
            else:
                raw_notes = process_anki_data(subset, batch_size=batch_size)
                if raw_notes:
                    buf = create_anki_package(raw_notes, deck_name, generate_audio=include_audio)
                    st.session_state.anki_package = buf.getvalue()
                    st.session_state.anki_filename = f"Deck_{datetime.now().strftime('%H%M')}.apkg"
                    processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                    st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(processed_vocabs), 'status'] = 'Done'
                    save_to_github(st.session_state.vocab_df); st.success("Ready!")
        if "anki_package" in st.session_state:
            st.download_button("📥 Download", st.session_state.anki_package, st.session_state.anki_filename, use_container_width=True)
