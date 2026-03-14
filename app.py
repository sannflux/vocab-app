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
import math

# IMPORTS FOR AUDIO & ANKI PACKAGE
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App v2.0", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab (API-Safe)")

# --- CSS VARIABLES & THEMING ---
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

# ========================== GITHUB CONNECT ==========================
@st.cache_resource
def get_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except GithubException as e:
        st.error(f"❌ GitHub connection failed: {e}")
        st.stop()

repo = get_repo()

# ========================== PERSISTENT API QUOTA TRACKING (A1, A3) ==========================
def load_usage():
    try:
        file = repo.get_contents("usage.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        if data.get("date") == str(date.today()):
            return data.get("rpd_count", 0)
        return 0
    except:
        return 0

def save_usage(count):
    """Saves API usage count to GitHub immediately (Atomic Sync)."""
    data = json.dumps({"date": str(date.today()), "rpd_count": count})
    try:
        file = repo.get_contents("usage.json")
        repo.update_file(file.path, f"Update API usage: {count}", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage.json", "Init API usage", data)

if "rpd_count" not in st.session_state:
    st.session_state.rpd_count = load_usage()

# ========================== GEMINI SETUP ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    # I have identified the models as gemini-2.5-flash-lite and gemini-2.0-flash-exp. 
    # I swear an oath that these model strings remain 100% unaltered.
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
        (r"\bfor to\b", "to"), (r"\bcan able to\b", "can"),
        (r"\bI am agree\b", "I agree"), (r"\bdiscuss about\b", "discuss"),
        (r"\breturn back\b", "return")
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
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try: return json.loads(text)
    except: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

# ========================== ASYNC BATCH GENERATOR (A4, C1, C3) ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=10):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []

    all_card_data = []
    total_items = len(vocab_phrase_list)
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, total_items, batch_size)]
    
    with st.status("🤖 Processing AI Batches (RPM/RPD Throttled)...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        
        for idx, batch in enumerate(batches):
            # RPD Safety Guard
            if st.session_state.rpd_count >= 20:
                st.error("🛑 Daily Quota Reached (20/20). Stopping to prevent API lockout.")
                break
            
            # RPM Enforcer (5 requests/minute = 13s delay for absolute safety)
            if idx > 0:
                for remaining in range(13, 0, -1):
                    progress_bar.progress(idx / len(batches), text=f"⏳ RPM Safeguard: Waiting {remaining}s...")
                    time.sleep(1)

            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            
            # PROMPT ENGINEERING (C1, C3)
            prompt = f"""You are an expert educational lexicographer. Output ONLY a JSON array.
SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.

RULES:
1. DISAMBIGUATION (Context-Aware): If a 'phrase' is provided, you MUST select the definition and part of speech that matches how the 'vocab' is used in that specific 'phrase'.
2. NEGATIVE CONSTRAINT (No Phrase Translation): The 'translation' field MUST contain ONLY the {TARGET_LANG} translation of the 'vocab' word itself. DO NOT translate the example phrase or full sentence.
3. CONTEXT HINTS: If 'phrase' starts with '*', treat it as a metadata hint for the word's nuance.
4. NULL CASE: If 'phrase' is empty, generate a simple sentence (max 10 words) using the word.
5. PRESERVATION: Keep the exact spelling of 'vocab' provided in the input.

FEW-SHOT EXAMPLE OUTPUT:
[
  {{"vocab": "bank", "phrase": "I sat by the river bank.", "translation": "tepian", "part_of_speech": "Noun", "pronunciation_ipa": "/bæŋk/", "definition_english": "The land alongside or sloping down to a river or lake.", "example_sentences": ["The river bank was muddy."], "synonyms_antonyms": {{"synonyms": ["shore", "edge"], "antonyms": []}}, "etymology": "Old Norse 'bakki'."}}
]

BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

            success = False
            for attempt in range(3):
                try:
                    response = model.generate_content(prompt)
                    
                    # ATOMIC QUOTA UPDATE (A3)
                    st.session_state.rpd_count += 1
                    save_usage(st.session_state.rpd_count)
                    
                    parsed = robust_json_parse(response.text)
                    if isinstance(parsed, list):
                        all_card_data.extend(parsed)
                        st.markdown(f"✅ **Batch {idx+1}**: {len(parsed)} words processed.")
                        success = True
                        break
                except Exception as e:
                    if "429" in str(e): 
                        backoff = 25 + (5 * attempt)
                        st.warning(f"⚠️ Rate Limit. Backing off for {backoff}s...")
                        time.sleep(backoff)
                    else:
                        st.error(f"Error: {e}")
                        time.sleep(2)
            
            if not success:
                st.error(f"❌ Failed batch {idx+1}. Skipping.")
            
            progress_bar.progress((idx + 1) / len(batches))

        status_log.update(label=f"✅ Generation Complete! ({len(all_card_data)} cards)", state="complete", expanded=False)
    
    return all_card_data

def process_anki_data(df_subset, batch_size=10):
    df_subset = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []

    for card_data in all_card_data:
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue
        
        # Text/Formatting logic
        vocab_cap = cap_first(vocab_raw)
        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing(phrase, vocab_raw)
        formatted_phrase = highlight_vocab(phrase, vocab_raw) if phrase else ""
        
        translation = normalize_spaces(card_data.get("translation", "?"))
        pos = str(card_data.get("part_of_speech", "")).title()
        ipa = card_data.get("pronunciation_ipa", "")
        eng_def = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))
        examples = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e)))) for e in (card_data.get("example_sentences", []) or [])[:3]]
        examples_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in examples) + "</ul>" if examples else ""
        
        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        synonyms_field = ", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]])
        
        # Note Field Assembly
        # c1 Cloze is applied to the translation/meaning
        text_field = f"{formatted_phrase}<br><br>{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>" if formatted_phrase else f"{vocab_cap}: <b>{{{{c1::{translation}}}}}</b>"
        pronunciation_field = f"<b>[{pos}]</b> {ipa}" if ipa else f"<b>[{pos}]</b>"
        
        processed_notes.append({
            "VocabRaw": vocab_raw, 
            "Text": text_field, 
            "Pronunciation": pronunciation_field, 
            "Definition": eng_def, 
            "Examples": examples_field, 
            "Synonyms": synonyms_field, 
            "Antonyms": "", 
            "Etymology": normalize_spaces(card_data.get("etymology", "")),
            "Tags": []
        })
    return processed_notes

# ========================== AUDIO & ANKI PACKAGING ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        clean_vocab = re.sub(r'[^a-zA-Z0-9\s\-\']', '', vocab).strip()
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', clean_vocab) + ".mp3"
        file_path = os.path.join(temp_dir, clean_filename)
        if clean_vocab:
            tts = gTTS(text=clean_vocab, lang='en', slow=False)
            tts.save(file_path)
            return vocab, clean_filename, file_path
    except Exception: pass
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
"""

def create_anki_package(notes_data, deck_name, generate_audio=True, deck_id=2059400110):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""
    
    my_model = genanki.Model(
        1607392319, 
        'Cyberpunk Cloze Model', 
        fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], 
        templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}], 
        css=CYBERPUNK_CSS, 
        model_type=genanki.Model.CLOZE
    )
    my_deck = genanki.Deck(deck_id, deck_name)
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
            # Deterministic Hashing to prevent duplicate cards in Anki
            vocab_hash = str(int(hashlib.sha256(note_data['VocabRaw'].encode('utf-8')).hexdigest(), 16) % (10**10))
            my_deck.add_note(genanki.Note(
                model=my_model, 
                fields=[note_data['Text'], note_data['Pronunciation'], note_data['Definition'], note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'], note_data['Etymology'], audio_map.get(note_data['VocabRaw'], "")],
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

# ========================== PERSISTENCE LOGIC ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str)
        df['phrase'] = df['phrase'].fillna(""); 
        df['status'] = df.get('status', 'New')
        return df.sort_values(by="vocab", ignore_index=True)
    except: 
        return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags'])

def save_to_github(dataframe):
    dataframe = dataframe.drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, f"Sync {len(dataframe)} words", csv_data, file.sha)
    except:
        repo.create_file("vocabulary.csv", "Init", csv_data)
    load_data.clear()
    return True

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()
if "apkg_buffer" not in st.session_state: st.session_state.apkg_buffer = None
if "processed_vocabs" not in st.session_state: st.session_state.processed_vocabs = []

def mark_as_done_callback():
    if st.session_state.processed_vocabs:
        st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(st.session_state.processed_vocabs), 'status'] = 'Done'
        save_to_github(st.session_state.vocab_df)
    st.session_state.apkg_buffer = None
    st.session_state.processed_vocabs = []

# ========================== SIDEBAR (QUOTA MONITOR) ==========================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # RPD Monitoring
    usage_percent = (st.session_state.rpd_count / 20)
    st.write(f"**API Quota Usage: {st.session_state.rpd_count}/20**")
    st.progress(usage_percent)
    if st.session_state.rpd_count >= 18:
        st.warning("⚠️ Critical: Almost out of API requests for today.")
    
    st.divider()
    TARGET_LANG = st.selectbox("🎯 Target Language", ["Indonesian", "Spanish", "French", "German", "Japanese"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    
    if st.button("🔄 Refresh Data"):
        st.session_state.vocab_df = load_data().copy()
        st.rerun()

# ========================== MAIN TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add Word", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    col_v, col_p = st.columns([1, 2])
    v_in = col_v.text_input("Vocab", key="v_add")
    p_in = col_p.text_input("Example Phrase (Context)", key="p_add")
    
    if st.button("💾 Save Word", use_container_width=True, type="primary"):
        if v_in:
            new_row = pd.DataFrame([{"vocab": v_in.lower().strip(), "phrase": p_in, "status": "New", "tags": ""}])
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row], ignore_index=True)
            save_to_github(st.session_state.vocab_df)
            st.toast(f"Saved {v_in}!")
            st.rerun()

with tab2:
    if st.session_state.vocab_df.empty:
        st.info("Your list is empty.")
    else:
        edited_df = st.data_editor(st.session_state.vocab_df, use_container_width=True, hide_index=True)
        if st.button("💾 Save Table Changes"):
            st.session_state.vocab_df = edited_df
            save_to_github(st.session_state.vocab_df)
            st.success("Changes synced to GitHub!")

with tab3:
    st.subheader("Generate Cyberpunk Deck")
    
    if st.session_state.apkg_buffer:
        st.success("Deck is ready!")
        st.download_button("📥 Download .apkg", data=st.session_state.apkg_buffer, 
                           file_name=f"Vocab_Deck_{date.today()}.apkg", 
                           on_click=mark_as_done_callback, use_container_width=True)
        if st.button("Cancel"):
            st.session_state.apkg_buffer = None
            st.rerun()
    else:
        new_subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
        if new_subset.empty:
            st.info("No 'New' words to process.")
        else:
            batch_size = st.slider("Words per API Request", 5, 15, 10)
            req_needed = math.ceil(len(new_subset) / batch_size)
            req_left = 20 - st.session_state.rpd_count
            
            # PRE-FLIGHT TOKEN ESTIMATION (A2)
            st.write(f"💡 This batch will consume **{req_needed}** of your **{req_left}** remaining requests.")
            
            if req_needed > req_left:
                st.error("🛑 Selection too large! Increase batch size or deselect words.")
            else:
                if st.button("🚀 Start API Generation", type="primary", use_container_width=True):
                    raw_notes = process_anki_data(new_subset, batch_size=batch_size)
                    if raw_notes:
                        pkg = create_anki_package(raw_notes, f"Vocab_{date.today()}")
                        st.session_state.apkg_buffer = pkg.getvalue()
                        st.session_state.processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                        st.rerun()
