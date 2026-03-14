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

# ========================== SETUP & CACHING ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")

@st.cache_resource
def get_github_repo(token, repo_name):
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except Exception as e:
        st.error(f"❌ GitHub connection failed: {e}")
        return None

# --- SECRETS MANAGEMENT ---
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_NAME = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

repo = get_github_repo(GITHUB_TOKEN, REPO_NAME)

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

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

# ========================== OPTIMIZED CLEANING ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    return s if s and s[-1] in ".!?" else (s + "." if s else "")

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip() if text else ""

def clean_grammar(text: str) -> str:
    if not isinstance(text, str) or not text: return text
    rules = {
        r"\bto doing\b": "to do", r"\bfor helps\b": "to help",
        r"\bis use to\b": "is used to", r"\bhelp for to\b": "help to",
        r"\bfor to\b": "to", r"\bcan able to\b": "can"
    }
    for pat, repl in rules.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text

def cap_each_sentence(text: str) -> str:
    if not isinstance(text, str) or not text: return text
    return ". ".join(cap_first(s) for s in re.split(r'(?<=[.!?])\s+', text) if s.strip())

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    return re.sub(r'\b' + re.escape(vocab) + r'\b', f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    return re.sub(r'\b' + re.escape(vocab.lower()) + r'\b', vocab, phrase, flags=re.IGNORECASE)

def robust_json_parse(text: str):
    try:
        return json.loads(text)
    except:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: return None
    return None

# ========================== DYNAMIC THROTTLED AI BATCHER ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=10):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []

    all_card_data = []
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    
    if "rpd_count" not in st.session_state:
        st.session_state.rpd_count = 0

    with st.status("🚀 Fast AI Generation...", expanded=True) as status_log:
        progress_bar = st.progress(0)
        
        for idx, batch in enumerate(batches):
            if st.session_state.rpd_count >= 20:
                st.warning("🛑 Daily AI Limit reached.")
                break
            
            start_time = time.time()
            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES: 1. Copy ALL fields. 2. '*' = Context Hint. 3. Empty Phrase = Generate example.
OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}]
INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

            success = False
            for attempt in range(2):
                try:
                    response = model.generate_content(prompt)
                    st.session_state.rpd_count += 1
                    parsed = robust_json_parse(response.text)
                    if isinstance(parsed, list):
                        all_card_data.extend(parsed)
                        success = True
                        break
                except Exception as e:
                    if "429" in str(e): time.sleep(15)
                    else: time.sleep(1)
            
            # Dynamic Throttle: Only wait if we are moving too fast for the 5 RPM limit
            elapsed = time.time() - start_time
            if idx < len(batches) - 1 and elapsed < 12:
                time.sleep(12 - elapsed)
                
            progress_bar.progress((idx + 1) / len(batches))

        status_log.update(label=f"✅ AI Done ({len(all_card_data)} items)", state="complete", expanded=False)
    
    return all_card_data

def process_anki_data(df_subset, batch_size=10):
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []

    for card_data in all_card_data:
        v_raw = str(card_data.get("vocab", "")).strip().lower()
        if not v_raw: continue
        
        # Fast Cleaning
        ph = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("phrase", "")))))
        ph = fix_vocab_casing(ph, v_raw)
        tr = ensure_trailing_dot(clean_grammar(normalize_spaces(card_data.get("translation", "?"))))
        pos = str(card_data.get("part_of_speech", "")).title()
        eng_def = ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(card_data.get("definition_english", "")))))
        
        ex_list = [ensure_trailing_dot(cap_each_sentence(clean_grammar(normalize_spaces(e)))) for e in (card_data.get("example_sentences", []) or [])[:3]]
        ex_field = "<ul>" + "".join(f"<li><i>{e}</i></li>" for e in ex_list) + "</ul>" if ex_list else ""
        
        syn_ant = card_data.get("synonyms_antonyms", {}) or {}
        syns = ", ".join([cap_first(s) for s in (syn_ant.get("synonyms", []) or [])[:5]])
        
        processed_notes.append({
            "VocabRaw": v_raw,
            "Text": f"{highlight_vocab(ph, v_raw)}<br><br>{cap_first(v_raw)}: <b>{{{{c1::{tr}}}}}</b>" if ph else f"{cap_first(v_raw)}: <b>{{{{c1::{tr}}}}}</b>",
            "Pronunciation": f"<b>[{pos}]</b> {card_data.get('pronunciation_ipa', '')}",
            "Definition": eng_def,
            "Examples": ex_field,
            "Synonyms": ensure_trailing_dot(syns),
            "Antonyms": ensure_trailing_dot(", ".join([cap_first(a) for a in (syn_ant.get("antonyms", []) or [])[:5]])),
            "Etymology": normalize_spaces(card_data.get("etymology", ""))
        })
    return processed_notes

# ========================== ASYNC AUDIO ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        fn = re.sub(r'[^a-zA-Z0-9]', '', vocab) + ".mp3"
        fp = os.path.join(temp_dir, fn)
        if vocab.strip():
            gTTS(text=vocab, lang='en', slow=False).save(fp)
            return vocab, fn, fp
    except: pass
    return vocab, None, None

# ========================== ANKI LOGIC ==========================
CYBERPUNK_CSS = """
.card { font-family: 'Roboto Mono', monospace; font-size: 18px; color: #00ff41; background-color: #111; padding: 30px; }
.vellum-focus-container { background: #0d0d0d; padding: 25px; border: 2px solid #00ff41; box-shadow: 0 0 10px #00ff41; text-align: center; }
.prompt-text { font-size: 1.8em; font-weight: 900; color: #fff; text-shadow: 1px 1px 0 #ff00ff; }
.cloze { color: #111; background-color: #00ff41; }
.solved-text .cloze { color: #ff00ff; background: none; border-bottom: 2px solid #00ffff; }
.vellum-section { margin-top: 15px; border-bottom: 1px dashed #444; }
.section-header { font-weight: bold; color: #00ffff; }
"""

def create_anki_package(notes_data, deck_name, generate_audio=True):
    model = genanki.Model(1607392319, 'Cyberpunk Vocab', fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], templates=[{'name': 'Card 1', 'qfmt': '<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>', 'afmt': '<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFITION</div><div>{{Definition}}</div></div>{{/Definition}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div>{{Examples}}</div></div>{{/Examples}}</div>{{Audio}}'}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            unique_v = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(generate_audio_file, v, temp_dir) for v in unique_v]
                for f in concurrent.futures.as_completed(futures):
                    v, fn, fp = f.result()
                    if fn: media_files.append(fp); audio_map[v] = f"[sound:{fn}]"
        
        for n in notes_data:
            deck.add_note(genanki.Note(model=model, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")]))
        
        output_path = os.path.join(temp_dir, 'out.apkg')
        pkg = genanki.Package(deck)
        pkg.media_files = media_files
        pkg.write_to_file(output_path)
        with open(output_path, "rb") as f: return io.BytesIO(f.read())

# ========================== GITHUB DATA OPS ==========================
@st.cache_data(ttl=300)
def load_data():
    try:
        content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        df['status'] = df.get('status', 'New')
        return df.sort_values(by="vocab", ignore_index=True)
    except: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        f = repo.get_contents("vocabulary.csv")
        repo.update_file(f.path, "⚡ Fast Update", csv_data, f.sha)
    except: repo.create_file("vocabulary.csv", "Init", csv_data)
    load_data.clear(); return True

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()

# ========================== UI & FORMS ==========================
st.title("📚 My Cloud Vocab")

# Mobile Keyboard Fix
st.components.v1.html("""<script>const doc = window.parent.document; doc.addEventListener('keydown', function(e) { if (e.key === 'Enter' && e.target.tagName === 'INPUT') { setTimeout(() => { e.target.blur(); }, 50); } }, true);</script>""", height=0)

tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Anki"])

with tab1:
    with st.form("add_form", clear_on_submit=True):
        f_phrase = st.text_input("🔤 Phrase / Context", placeholder="Paste sentence...")
        f_vocab = st.text_input("📝 Vocab Word", placeholder="e.g. ephemeral")
        submitted = st.form_submit_button("💾 Save to Cloud", use_container_width=True)
        
        if submitted:
            v = f_vocab.lower().strip()
            if v:
                p = cap_first(ensure_trailing_dot(clean_grammar(f_phrase.strip())))
                new_entry = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New"}])
                st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_entry]).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(st.session_state.vocab_df)
                st.success(f"Saved {v}!")
                st.rerun()

    st.divider()
    with st.expander("📥 Bulk Import"):
        bulk_text = st.text_area("Format: word, phrase (one per line)")
        if st.button("Bulk Process"):
            lines = [l.split(',', 1) for l in bulk_text.split('\n') if ',' in l]
            if lines:
                new_rows = [{"vocab": l[0].strip().lower(), "phrase": l[1].strip(), "status": "New"} for l in lines]
                st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, pd.DataFrame(new_rows)]).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(st.session_state.vocab_df)
                st.rerun()

with tab2:
    if not st.session_state.vocab_df.empty:
        search = st.text_input("🔎 Search...").lower()
        df_view = st.session_state.vocab_df
        if search: df_view = df_view[df_view['vocab'].str.contains(search)]
        
        edited = st.data_editor(df_view, num_rows="dynamic", use_container_width=True, hide_index=True)
        if st.button("💾 Sync Changes"):
            st.session_state.vocab_df = edited
            save_to_github(edited)
            st.toast("Synced!")

with tab3:
    col1, col2 = st.columns(2)
    with col1: TARGET_LANG = st.selectbox("🎯 Lang", ["Indonesian", "Spanish", "French", "German", "English"])
    with col2: GEMINI_MODEL = st.selectbox("🤖 Model", ["gemini-2.0-flash-exp", "gemini-2.5-flash-lite"])
    
    batch_size = st.slider("⚡ Speed (Words/Req)", 1, 15, 12)
    
    if st.button("🚀 Generate & Download", type="primary", use_container_width=True):
        subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']
        if not subset.empty:
            raw_notes = process_anki_data(subset, batch_size=batch_size)
            if raw_notes:
                buf = create_anki_package(raw_notes, "Vocabulary Deck")
                st.download_button("📥 Download .apkg", buf, "deck.apkg", "application/octet-stream", use_container_width=True)
                
                # Fast Vectorized Status Update
                processed_list = [n['VocabRaw'] for n in raw_notes]
                st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(processed_list), 'status'] = 'Done'
                save_to_github(st.session_state.vocab_df)
        else:
            st.warning("No 'New' words.")

with st.sidebar:
    st.caption(f"Last Sync: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Force Refresh"):
        load_data.clear()
        st.session_state.vocab_df = load_data()
        st.rerun()
