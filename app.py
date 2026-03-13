import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import random
from datetime import date, datetime, timedelta
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

# ========================== GITHUB CONNECT ==========================
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== LOAD / SAVE WITH SESSION STATE ==========================
@st.cache_data(ttl=600)
def load_data_from_github():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        if 'status' not in df.columns: df['status'] = 'New'
        if 'date_added' not in df.columns: df['date_added'] = date.today().isoformat()
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404: return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'date_added'])
        else: st.error(f"❌ CRITICAL: GitHub Error {e.status}"); st.stop()
    except Exception as e:
        st.error(f"❌ CRITICAL: CSV Error. {e}"); st.stop()

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data_from_github.clear()
    return True

# Initialize Session State Dataframe (Instant UI, Background Sync)
if "vocab_df" not in st.session_state:
    st.session_state.vocab_df = load_data_from_github().copy()

df = st.session_state.vocab_df

# Smart Merge Logic (Idea 8)
def merge_new_words(existing_df, new_rows):
    new_df = pd.DataFrame(new_rows)
    for _, row in new_df.iterrows():
        v = str(row['vocab']).strip().lower()
        p = str(row['phrase']).strip()
        
        if v in existing_df['vocab'].values:
            existing_idx = existing_df.index[existing_df['vocab'] == v][0]
            existing_p = str(existing_df.at[existing_idx, 'phrase']).strip()
            
            if p and p not in existing_p:
                merged_p = f"{existing_p} | {p}".strip(" |")
                existing_df.at[existing_idx, 'phrase'] = merged_p
                
            existing_df.at[existing_idx, 'status'] = 'New'
            existing_df.at[existing_idx, 'date_added'] = date.today().isoformat()
        else:
            row_df = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "date_added": date.today().isoformat()}])
            existing_df = pd.concat([existing_df, row_df], ignore_index=True)
    return existing_df

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_LANG = st.selectbox("🎯 Definition Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    
    st.divider()
    
    # Idea 6: Automated Metrics
    st.header("📈 Learning Velocity")
    try:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        seven_days_ago = pd.Timestamp(date.today() - timedelta(days=7))
        recent_count = len(df[df['date_added'] >= seven_days_ago])
        st.metric("Words Added (Last 7 Days)", recent_count)
        df['date_added'] = df['date_added'].dt.strftime('%Y-%m-%d') # Convert back for clean display
    except:
        pass

    st.divider()
    st.header("🔑 API Key")
    alt_key = st.text_input("Alternative key", type="password", value="", placeholder="AIzaSy...")
    if alt_key and alt_key != st.session_state.gemini_key:
        st.session_state.gemini_key = alt_key
        st.success("✅ Switched!")
        st.rerun()

    st.divider()
    if not df.empty:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Backup Database", csv_full, f"vocab_backup_{date.today()}.csv", "text/csv")
    
    st.divider()
    st.header("🌟 Word of the Day")
    if not df.empty:
        random.seed(date.today().isoformat())
        try:
            row = df.sample(n=1).iloc[0]
            st.subheader(row["vocab"].upper())
            if row["phrase"]: st.caption(row["phrase"])
            safe_text = row["vocab"].replace('"', '\\"').replace("'", "\\'")
            js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="en-US";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
            if st.button("🔊 Pronounce"): st.components.v1.html(js, height=0)
        except: pass

# ========================== GEMINI ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json", "temperature": 0.1})
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
    return re.sub(r'\b' + re.escape(vocab) + r'\b', f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

def fix_vocab_casing(phrase: str, vocab: str) -> str:
    if not phrase or not vocab: return phrase
    return re.sub(r'\b' + re.escape(vocab.lower()) + r'\b', vocab, phrase, flags=re.IGNORECASE)

def robust_json_parse(text: str):
    try: return json.loads(text)
    except: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

# ========================== ASYNC BATCH GENERATOR (Idea 4) ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []

    all_card_data = []
    total_items = len(vocab_phrase_list)
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, total_items, batch_size)]
    
    progress_bar = st.progress(0, text="🤖 Initializing AI threads...")
    completed_items = 0

    def fetch_batch(batch):
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES: 
1. Copy ALL fields exactly. 
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT. Use this hint to pick the specific definition, but generate a NEW sentence for the final 'phrase' field.
3. IF 'phrase' is normal text: Define based on that usage.
4. IF 'phrase' is empty: Generate ONE simple sentence (max 12 words) using the most common definition.
5. EXACT vocab unchanged.
NEVER use markdown. Plain text only.
OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "Plain text only."}}]
BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

        for attempt in range(4):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list): return parsed
            except Exception as e:
                time.sleep((2 ** attempt) + 1)
        return []

    # Using ThreadPoolExecutor to async generate AI definitions
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_batch = {executor.submit(fetch_batch, b): b for b in batches}
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            result = future.result()
            all_card_data.extend(result)
            completed_items += len(batch)
            progress_bar.progress(min(completed_items / total_items, 1.0), text=f"🤖 AI Processing {completed_items}/{total_items} words...")

    progress_bar.empty()
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
        phrase = fix_vocab_casing(ensure_trailing_dot(cap_each_sentence(clean_grammar(phrase))), vocab_raw)
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

# ========================== AUDIO & ANKI HELPER ==========================
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

    my_model = genanki.Model(1607392319, 'Cyberpunk Model', fields=[{'name':'Text'},{'name':'Pronunciation'},{'name':'Definition'},{'name':'Examples'},{'name':'Synonyms'},{'name':'Antonyms'},{'name':'Etymology'},{'name':'Audio'}], templates=[{'name': 'Card 1', 'qfmt': front_html, 'afmt': back_html}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    my_deck = genanki.Deck(2059400110, deck_name)
    media_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_map = {}
        if generate_audio:
            progress_bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                unique_vocabs = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
                future_to_vocab = {executor.submit(generate_audio_file, v, temp_dir): v for v in unique_vocabs}
                for i, future in enumerate(concurrent.futures.as_completed(future_to_vocab)):
                    v_key, fname, fpath = future.result()
                    if fname and fpath: media_files.append(fpath); audio_map[v_key] = f"[sound:{fname}]"
                    progress_bar.progress((i + 1) / len(unique_vocabs), text=f"🔊 Generating Audio...")
            progress_bar.empty()

        for nd in notes_data:
            my_note = genanki.Note(model=my_model, fields=[nd['Text'], nd['Pronunciation'], nd['Definition'], nd['Examples'], nd['Synonyms'], nd['Antonyms'], nd['Etymology'], audio_map.get(nd['VocabRaw'], "")])
            my_deck.add_note(my_note)
        
        my_package = genanki.Package(my_deck)
        my_package.media_files = media_files
        buffer = io.BytesIO()
        out_path = os.path.join(temp_dir, 'out.apkg')
        my_package.write_to_file(out_path)
        with open(out_path, "rb") as f: buffer.write(f.read())
        buffer.seek(0)
        return buffer

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    st.subheader("Add new word")
    # Idea 3: Word Extraction Tool Added to modes
    add_mode = st.radio("Mode", ["Single", "Bulk", "Extract"], horizontal=True, label_visibility="collapsed")
    
    if add_mode == "Single":
        with st.form("add_form", clear_on_submit=True):
            v = st.text_input("📝 Vocab", placeholder="e.g. serendipity").lower().strip()
            p_raw = st.text_input("🔤 Phrase", placeholder="I found it by serendipity! (or type '1' to skip)").strip()
            submitted = st.form_submit_button("💾 Save to Cloud", use_container_width=True)

        if submitted and v:
            p = "" if p_raw == "1" else p_raw if p_raw.startswith("*") else p_raw.capitalize()
            # Smart Sync to Session State
            st.session_state.vocab_df = merge_new_words(st.session_state.vocab_df, [{"vocab": v, "phrase": p}])
            save_to_github(st.session_state.vocab_df)
            st.success(f"✅ Saved '{v}'!")
            time.sleep(0.5); st.rerun()

    elif add_mode == "Bulk":
        st.info("Paste words separated by newlines. Optional: add comma for phrase.")
        bulk_text = st.text_area("Paste List", height=150)
        if st.button("💾 Process Bulk List", type="primary"):
            new_rows = []
            for line in [l.strip() for l in bulk_text.split('\n') if l.strip()]:
                parts = line.split(',', 1)
                bv = parts[0].strip().lower()
                bp = parts[1].strip() if len(parts) > 1 else ""
                if bv: new_rows.append({"vocab": bv, "phrase": bp})
            
            if new_rows:
                st.session_state.vocab_df = merge_new_words(st.session_state.vocab_df, new_rows)
                save_to_github(st.session_state.vocab_df)
                st.success(f"✅ Added {len(new_rows)} words!")
                time.sleep(0.5); st.rerun()
                
    else: # Extract Mode (Idea 3)
        st.info("Paste an article or paragraph. We'll extract words so you can quickly select which ones to learn.")
        article_text = st.text_area("Paste Text Here", height=150)
        if article_text:
            sentences = re.split(r'(?<=[.!?])\s+', article_text.replace('\n', ' '))
            raw_words = re.findall(r'\b[A-Za-z\-]{3,}\b', article_text.lower())
            unique_words = sorted(list(set(raw_words)))
            
            selected_words = st.multiselect("Select words to add:", unique_words)
            if st.button("💾 Add Selected Words", type="primary"):
                new_rows = []
                for w in selected_words:
                    # Find the first sentence containing this word to use as the context phrase
                    context = next((s for s in sentences if re.search(r'\b' + re.escape(w) + r'\b', s, re.IGNORECASE)), "")
                    new_rows.append({"vocab": w, "phrase": context})
                
                if new_rows:
                    st.session_state.vocab_df = merge_new_words(st.session_state.vocab_df, new_rows)
                    save_to_github(st.session_state.vocab_df)
                    st.success(f"✅ Extracted and added {len(new_rows)} words!")
                    time.sleep(1); st.rerun()

with tab2:
    if st.session_state.vocab_df.empty: st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        c1, c2 = st.columns([2, 1])
        with c1: search = st.text_input("🔎 Search...", "").lower().strip()
        with c2: filter_new = st.checkbox("Show 'New' only")
        
        display_df = st.session_state.vocab_df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        if filter_new: display_df = display_df[display_df['status'] == 'New']
        
        # Idea 1: Enhanced Data Grid using st.column_config
        edited = st.data_editor(
            display_df, 
            num_rows="dynamic", 
            use_container_width=True, 
            hide_index=True, 
            column_config={
                "vocab": st.column_config.TextColumn("Vocab", disabled=True), # Protect primary key
                "phrase": st.column_config.TextColumn("Phrase Context"),
                "status": st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True),
                "date_added": st.column_config.DateColumn("Date Added", disabled=True)
            }
        )
        
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            if search or filter_new:
                for index, row in edited.iterrows():
                    mask = st.session_state.vocab_df['vocab'] == row['vocab']
                    st.session_state.vocab_df.loc[mask, ['phrase', 'status']] = [row['phrase'], row['status']]
            else: 
                st.session_state.vocab_df = edited
                
            save_to_github(st.session_state.vocab_df.sort_values(by="vocab", ignore_index=True))
            st.toast("✅ Saved!", icon="🎉"); time.sleep(0.5); st.rerun()

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")
    
    with st.expander("👁️ Preview Card Style"):
        st.markdown(f"<style>{CYBERPUNK_CSS}</style>", unsafe_allow_html=True)
        st.markdown("""<div class="card"><div class="vellum-focus-container back"><div class="prompt-text solved-text"><span class="cloze">Serendipity</span> is key.</div></div><div class="vellum-detail-container"><div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">The occurrence of events by chance in a happy or beneficial way.</div></div><div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content"><b>[Noun]</b> /ˌser.ənˈdɪp.ə.ti/</div></div></div></div>""", unsafe_allow_html=True)

    if st.session_state.vocab_df.empty: st.info("Add words first!")
    else:
        col_new, col_all = st.columns(2)
        col_new.metric("New Words", len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']))
        col_all.metric("Total Words", len(st.session_state.vocab_df))
        st.divider()
        
        deck_name_input = st.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
        c1, c2, c3 = st.columns(3)
        with c1: batch_size = st.slider("⚡ Batch Size", 1, 10, 5)
        with c2: include_audio = st.checkbox("🔊 Audio", value=True)
        with c3: process_only_new = st.checkbox("Only process 'New'", value=True)

        if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'] if process_only_new else st.session_state.vocab_df
            if subset.empty: st.warning("⚠️ No words to process!")
            else:
                raw_notes = process_anki_data(subset, batch_size=batch_size)
                
                # Idea 7: Graceful Degradation Handling
                if not raw_notes: 
                    st.error("❌ Generation failed completely. Check API Key or Rate Limits.")
                else:
                    if len(raw_notes) < len(subset):
                        st.warning(f"⚠️ Partial success: {len(raw_notes)} out of {len(subset)} words were generated successfully.")
                        
                    with st.spinner("📦 Packaging Deck..."):
                        apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio)
                    
                    st.success(f"🎉 {len(raw_notes)} cards ready!")
                    st.download_button("📥 Download .apkg", apkg_buffer, f"AnkiDeck_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg", "application/octet-stream", use_container_width=True)
                    
                    if process_only_new:
                        successful_vocabs = [n['VocabRaw'] for n in raw_notes]
                        st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(successful_vocabs), 'status'] = 'Done'
                        save_to_github(st.session_state.vocab_df)
                        st.caption("✅ Successfully generated words marked as 'Done'.")
