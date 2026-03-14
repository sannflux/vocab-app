import streamlit as st
import pandas as pd
from github import Github, GithubException
import io
import json
import re
import time
import os
import tempfile
import hashlib
import concurrent.futures
from datetime import date, datetime
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# IMPORTS FOR AUDIO & ANKI PACKAGE
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS`, `genanki`, and `tenacity` to your requirements.txt")
    st.stop()

# ========================== CONSTANTS & CONFIG ==========================
STABLE_MODEL_ID = 1607392319
MAX_RPD = 20
RPM_WAIT = 12.5  # 5 RPM = 12 seconds per request
st.set_page_config(page_title="Cyber Vocab Cloud", layout="centered", page_icon="🔮")

# ========================== MOBILE UI FIX ==========================
st.components.v1.html("""
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
        setTimeout(() => { e.target.blur(); }, 50);
    }
}, true);
</script>
""", height=0)

# ========================== SECRETS & GITHUB ==========================
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    REPO_NAME = st.secrets["REPO_NAME"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check .streamlit/secrets.toml")
    st.stop()

@st.cache_resource
def get_github_repo():
    try:
        g = Github(GITHUB_TOKEN)
        return g.get_repo(REPO_NAME)
    except Exception as e:
        st.error(f"GitHub Error: {e}")
        return None

repo = get_github_repo()

# ========================== PERSISTENT QUOTA TRACKING ==========================
def load_metadata():
    try:
        content = repo.get_contents("_metadata.json")
        data = json.loads(content.decoded_content.decode())
        if data.get("date") != str(date.today()):
            return {"date": str(date.today()), "rpd_count": 0}
        return data
    except:
        return {"date": str(date.today()), "rpd_count": 0}

def save_metadata(count):
    data = {"date": str(date.today()), "rpd_count": count}
    content_str = json.dumps(data)
    try:
        file = repo.get_contents("_metadata.json")
        repo.update_file(file.path, "Update Quota", content_str, file.sha)
    except:
        repo.create_file("_metadata.json", "Init Quota", content_str)

if "metadata" not in st.session_state:
    st.session_state.metadata = load_metadata()

# ========================== GEMINI SETUP ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json", "temperature": 0.1}
        )
    except Exception as e:
        st.error(f"Gemini Init Error: {e}")
        return None

# ========================== CLEANING & UTILS ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    # Basic Grammar Fixes
    rules = [(r"\bfor helps\b", "to help"), (r"\bis use to\b", "is used to"), (r"\bcan able to\b", "can")]
    for pat, repl in rules:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    # Ensure end punctuation
    if text and text[-1] not in ".!?": text += "."
    return cap_first(text)

def generate_stable_id(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (10**10)

def highlight_vocab(text: str, vocab: str) -> str:
    if not text or not vocab: return text
    pattern = r'\b' + re.escape(vocab) + r'\b'
    return re.sub(pattern, f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

# ========================== AI CORE (THROTTLED) ==========================
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=15),
    retry=retry_if_exception_type(Exception)
)
def call_gemini_with_retry(model, prompt):
    return model.generate_content(prompt)

def generate_anki_card_data_batched(vocab_phrase_list, target_lang, model_name, batch_size=6):
    model = get_gemini_model(GEMINI_API_KEY, model_name)
    if not model: return []

    all_card_data = []
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    
    with st.status("🔮 AI Oracle: Processing Batches...", expanded=True) as status:
        progress_bar = st.progress(0)
        
        for idx, batch in enumerate(batches):
            remaining_rpd = MAX_RPD - st.session_state.metadata["rpd_count"]
            if remaining_rpd <= 0:
                st.warning("🛑 Daily API Limit Reached (20/20). Stopping.")
                break
            
            # RPM Throttling (Except first call)
            if idx > 0:
                for t in range(int(RPM_WAIT), 0, -1):
                    progress_bar.progress(idx / len(batches), text=f"⏳ Throttling for Free Tier ({t}s)...")
                    time.sleep(1)

            batch_json = json.dumps([{"vocab": v[0], "phrase": v[1]} for v in batch])
            prompt = f"""Output a JSON array for English learners.
            TARGET LANGUAGE: {target_lang}
            EXAMPLE INPUT: [{{"vocab": "serendipity", "phrase": "It was pure serendipity."}}]
            EXAMPLE OUTPUT: [{{
                "vocab": "serendipity",
                "phrase": "It was pure serendipity.",
                "translation": "Penemuan yang tidak disengaja",
                "part_of_speech": "noun",
                "pronunciation_ipa": "/ˌserənˈdipədē/",
                "definition_english": "The occurrence of events by chance in a happy or beneficial way.",
                "example_sentences": ["Finding this book was serendipity."],
                "synonyms_antonyms": {{"synonyms": ["fluke", "chance"], "antonyms": ["misfortune"]}},
                "etymology": "Coined by Horace Walpole in 1754."
            }}]
            BATCH: {batch_json}"""

            try:
                response = call_gemini_with_retry(model, prompt)
                st.session_state.metadata["rpd_count"] += 1
                save_metadata(st.session_state.metadata["rpd_count"])
                
                parsed = json.loads(response.text)
                if isinstance(parsed, list):
                    all_card_data.extend(parsed)
                    st.write(f"✅ Batch {idx+1} complete.")
            except Exception as e:
                st.error(f"AI Failure on Batch {idx+1}: {e}")
            
            progress_bar.progress((idx + 1) / len(batches))
        status.update(label="✅ Generation Finished!", state="complete")
    
    return all_card_data

# ========================== ANKI GENERATOR ==========================
CYBERPUNK_CSS = """
.card { font-family: 'Segoe UI', Roboto, sans-serif; font-size: 19px; color: #e0e0e0; background-color: #0a0a0c; text-align: left; padding: 20px; }
.vellum-focus-container { border: 2px solid #00f3ff; background: #111; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0, 243, 255, 0.2); text-align: center; }
.prompt-text { font-size: 1.6em; font-weight: 800; color: #fff; text-shadow: 0 0 8px #00f3ff; }
.cloze { color: #00f3ff; font-weight: bold; background: rgba(0, 243, 255, 0.1); padding: 0 4px; border-radius: 4px; }
.section-header { color: #ff00ff; font-size: 0.8em; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; border-bottom: 1px solid #333; margin-top: 15px; }
.content { padding: 5px 0 10px 0; color: #ccc; }
ul { margin: 5px 0; padding-left: 20px; }
"""

def create_anki_package(notes_data, deck_name, include_audio=True):
    model_id = STABLE_MODEL_ID
    deck_id = generate_stable_id(deck_name)
    
    front_html = """<div class="vellum-focus-container"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """
    <div class="vellum-focus-container"><div class="prompt-text">{{cloze:Text}}</div></div>
    <div class="section-header">Meaning</div><div class="content">{{Definition}}</div>
    <div class="section-header">IPA</div><div class="content">{{Pronunciation}}</div>
    {{#Examples}}<div class="section-header">Examples</div><div class="content">{{Examples}}</div>{{/Examples}}
    {{#Etymology}}<div class="section-header">History</div><div class="content">{{Etymology}}</div>{{/Etymology}}
    <div style='display:none'>{{Audio}}</div>{{Audio}}
    """
    
    my_model = genanki.Model(model_id, 'CyberVocabModelV3', 
        fields=[{'name': 'Text'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}],
        templates=[{'name': 'Cloze', 'qfmt': front_html, 'afmt': back_html}], css=CYBERPUNK_CSS, model_type=genanki.Model.CLOZE)
    
    my_deck = genanki.Deck(deck_id, deck_name)
    media_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for card in notes_data:
            vocab = card.get("vocab", "word")
            audio_tag = ""
            if include_audio:
                try:
                    safe_name = hashlib.md5(vocab.encode()).hexdigest() + ".mp3"
                    path = os.path.join(temp_dir, safe_name)
                    tts = gTTS(text=vocab, lang='en')
                    tts.save(path)
                    media_files.append(path)
                    audio_tag = f"[sound:{safe_name}]"
                except: pass

            v_cap = cap_first(vocab)
            raw_phrase = card.get("phrase", "")
            formatted_phrase = highlight_vocab(clean_text(raw_phrase), vocab) if raw_phrase else ""
            
            # FIXED SYNTAX: Avoiding nesting 6 braces in f-string
            translation_val = card.get('translation', '?')
            cloze_markup = "{{c1::" + str(translation_val) + "}}"
            display_text = f"{formatted_phrase}<br><br>{v_cap}: <b>{cloze_markup}</b>"
            
            examples_html = "<ul>" + "".join([f"<li>{clean_text(e)}</li>" for e in card.get("example_sentences", [])]) + "</ul>"
            
            my_deck.add_note(genanki.Note(model=my_model, fields=[
                display_text, 
                f"[{card.get('part_of_speech', '')}] {card.get('pronunciation_ipa', '')}",
                clean_text(card.get('definition_english', '')),
                examples_html,
                ", ".join(card.get("synonyms_antonyms", {}).get("synonyms", [])),
                ", ".join(card.get("synonyms_antonyms", {}).get("antonyms", [])),
                card.get("etymology", ""),
                audio_tag
            ]))

        pkg = genanki.Package(my_deck)
        pkg.media_files = media_files
        buf = io.BytesIO()
        pkg.write_to_file(os.path.join(temp_dir, 'out.apkg'))
        with open(os.path.join(temp_dir, 'out.apkg'), 'rb') as f:
            buf.write(f.read())
        buf.seek(0)
        return buf

# ========================== GITHUB DATA OPS ==========================
@st.cache_data(ttl=300)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode()))
        df['phrase'] = df['phrase'].fillna("")
        df['status'] = df.get('status', 'New')
        return df.sort_values(by="vocab")
    except:
        return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_data(df):
    df = df[df['vocab'].str.strip().astype(bool)].drop_duplicates('vocab')
    csv_str = df.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, f"Update {len(df)} words", csv_str, file.sha)
    except:
        repo.create_file("vocabulary.csv", "Initial vocab", csv_str)
    load_data.clear()

if "vocab_df" not in st.session_state:
    st.session_state.vocab_df = load_data()

# ========================== SIDEBAR UI ==========================
with st.sidebar:
    st.title("⚙️ Control Panel")
    TARGET_LANG = st.selectbox("🎯 Target Language", ["Indonesian", "Spanish", "French", "Japanese", "German"])
    GEMINI_MODEL = st.selectbox("🤖 Brain Model", ["gemini-2.0-flash-exp", "gemini-1.5-flash"], index=0)
    
    st.divider()
    rpd = st.session_state.metadata["rpd_count"]
    st.metric("Daily AI Quota", f"{rpd}/{MAX_RPD}", f"{MAX_RPD - rpd} left")
    st.progress(min(rpd / MAX_RPD, 1.0))
    
    if st.button("🔄 Force Sync Cloud"):
        st.session_state.vocab_df = load_data()
        st.rerun()

# ========================== APP TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add Word", "✏️ Management", "📦 Export Anki"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        phrase_in = st.text_area("1. Context Phrase", placeholder="Paste sentence here...", key="p_in")
    with col2:
        if phrase_in:
            words = [w.strip(",.!?") for w in phrase_in.split()]
            chosen = st.multiselect("2. Extract Vocab", list(dict.fromkeys(words)))
            vocab_in = " ".join(chosen)
        else:
            vocab_in = st.text_input("2. Vocab Word", key="v_in")

    if st.button("💾 Save to Cloud", use_container_width=True, type="primary"):
        if vocab_in:
            v_final = vocab_in.lower().strip()
            new_row = pd.DataFrame([{"vocab": v_final, "phrase": phrase_in.strip(), "status": "New"}])
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, new_row]).drop_duplicates('vocab', keep='last')
            save_data(st.session_state.vocab_df)
            st.success(f"Added '{v_final}'")
            time.sleep(1)
            st.rerun()

with tab2:
    search = st.text_input("🔍 Filter vocabulary...")
    df_disp = st.session_state.vocab_df
    if search:
        df_disp = df_disp[df_disp['vocab'].str.contains(search, case=False)]
    
    edited_df = st.data_editor(
        df_disp, 
        num_rows="dynamic", 
        use_container_width=True,
        hide_index=True,
        column_config={"status": st.column_config.SelectboxColumn("Status", options=["New", "Done"])}
    )
    if st.button("💾 Commit Changes"):
        st.session_state.vocab_df = edited_df
        save_data(edited_df)
        st.toast("Cloud Updated!")

with tab3:
    st.subheader("Generate Anki Package")
    new_words = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']
    
    if new_words.empty:
        st.info("No 'New' words found. Add some in Tab 1!")
    else:
        st.write(f"Found **{len(new_words)}** new words.")
        deck_name_in = st.text_input("Deck Name", "-English Learning::Vocabulary")
        batch_size_in = st.slider("Words per AI call", 1, 10, 5)
        
        if st.button("🚀 Start AI Generation", type="primary"):
            vocab_list = new_words[['vocab', 'phrase']].values.tolist()
            ai_results = generate_anki_card_data_batched(vocab_list, TARGET_LANG, GEMINI_MODEL, batch_size_in)
            
            if ai_results:
                pkg_buf = create_anki_package(ai_results, deck_name_in)
                st.download_button(
                    label="📥 Download .apkg File",
                    data=pkg_buf,
                    file_name=f"Vocab_{date.today().strftime('%Y%m%d')}.apkg",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
                # Mark as done
                processed_vocabs = [c.get('vocab').lower() for c in ai_results]
                st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(processed_vocabs), 'status'] = 'Done'
                save_data(st.session_state.vocab_df)
                st.success("Cards generated! Words marked as 'Done' in cloud.")
