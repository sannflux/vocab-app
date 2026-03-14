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
import concurrent.futures

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")

# --- MOBILE KEYBOARD FIX (Global & Instant) ---
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

# --- SECRETS MANAGEMENT ---
try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}")
    st.stop()

# ========================== GITHUB CONNECT (Persistent) ==========================
@st.cache_resource
def get_github_repo():
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except Exception as e:
        st.error(f"❌ GitHub connection failed: {e}")
        return None

# ========================== CLEANING FUNCTIONS ==========================
def cap_first(s: str) -> str:
    s = str(s).strip()
    return s[0].upper() + s[1:] if s else s

def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    if not s: return ""
    return s if s[-1] in ".!?" else (s + ".")

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

def robust_json_parse(text: str):
    try: return json.loads(text)
    except: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return None

def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== AI LOGIC ==========================
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key.strip())
        return genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json", "temperature": 0.1})
    except: return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size, target_lang, model_name, keys_str):
    key_pool = [k.strip() for k in keys_str.split(",") if k.strip()]
    current_key_idx = 0
    model = get_gemini_model(key_pool[current_key_idx], model_name)
    
    all_card_data = []
    batches = [vocab_phrase_list[i:i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    
    with st.status("🤖 Generating Content...", expanded=True) as status:
        progress_bar = st.progress(0)
        for idx, batch in enumerate(batches):
            if idx > 0: time.sleep(12) # RPM Guard
            
            success = False
            while not success and current_key_idx < len(key_pool):
                batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
                prompt = f"""Output ONLY JSON array. Rules: Copy fields exactly. IF 'phrase' starts with '*': Context Hint. ELSE: usage based. FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{target_lang} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "..."}}] INPUT: {json.dumps(batch_dicts)}"""
                try:
                    res = model.generate_content(prompt)
                    parsed = robust_json_parse(res.text)
                    if isinstance(parsed, list):
                        all_card_data.extend(parsed)
                        st.write(f"✅ Batch {idx+1} complete.")
                        success = True
                    else: raise Exception("Retry")
                except:
                    current_key_idx += 1
                    if current_key_idx < len(key_pool):
                        model = get_gemini_model(key_pool[current_key_idx], model_name)
                        time.sleep(2)
                    else: break
            if not success: break
            progress_bar.progress((idx + 1) / len(batches))
        status.update(label="✅ AI Complete", state="complete", expanded=False)
    return all_card_data

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    repo = get_github_repo()
    if not repo: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])
    try:
        content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna(""); df['status'] = df.get('status', 'New')
        return df.sort_values(by="vocab", ignore_index=True)
    except: return pd.DataFrame(columns=['vocab', 'phrase', 'status'])

def save_to_github(dataframe):
    repo = get_github_repo()
    if not repo: return False
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Sync", csv_data, file.sha)
    except:
        repo.create_file("vocabulary.csv", "Init", csv_data)
    load_data.clear(); return True

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()

# ========================== CALLBACKS ==========================
def save_single_callback():
    v = st.session_state.input_vocab.lower().strip()
    p_raw = st.session_state.input_phrase
    if v:
        p = p_raw.strip()
        if p and p not in ["1", "*"]:
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
    else: st.session_state.save_error = "⚠️ Enter vocab."

# ========================== TABS ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_LANG = st.selectbox("🎯 Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"])
    GEMINI_MODEL = st.selectbox("🤖 Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"])
    if "gemini_keys" not in st.session_state: st.session_state.gemini_keys = DEFAULT_GEMINI_KEY
    st.session_state.gemini_keys = st.text_area("Keys (comma separated)", value=st.session_state.gemini_keys, height=100)

tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Anki"])

with tab1:
    if "save_message" in st.session_state: st.success(st.session_state.save_message); del st.session_state.save_message
    if "save_error" in st.session_state: st.error(st.session_state.save_error); del st.session_state.save_error
    
    p_raw = st.text_input("Phrase", key="input_phrase")
    v_sel = ""
    if p_raw and p_raw not in ["1", "*"]:
        words = list(dict.fromkeys([w.lower() for w in re.sub(r'[^\w\s\-\']', '', p_raw).split() if w.strip()]))
        if words:
            try:
                pills = st.pills("Pick", words, selection_mode="multi", label_visibility="collapsed")
                v_sel = " ".join(pills) if pills else ""
            except: pass
    if v_sel and v_sel != st.session_state.get("input_vocab", ""): st.session_state.input_vocab = v_sel
    st.text_input("Vocab", key="input_vocab")
    st.button("💾 Save", type="primary", use_container_width=True, on_click=save_single_callback)

with tab2:
    search = st.text_input("🔎 Search").lower()
    df_disp = st.session_state.vocab_df.copy()
    if search: df_disp = df_disp[df_disp['vocab'].str.contains(search, case=False)]
    edited = st.data_editor(df_disp, num_rows="dynamic", use_container_width=True, hide_index=True)
    if st.button("💾 Sync"): st.session_state.vocab_df = edited; save_to_github(st.session_state.vocab_df); st.rerun()

with tab3:
    st.subheader("Generate Anki")
    batch_size = st.slider("Batch Size", 1, 15, 10)
    if st.button("🚀 Generate", type="primary", use_container_width=True):
        subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New']
        if subset.empty: st.warning("No new words.")
        else:
            # Lazy Load Generation Modules
            from gtts import gTTS
            import genanki
            
            raw_data = generate_anki_card_data_batched(subset[['vocab', 'phrase']].values.tolist(), batch_size, TARGET_LANG, GEMINI_MODEL, st.session_state.gemini_keys)
            if raw_data:
                processed = []
                unique_vocabs = set()
                for c in raw_data:
                    vr = str(c.get("vocab", "")).lower().strip()
                    p = cap_each_sentence(ensure_trailing_dot(clean_grammar(c.get("phrase", ""))))
                    processed.append({
                        "vr": vr, "Text": f"{highlight_vocab(p, vr)}<br><br>{cap_first(vr)}: <b>{{{{c1::{ensure_trailing_dot(c.get('translation', ''))}}}}}</b>",
                        "Pron": f"<b>[{c.get('part_of_speech', '')}]</b> {c.get('pronunciation_ipa', '')}",
                        "Def": ensure_trailing_dot(c.get("definition_english", "")),
                        "Ex": "<ul>" + "".join(f"<li>{e}</li>" for e in c.get("example_sentences", [])) + "</ul>",
                        "Syn": ", ".join(c.get("synonyms_antonyms", {}).get("synonyms", [])),
                        "Etym": c.get("etymology", "")
                    })
                    unique_vocabs.add(vr)
                
                # Anki Packaging
                model = genanki.Model(1607392319, 'CyberModel', fields=[{'name': 'Text'}, {'name': 'Pron'}, {'name': 'Def'}, {'name': 'Ex'}, {'name': 'Syn'}, {'name': 'Ant'}, {'name': 'Etym'}, {'name': 'Audio'}], templates=[{'name': 'C1', 'qfmt': '<div style="border:2px solid #00ff41;padding:20px;color:white;background:#111">{{cloze:Text}}</div>', 'afmt': '{{cloze:Text}}<hr>{{Def}}<br>{{Pron}}<br>{{Ex}}'}], css=".card{font-family:arial;font-size:20px;text-align:left;color:#00ff41;background-color:#111;}")
                deck = genanki.Deck(2059400110, "Vocab Deck")
                media = []
                with tempfile.TemporaryDirectory() as tmp:
                    for v in unique_vocabs:
                        try:
                            f = os.path.join(tmp, f"{v}.mp3")
                            gTTS(text=v, lang='en').save(f)
                            media.append(f)
                        except: pass
                    for p in processed:
                        deck.add_note(genanki.Note(model=model, fields=[p['Text'], p['Pron'], p['Def'], p['Ex'], p['Syn'], "", p['Etym'], f"[sound:{p['vr']}.mp3]" if os.path.exists(os.path.join(tmp, f"{p['vr']}.mp3")) else ""]))
                    
                    pkg = genanki.Package(deck); pkg.media_files = media
                    buf = io.BytesIO(); out = os.path.join(tmp, "deck.apkg"); pkg.write_to_file(out)
                    with open(out, "rb") as f: buf.write(f.read())
                    st.download_button("📥 Download", buf.getvalue(), "vocab.apkg", "application/octet-stream")
                    st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin([n['vr'] for n in processed]), 'status'] = 'Done'
                    save_to_github(st.session_state.vocab_df)
