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

# IMPORTS
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

# SECRETS
try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}")
    st.stop()

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = DEFAULT_GEMINI_KEY

# GITHUB
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# GEMINI
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

# ========================== DATA HANDLING ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        
        # Ensure columns exist
        if 'status' not in df.columns: df['status'] = 'New'
        if 'date_added' not in df.columns: 
            df['date_added'] = datetime.now().strftime("%Y-%m-%d")
        
        # FIX: Fill any empty dates with today to prevent errors
        df['date_added'] = df['date_added'].fillna(datetime.now().strftime("%Y-%m-%d"))
            
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404: 
            return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'date_added'])
        else: st.error(f"❌ GitHub Error {e.status}"); st.stop()
    except Exception as e:
        st.error(f"❌ CSV Error: {e}"); st.stop()

def save_to_github(dataframe):
    # Cleaning
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=['vocab'], keep='last')
    
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear()
    return True

df = load_data().copy()

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_LANG = st.selectbox("🎯 Definition Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    
    st.divider()
    if not df.empty:
        total = len(df)
        done = len(df[df['status'] == 'Done'])
        st.progress(done / total if total > 0 else 0, text=f"Progress: {done}/{total} Done")
    
    st.divider()
    if not df.empty and st.button("💾 Backup CSV"):
        st.download_button("Download", df.to_csv(index=False).encode('utf-8'), f"vocab_{date.today()}.csv", "text/csv")

# ========================== AI LOGIC ==========================
def generate_anki_batch(vocab_list):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []
    
    batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in vocab_list]
    prompt = f"""Output ONLY a JSON array. 
    RULES:
    1. Define 'vocab' based on 'phrase'.
    2. If 'phrase' starts with '*', use it as a context hint.
    3. If 'phrase' is empty, generate a simple sentence.
    OUTPUT FORMAT: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "Plain text."}}]
    INPUT: {json.dumps(batch_dicts)}"""
    
    for _ in range(3):
        try:
            resp = model.generate_content(prompt)
            clean = re.search(r'\[.*\]', resp.text, re.DOTALL)
            if clean: return json.loads(clean.group(0))
        except: time.sleep(1)
    return []

def process_data(subset, batch_size=5):
    all_data = []
    items = subset[['vocab', 'phrase']].values.tolist()
    
    bar = st.progress(0)
    for i in range(0, len(items), batch_size):
        bar.progress(i/len(items), f"🤖 AI Processing {i}/{len(items)}...")
        batch = items[i:i+batch_size]
        res = generate_anki_batch(batch)
        all_data.extend(res)
    bar.empty()
    
    notes = []
    for d in all_data:
        v = str(d.get('vocab', '')).strip().lower()
        if not v: continue
        
        ph = str(d.get('phrase', ''))
        ph_fmt = re.sub(r'\b'+re.escape(v)+r'\b', f'<b><u>{v}</u></b>', ph, flags=re.I)
        
        notes.append({
            'VocabRaw': v,
            'Text': f"{ph_fmt}<br><br>{v.capitalize()}: <b>{{{{c1::{d.get('translation','')}}}}}</b>",
            'Pronunciation': f"<b>[{d.get('part_of_speech','').title()}]</b> {d.get('pronunciation_ipa','')}",
            'Definition': d.get('definition_english',''),
            'Examples': "<ul>" + "".join(f"<li>{e}</li>" for e in d.get('example_sentences',[])[:3]) + "</ul>",
            'Synonyms': ", ".join(d.get('synonyms_antonyms',{}).get('synonyms',[])[:5]),
            'Antonyms': ", ".join(d.get('synonyms_antonyms',{}).get('antonyms',[])[:5]),
            'Etymology': d.get('etymology','')
        })
    return notes

# ========================== ANKI PACKAGE ==========================
def gen_audio(txt, folder):
    try:
        fn = re.sub(r'\W', '', txt) + ".mp3"
        fp = os.path.join(folder, fn)
        gTTS(txt).save(fp)
        return txt, fp
    except: return txt, None

def make_package(notes, name, audio=True):
    css = ".card{font-family:'Roboto Mono',monospace;background:#111;color:#0f0;padding:20px}.cloze{color:#111;background:#0f0}.nightMode .cloze{color:#000;background:#0af}"
    m = genanki.Model(12345, 'Cyberpunk', 
        fields=[{'name':x} for x in ['Text','Pronunciation','Definition','Examples','Synonyms','Antonyms','Etymology','Audio']],
        templates=[{'name':'C1', 'qfmt':'{{cloze:Text}}', 'afmt':'{{cloze:Text}}<br><br>{{Definition}}<br>{{Audio}}'}],
        css=css, model_type=genanki.Model.CLOZE)
    
    d = genanki.Deck(54321, name)
    files = []
    
    with tempfile.TemporaryDirectory() as tmp:
        amap = {}
        if audio:
            with concurrent.futures.ThreadPoolExecutor(5) as exe:
                futs = {exe.submit(gen_audio, n['VocabRaw'], tmp): n for n in notes}
                for f in concurrent.futures.as_completed(futs):
                    k, p = f.result()
                    if p: files.append(p); amap[k] = f"[sound:{os.path.basename(p)}]"

        for n in notes:
            d.add_note(genanki.Note(model=m, fields=[n['Text'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], amap.get(n['VocabRaw'],'')]))
        
        pkg = genanki.Package(d); pkg.media_files = files
        out = io.BytesIO()
        pkg.write_to_file(os.path.join(tmp, 'deck.apkg'))
        with open(os.path.join(tmp, 'deck.apkg'), 'rb') as f: out.write(f.read())
        out.seek(0)
        return out

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Generate"])

with tab1:
    st.subheader("Add Words")
    mode = st.radio("Input Mode", ["Single", "Bulk"], horizontal=True)
    today_str = datetime.now().strftime("%Y-%m-%d")

    if mode == "Single":
        with st.form("add"):
            v = st.text_input("Vocab").lower().strip()
            p = st.text_input("Phrase (Optional)").strip()
            if st.form_submit_button("Save"):
                if v:
                    new = {"vocab": v, "phrase": p, "status": "New", "date_added": today_str}
                    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                    save_to_github(df); st.success("Saved!"); time.sleep(0.5); st.rerun()
    else:
        txt = st.text_area("Bulk (vocab, phrase)", height=150)
        if st.button("Process Bulk"):
            rows = [{"vocab": l.split(',',1)[0].strip().lower(), "phrase": l.split(',',1)[1].strip() if ',' in l else "", "status": "New", "date_added": today_str} for l in txt.split('\n') if l.strip()]
            if rows:
                df = pd.concat([df, pd.DataFrame(rows)]).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(df); st.success(f"Added {len(rows)} words!"); time.sleep(0.5); st.rerun()

with tab2:
    st.subheader("Database")
    s = st.text_input("Search")
    show = df[df['vocab'].str.contains(s, case=False)] if s else df
    edited = st.data_editor(show, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"status": st.column_config.SelectboxColumn("Status", options=["New", "Done"])})
    if st.button("Save Changes"):
        for i, row in edited.iterrows():
            df.loc[df['vocab'] == row['vocab'], ['phrase', 'status']] = [row['phrase'], row['status']]
        save_to_github(df); st.toast("Saved!")

with tab3:
    st.subheader("Generate Anki")
    
    # 1. Filter Logic
    filter_option = st.selectbox("📅 Date Filter", ["All Words", "Added Today", "Added Last 7 Days"])
    status_option = st.radio("Status Filter", ["Only 'New'", "All Statuses (Redo)"], horizontal=True)
    
    # Pre-calculate counts for display
    # Convert 'date_added' to datetime objects safely for filtering
    temp_df = df.copy()
    temp_df['dt_obj'] = pd.to_datetime(temp_df['date_added'], errors='coerce').dt.date
    
    today_date = date.today()
    
    if filter_option == "Added Today":
        temp_df = temp_df[temp_df['dt_obj'] == today_date]
    elif filter_option == "Added Last 7 Days":
        cutoff = today_date - timedelta(days=7)
        temp_df = temp_df[temp_df['dt_obj'] >= cutoff]
        
    if status_option == "Only 'New'":
        temp_df = temp_df[temp_df['status'] == 'New']
        
    st.info(f"⚡ {len(temp_df)} words match your filters.")

    # 2. Settings
    c1, c2 = st.columns(2)
    with c1: batch = st.slider("Batch Size", 1, 10, 5)
    with c2: audio = st.checkbox("Audio", True)
    
    # 3. Action
    if st.button("🚀 Generate Deck", type="primary", disabled=(len(temp_df)==0)):
        raw_notes = process_data(temp_df, batch)
        if raw_notes:
            apkg = make_package(raw_notes, "MyVocabDeck", audio)
            st.download_button("Download .apkg", apkg, f"deck_{datetime.now().strftime('%H%M')}.apkg")
            
            # Update Status
            if status_option == "Only 'New'":
                df.loc[df['vocab'].isin(temp_df['vocab']), 'status'] = 'Done'
                save_to_github(df)
                st.success("✅ Database updated: Words marked as 'Done'")
    
