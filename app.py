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

# ========================== LOAD / SAVE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        
        # MIGRATION: Ensure columns exist
        if 'status' not in df.columns: df['status'] = 'New'
        if 'date_added' not in df.columns: df['date_added'] = datetime.now().strftime("%Y-%m-%d")
            
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404: 
            return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'date_added'])
        else: st.error(f"❌ CRITICAL: GitHub Error {e.status}"); st.stop()
    except Exception as e:
        st.error(f"❌ CRITICAL: CSV Error. {e}"); st.stop()

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

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    TARGET_LANG = st.selectbox("🎯 Definition Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    
    st.divider()
    
    # NEW: Progress Bar
    if not df.empty:
        total = len(df)
        done = len(df[df['status'] == 'Done'])
        percent = done / total if total > 0 else 0
        st.progress(percent, text=f"Progress: {done}/{total} Done")
    
    st.divider()
    
    # Word of the Day
    st.header("🌟 Word of the Day")
    if not df.empty:
        today_str = date.today().isoformat()
        random.seed(today_str)
        try:
            row = df.sample(n=1).iloc[0]
            st.subheader(row["vocab"].upper())
            if row["phrase"]: st.caption(row["phrase"])
            
            # Simple JS Speech
            safe_text = row["vocab"].replace("'", "\\'")
            if st.button("🔊 Pronounce"):
                st.components.v1.html(f"<script>speechSynthesis.speak(new SpeechSynthesisUtterance('{safe_text}'));</script>", height=0)
        except: pass
    else: st.info("No words yet!")
    
    st.divider()
    # Backup
    if not df.empty:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Backup CSV", csv_full, f"vocab_{date.today()}.csv", "text/csv")

# ========================== HELPERS ==========================
def cap_first(s): return str(s).strip()[0].upper() + str(s).strip()[1:] if s else ""
def ensure_dot(s): s = str(s).strip(); return s + "." if s and s[-1] not in ".!?" else s
def clean_txt(t): return re.sub(r"\s+", " ", str(t)).strip() if t else ""

# ========================== AI & ANKI LOGIC ==========================
def generate_anki_batch(vocab_list):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []
    
    batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in vocab_list]
    prompt = f"""Output ONLY a JSON array. 
    1. Define 'vocab' strictly based on 'phrase'. 
    2. If 'phrase' starts with '*', use it as a context hint. 
    3. If 'phrase' is empty, generate a simple sentence.
    OUTPUT: [{{"vocab": "...", "phrase": "...", "translation": "{TARGET_LANG} meaning", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "Plain text."}}]
    INPUT: {json.dumps(batch_dicts)}"""
    
    try:
        resp = model.generate_content(prompt)
        # Robust parsing
        clean_resp = re.search(r'\[.*\]', resp.text, re.DOTALL)
        return json.loads(clean_resp.group(0)) if clean_resp else []
    except Exception as e:
        print(f"AI Error: {e}")
        return []

def process_data(subset, batch_size=5):
    all_data = []
    subset = subset[subset['vocab'].str.len() > 0]
    items = subset[['vocab', 'phrase']].values.tolist()
    
    bar = st.progress(0)
    for i in range(0, len(items), batch_size):
        bar.progress(i/len(items), f"🤖 AI Processing {i}/{len(items)}...")
        batch = items[i:i+batch_size]
        res = generate_anki_batch(batch)
        all_data.extend(res)
        time.sleep(0.5)
    bar.empty()
    
    notes = []
    for d in all_data:
        v = str(d.get('vocab', '')).strip().lower()
        if not v: continue
        
        ph = clean_txt(d.get('phrase', ''))
        ph_fmt = re.sub(r'\b'+re.escape(v)+r'\b', f'<b><u>{v}</u></b>', ph, flags=re.I)
        tr = ensure_dot(clean_txt(d.get('translation', '')))
        
        text_field = f"{ph_fmt}<br><br>{cap_first(v)}: <b>{{{{c1::{tr}}}}}</b>"
        
        notes.append({
            'VocabRaw': v,
            'Text': text_field,
            'Pronunciation': f"<b>[{d.get('part_of_speech','').title()}]</b> {d.get('pronunciation_ipa','')}",
            'Definition': ensure_dot(clean_txt(d.get('definition_english',''))),
            'Examples': "<ul>" + "".join(f"<li>{e}</li>" for e in d.get('example_sentences',[])[:3]) + "</ul>",
            'Synonyms': ", ".join(d.get('synonyms_antonyms',{}).get('synonyms',[])[:5]),
            'Antonyms': ", ".join(d.get('synonyms_antonyms',{}).get('antonyms',[])[:5]),
            'Etymology': clean_txt(d.get('etymology',''))
        })
    return notes

def make_package(notes, name, audio=True):
    # CSS
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

def gen_audio(txt, folder):
    try:
        fn = re.sub(r'\W', '', txt) + ".mp3"
        fp = os.path.join(folder, fn)
        gTTS(txt).save(fp)
        return txt, fp
    except: return txt, None

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit", "📇 Generate"])

with tab1:
    st.subheader("Add Words")
    mode = st.radio("Input Mode", ["Single", "Bulk"], horizontal=True)
    
    if mode == "Single":
        with st.form("add"):
            v = st.text_input("Vocab").lower().strip()
            p = st.text_input("Phrase (Optional)").strip()
            if st.form_submit_button("Save"):
                if v:
                    new = {"vocab": v, "phrase": p, "status": "New", "date_added": datetime.now().strftime("%Y-%m-%d")}
                    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                    if save_to_github(df): st.success("Saved!"); time.sleep(0.5); st.rerun()
    else:
        txt = st.text_area("Bulk (vocab, phrase)", height=150, help="One per line. Comma optional.")
        if st.button("Process Bulk"):
            rows = []
            for l in txt.split('\n'):
                if ',' in l: v, p = l.split(',', 1)
                else: v, p = l, ""
                if v.strip(): rows.append({"vocab": v.strip().lower(), "phrase": p.strip(), "status": "New", "date_added": datetime.now().strftime("%Y-%m-%d")})
            if rows:
                df = pd.concat([df, pd.DataFrame(rows)]).drop_duplicates(subset=['vocab'], keep='last')
                if save_to_github(df): st.success(f"Added {len(rows)} words!"); time.sleep(0.5); st.rerun()

with tab2:
    st.subheader("Database")
    s = st.text_input("Search")
    show = df[df['vocab'].str.contains(s, case=False)] if s else df
    
    # Editable Grid
    edited = st.data_editor(show, num_rows="dynamic", use_container_width=True, 
                            column_config={
                                "status": st.column_config.SelectboxColumn("Status", options=["New", "Done"]),
                                "date_added": st.column_config.TextColumn("Date", disabled=True)
                            }, hide_index=True)
    
    if st.button("Save Changes"):
        # Merge changes back to master DF
        for i, row in edited.iterrows():
            idx = df[df['vocab'] == row['vocab']].index
            if not idx.empty:
                df.loc[idx, ['phrase', 'status']] = [row['phrase'], row['status']]
        save_to_github(df)
        st.toast("Saved changes!")

with tab3:
    st.subheader("Generate Anki")
    c1, c2 = st.columns(2)
    c1.metric("Total", len(df)); c2.metric("New", len(df[df['status']=='New']))
    
    # NEW: Date Filter
    filter_mode = st.selectbox("Filter words by:", ["All 'New' Words", "Added Today", "Added Last 7 Days"])
    
    if st.button("🚀 Generate Deck", type="primary"):
        # Logic for filter
        target_df = df[df['status'] == 'New'].copy()
        
        if filter_mode == "Added Today":
            today = datetime.now().strftime("%Y-%m-%d")
            target_df = target_df[target_df['date_added'] == today]
        elif filter_mode == "Added Last 7 Days":
            cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            target_df = target_df[target_df['date_added'] >= cutoff]
            
        if target_df.empty:
            st.warning("No words match your filter!")
        else:
            raw_notes = process_data(target_df)
            if raw_notes:
                apkg = make_package(raw_notes, "MyVocabDeck")
                st.download_button("Download .apkg", apkg, "deck.apkg")
                
                # Auto-update status
                df.loc[df['vocab'].isin(target_df['vocab']), 'status'] = 'Done'
                save_to_github(df)
                st.success(f"Processed {len(raw_notes)} cards & updated status!")
    
