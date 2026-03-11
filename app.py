import streamlit as st
import pandas as pd
from github import Github, GithubException, Auth
import io
import random
from datetime import date, datetime, timezone, timedelta
import google.generativeai as genai
import json
import re
import time
import os
import tempfile
import hashlib
import concurrent.futures
import altair as alt

# IMPORTS FOR AUDIO & ANKI PACKAGE
try:
    from gtts import gTTS
    import genanki
except ImportError:
    st.error("⚠️ Missing libraries! Please add `gTTS` and `genanki` to your requirements.txt")
    st.stop()

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")

# --- JS FOR KEYBOARD & AUTO-AUDIO ---
st.components.v1.html("""
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            const activeElement = doc.activeElement;
            if (activeElement && activeElement.tagName === 'INPUT') {
                setTimeout(function() { activeElement.blur(); }, 50);
            }
        }
    });
    </script>
""", height=0, width=0)

st.title("📚 My Cloud Vocab")

# --- SECRETS MANAGEMENT ---
try:
    token = st.secrets["GITHUB_TOKEN"]
    repo_name = st.secrets["REPO_NAME"]
    DEFAULT_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

# --- TIMEZONE (WIB UTC+7) ---
WIB = timezone(timedelta(hours=7))
def get_wib_now():
    return datetime.now(WIB).strftime("%d-%m-%Y %H:%M")

# ========================== GITHUB CONNECT ==========================
try:
    auth = Auth.Token(token)
    g = Github(auth=auth)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== LOAD / SAVE DATA ==========================
@st.cache_data(ttl=600)
def load_data_from_github():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        df['phrase'] = df['phrase'].fillna("")
        if 'status' not in df.columns: df['status'] = 'New'
        if 'tags' not in df.columns: df['tags'] = ""
        df['tags'] = df['tags'].fillna("")
        if 'date_added' not in df.columns: df['date_added'] = get_wib_now()
        df['date_added'] = df['date_added'].fillna(get_wib_now()).replace("None", get_wib_now())
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e:
        if e.status == 404: return pd.DataFrame(columns=['vocab', 'phrase', 'tags', 'status', 'date_added'])
        else: st.error(f"❌ GitHub Error {e.status}"); st.stop()
    except Exception as e:
        st.error(f"❌ CSV Error. {e}"); st.stop()

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, f"Vocab App Update - {get_wib_now()}", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data_from_github.clear()
    return True

# --- SETTINGS CLOUD MANAGER ---
def load_settings_from_github():
    try:
        file_content = repo.get_contents("settings.json")
        return json.loads(file_content.decoded_content.decode('utf-8'))
    except Exception: return {}

def save_settings_to_github(settings_dict):
    settings_json = json.dumps(settings_dict, indent=2)
    try:
        file = repo.get_contents("settings.json")
        repo.update_file(file.path, "Update preferences", settings_json, file.sha)
    except GithubException as e:
        if e.status == 404: repo.create_file("settings.json", "Initial settings", settings_json)
    return True

# --- SESSION STATE INIT ---
if "settings_loaded" not in st.session_state:
    cloud_settings = load_settings_from_github()
    st.session_state.auto_sync = cloud_settings.get("auto_sync", False)
    st.session_state.target_lang = cloud_settings.get("target_lang", "Indonesian")
    st.session_state.ai_model = cloud_settings.get("ai_model", "gemini-2.5-flash-lite")
    st.session_state.cefr_level = cloud_settings.get("cefr_level", "B2 (Upper Intermediate)")
    st.session_state.custom_prompt = cloud_settings.get("custom_prompt", "")
    st.session_state.audio_accent = cloud_settings.get("audio_accent", "com")
    st.session_state.audio_speed = cloud_settings.get("audio_speed", False)
    st.session_state.settings_loaded = True

if "gemini_key" not in st.session_state: st.session_state.gemini_key = DEFAULT_GEMINI_KEY
if "unsaved_changes" not in st.session_state: st.session_state.unsaved_changes = False
if "deck_name" not in st.session_state: st.session_state.deck_name = "-English Learning::Vocabulary"
if "deleted_rows_history" not in st.session_state: st.session_state.deleted_rows_history = []
if "phrase_key" not in st.session_state: st.session_state.phrase_key = 0

if "quiz_active_row" not in st.session_state: st.session_state.quiz_active_row = None
if "quiz_options" not in st.session_state: st.session_state.quiz_options = []
if "quiz_answered" not in st.session_state: st.session_state.quiz_answered = False
if "quiz_correct" not in st.session_state: st.session_state.quiz_correct = False

def trigger_sync():
    if st.session_state.auto_sync:
        try:
            save_to_github(st.session_state.vocab_df)
            st.session_state.unsaved_changes = False
            st.toast("☁️ Auto-synced!", icon="✅")
        except Exception: st.session_state.unsaved_changes = True
    else: st.session_state.unsaved_changes = True

if "vocab_df" not in st.session_state:
    st.session_state.vocab_df = load_data_from_github().copy()

# ========================== SIDEBAR & CONFIG ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.auto_sync = st.checkbox("☁️ Auto-Sync", value=st.session_state.auto_sync)
    st.divider()
    lang_opts = ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"]
    TARGET_LANG = st.selectbox("🎯 Definition Language", lang_opts, index=lang_opts.index(st.session_state.target_lang) if st.session_state.target_lang in lang_opts else 0)
    model_opts = ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"]
    GEMINI_MODEL = st.selectbox("🤖 AI Model", model_opts, index=model_opts.index(st.session_state.ai_model) if st.session_state.ai_model in model_opts else 0)
    CEFR_LEVELS = ["A1 (Beginner)", "A2 (Elementary)", "B1 (Intermediate)", "B2 (Upper Intermediate)", "C1 (Advanced)", "C2 (Mastery)"]
    st.session_state.cefr_level = st.selectbox("📈 CEFR Level", CEFR_LEVELS, index=CEFR_LEVELS.index(st.session_state.cefr_level) if st.session_state.cefr_level in CEFR_LEVELS else 3)
    st.session_state.custom_prompt = st.text_area("🧠 AI Rules", value=st.session_state.custom_prompt)
    st.divider()
    accent_map = {"US English": "com", "UK English": "co.uk", "Australian English": "com.au", "Indian English": "co.in"}
    rev_accent = {v: k for k, v in accent_map.items()}
    sel_accent_name = st.selectbox("Accent", list(accent_map.keys()), index=list(accent_map.keys()).index(rev_accent.get(st.session_state.audio_accent, "US English")))
    st.session_state.audio_accent = accent_map[sel_accent_name]
    st.session_state.audio_speed = st.checkbox("Slow Audio", value=st.session_state.audio_speed)
    if st.button("💾 Save as Default", use_container_width=True):
        save_settings_to_github({"auto_sync": st.session_state.auto_sync, "target_lang": TARGET_LANG, "ai_model": GEMINI_MODEL, "cefr_level": st.session_state.cefr_level, "custom_prompt": st.session_state.custom_prompt, "audio_accent": st.session_state.audio_accent, "audio_speed": st.session_state.audio_speed})
        st.success("Saved!")
    st.divider()
    if st.session_state.unsaved_changes:
        if st.button("☁️ Sync to GitHub", type="primary"):
            if save_to_github(st.session_state.vocab_df): st.session_state.unsaved_changes = False; st.rerun()

# ========================== GEMINI ==========================
@st.cache_resource
def get_gemini_model(api_key: str, model_name: str):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json", "temperature": 0.1})
    except Exception: return None

# ========================== CLEANING ==========================
def cap_first(s: str) -> str: return s[0].upper() + s[1:] if s else s
def ensure_trailing_dot(s: str) -> str:
    s = str(s).strip()
    if not s: return s
    if s[-1] == ",": return s[:-1] + "."
    elif s[-1] not in ".!?": return s + "."
    return s

def normalize_spaces(text: str) -> str: return re.sub(r"\s+", " ", str(text)).strip() if text else ""
def highlight_vocab(text: str, vocab: str) -> str:
    pattern = r'\b' + re.escape(vocab) + r'\b'
    return re.sub(pattern, f'<b><u>{vocab}</u></b>', text, flags=re.IGNORECASE)

def generate_blanked_phrase(phrase: str, vocab: str) -> str:
    pattern = r'\b' + re.escape(vocab) + r'\b'
    return re.sub(pattern, "_____", phrase, flags=re.IGNORECASE)

# ========================== SPEECH & AI HELPERS ==========================
def speak_word(text: str):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    rate = "0.6" if st.session_state.audio_speed else "0.95"
    js = f"""<script>var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="en-US";u.rate={rate};window.speechSynthesis.speak(u);</script>"""
    st.components.v1.html(js, height=0)

@st.cache_data(ttl=3600)
def fetch_ai_definition(vocab, phrase, api_key, model_name, target_lang):
    model = get_gemini_model(api_key, model_name)
    if not model: return "AI Error"
    prompt = f'Provide a 1-sentence {target_lang} translation and short English definition for "{vocab}" in context of: "{phrase}". JSON format: {{"translation": "...", "definition": "..."}}'
    try:
        res = model.generate_content(prompt).text
        data = json.loads(res)
        return f"**{target_lang}:** {data.get('translation', '')} \n\n **Def:** {data.get('definition', '')}"
    except Exception: return "Error"

# ========================== BATCH GENERATOR ==========================
def robust_json_parse(text: str):
    try: return json.loads(text)
    except Exception: pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(0))
        except Exception: pass
    return None

def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
    if not model: return []
    all_card_data = []
    progress_bar = st.progress(0)
    total_items = len(vocab_phrase_list)
    
    custom_rule = f"\n5. CUSTOM RULE: {st.session_state.custom_prompt}" if st.session_state.custom_prompt else ""
    cefr_rule = f"\n6. DIFFICULTY: Use {st.session_state.cefr_level} vocabulary only."

    for i in range(0, total_items, batch_size):
        progress_bar.progress(i / total_items, text=f"🤖 Processing {i}/{total_items}...")
        batch = vocab_phrase_list[i:i + batch_size]
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        
        # --- THE CONTEXT-LOCK PROMPT ---
        prompt = f"""You are an expert lexicographer. Output ONLY a JSON array.
RULES:
1. IF 'phrase' is provided: Identify the EXACT Part of Speech and specific contextual meaning of the 'vocab' word in that sentence. Your 'translation' and 'definition_english' MUST reflect that exact usage.
2. IF 'phrase' is empty: Provide the most common high-frequency dictionary definition.
3. IF 'phrase' starts with '*': Treat as a context hint.{custom_rule}{cefr_rule}
FORMAT: [{{"vocab": "...", "phrase": "...", "phrase_translation": "{TARGET_LANG} meaning of the whole sentence", "translation": "{TARGET_LANG} meaning of the vocab word itself", "part_of_speech": "...", "pronunciation_ipa": "/.../", "definition_english": "...", "example_sentences": ["..."], "synonyms_antonyms": {{"synonyms": [], "antonyms": []}}, "etymology": "text"}}]
INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

        for attempt in range(5):
            try:
                response = model.generate_content(prompt)
                parsed = robust_json_parse(response.text)
                if isinstance(parsed, list): all_card_data.extend(parsed); break
            except Exception: time.sleep(2)
        else: break
    progress_bar.empty()
    return all_card_data

def process_anki_data(df_subset, batch_size=6):
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size)
    processed_notes = []
    for card_data in all_card_data:
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        phrase = ensure_trailing_dot(card_data.get("phrase", ""))
        text_field = f"{highlight_vocab(phrase, vocab_raw)}<br><br>{cap_first(vocab_raw)}: <b>{{{{c1::{ensure_trailing_dot(card_data.get('translation', '?'))}}}}}</b>" if phrase else f"{cap_first(vocab_raw)}: <b>{{{{c1::{ensure_trailing_dot(card_data.get('translation', '?'))}}}}}</b>"
        processed_notes.append({"VocabRaw": vocab_raw, "Text": text_field, "PhraseTranslation": ensure_trailing_dot(card_data.get("phrase_translation", "")), "Pronunciation": f"<b>[{str(card_data.get('part_of_speech', '')).title()}]</b> {card_data.get('pronunciation_ipa', '')}", "Definition": ensure_trailing_dot(card_data.get("definition_english", "")), "Examples": "<ul>" + "".join(f"<li><i>{ensure_trailing_dot(e)}</i></li>" for e in card_data.get("example_sentences", [])[:3]) + "</ul>", "Synonyms": ", ".join(card_data.get("synonyms_antonyms", {}).get("synonyms", [])[:5]), "Antonyms": ", ".join(card_data.get("synonyms_antonyms", {}).get("antonyms", [])[:5]), "Etymology": card_data.get("etymology", "")})
    return processed_notes

# ========================== ANKI PACKAGE ==========================
THEMES = {
    "Minimalist Light": """
        .card { font-family: 'Inter', sans-serif; font-size: 18px; line-height: 1.6; color: #333; background-color: #fff; padding: 30px 20px; }
        .vellum-focus-container { background: #f9fafb; padding: 30px 20px; border-radius: 8px; border: 1px solid #e5e7eb; text-align: center; margin-bottom: 20px; }
        .prompt-text { font-size: 1.6em; font-weight: 700; color: #111827; }
        .context-translation { font-style: italic; color: #4b5563; font-size: 0.9em; margin-top: 10px; }
        .cloze { color: #fff; background-color: #3b82f6; border-radius: 4px; padding: 2px 6px; }
        .solved-text .cloze { color: #2563eb; background: none; border-bottom: 2px solid #3b82f6; }
        .vellum-section { margin: 15px 0; padding: 10px 0; border-bottom: 1px solid #f3f4f6; }
        .section-header { font-weight: 600; color: #6b7280; font-size: 0.85em; text-transform: uppercase; }
        .nightMode .card { background-color: #121212 !important; color: #e0e0e0 !important; }
        .nightMode .vellum-focus-container { background: #1e1e1e !important; border-color: #333 !important; }
        .nightMode .prompt-text { color: #fff !important; }
        .nightMode .cloze { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
    """
}

def create_anki_package(notes_data, deck_name, css_theme, generate_audio=True, max_per_deck=0):
    front = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div>{{#PhraseTranslation}}<div class="context-translation">{{PhraseTranslation}}</div>{{/PhraseTranslation}}</div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}</div>{{Audio}}"""
    my_model = genanki.Model(1607392320, 'Vocab Model v2', fields=[{'name': 'Text'}, {'name': 'PhraseTranslation'}, {'name': 'Pronunciation'}, {'name': 'Definition'}, {'name': 'Examples'}, {'name': 'Synonyms'}, {'name': 'Antonyms'}, {'name': 'Etymology'}, {'name': 'Audio'}], templates=[{'name': 'Card 1', 'qfmt': front, 'afmt': back}], css=css_theme, model_type=genanki.Model.CLOZE)
    media = []
    with tempfile.TemporaryDirectory() as tmp:
        audio_map = {}
        if generate_audio:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
                unique = {n['VocabRaw'] for n in notes_data if n['VocabRaw']}
                futures = {ex.submit(generate_audio_file, v, tmp, st.session_state.audio_accent, st.session_state.audio_speed): v for v in unique}
                for f in concurrent.futures.as_completed(futures):
                    v, fname, fpath = f.result()
                    if fname: media.append(fpath); audio_map[v] = f"[sound:{fname}]"
        decks = []
        if max_per_deck > 0 and len(notes_data) > max_per_deck:
            for idx, chunk in enumerate([notes_data[i:i + max_per_deck] for i in range(0, len(notes_data), max_per_deck)]):
                d = genanki.Deck(2059400110 + idx, f"{deck_name}::Part {idx+1}")
                for n in chunk: d.add_note(genanki.Note(model=my_model, fields=[n['Text'], n['PhraseTranslation'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")]))
                decks.append(d)
        else:
            d = genanki.Deck(2059400110, deck_name)
            for n in notes_data: d.add_note(genanki.Note(model=my_model, fields=[n['Text'], n['PhraseTranslation'], n['Pronunciation'], n['Definition'], n['Examples'], n['Synonyms'], n['Antonyms'], n['Etymology'], audio_map.get(n['VocabRaw'], "")]))
            decks.append(d)
        pkg = genanki.Package(decks); pkg.media_files = media
        buf = io.BytesIO(); out = os.path.join(tmp, 'out.apkg'); pkg.write_to_file(out)
        with open(out, "rb") as f: buf.write(f.read())
        buf.seek(0); return buf

def generate_audio_file(v, tmp, accent, slow):
    try:
        f = re.sub(r'[^a-z0-9]', '', v) + ".mp3"
        p = os.path.join(tmp, f)
        gTTS(text=v, lang='en', tld=accent, slow=slow).save(p)
        return v, f, p
    except Exception: return v, None, None

# ========================== UI ==========================
st.divider()
st.header("🌟 Word of the Day")
if not st.session_state.vocab_df.empty:
    random.seed(date.today().isoformat())
    row = st.session_state.vocab_df.sample(1).iloc[0]
    st.subheader(row["vocab"].upper())
    if row["phrase"]: st.caption(row["phrase"])
    c1, c2 = st.columns([1,4])
    if c1.button("🔊 Pronounce"): speak_word(row["vocab"])
    if c2.button("✨ AI Define"):
        with st.spinner("..."): st.info(fetch_ai_definition(row["vocab"], row["phrase"], st.session_state.gemini_key, GEMINI_MODEL, TARGET_LANG))

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["➕ Add", "✏️ Edit", "📇 Anki", "🎮 Study"])

with tab1:
    st.subheader("Add Word")
    mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")
    if mode == "Single":
        dynamic_key = f"p_{st.session_state.phrase_key}"
        p_raw = st.text_input("🔤 Phrase", key=dynamic_key, placeholder="Paste and hit Enter...")
        v = ""
        if p_raw and not p_raw.startswith("*"):
            words = list(dict.fromkeys([w.lower() for w in re.findall(r'[^\W\d_]+(?:[-\'][^\W\d_]+)*', p_raw)]))
            if words:
                v_choices = st.pills("👉 Tap words:", options=words, selection_mode="multi")
                v = " ".join(sorted(v_choices, key=lambda x: words.index(x))) if v_choices else ""
                v_manual = st.text_input("📝 Manual override").lower().strip()
                v = v_manual if v_manual else v
                if v: st.info(f"Target: **{v}**")
        else: v = st.text_input("📝 Vocab").lower().strip()
        t_raw = st.text_input("🏷️ Tags")
        if st.button("💾 Save Word", type="primary", use_container_width=True):
            if v:
                p = "" if p_raw == "1" else p_raw if p_raw.startswith("*") else ensure_trailing_dot(p_raw.capitalize())
                t = ", ".join([x.strip().lower() for x in t_raw.split(",") if x.strip()])
                if v in st.session_state.vocab_df['vocab'].values: st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'] == v, ['phrase', 'status', 'tags']] = [p, 'New', t]
                else: st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, pd.DataFrame([{"vocab": v, "phrase": p, "tags": t, "status": "New", "date_added": get_wib_now()}])], ignore_index=True)
                trigger_sync(); st.session_state.phrase_key += 1; st.success(f"Saved {v}!"); time.sleep(1); st.rerun()
    else:
        bulk_text = st.text_area("Paste List (vocab, phrase)")
        bulk_tags = st.text_input("Apply tags to all")
        if st.button("💾 Process Bulk", type="primary", use_container_width=True):
            new_rows = []
            for line in [l.strip() for l in bulk_text.split('\n') if l.strip()]:
                parts = re.sub(r'^[\d\.\-\*\s]+', '', line).split(',', 1)
                bv = parts[0].strip().lower()
                bp = parts[1].strip() if len(parts) > 1 else ""
                bp = "" if bp == "1" else ensure_trailing_dot(bp.capitalize()) if bp and not bp.startswith("*") else bp
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "tags": bulk_tags, "status": "New", "date_added": get_wib_now()})
            if new_rows: st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, pd.DataFrame(new_rows)]).drop_duplicates(subset=['vocab'], keep='last').reset_index(drop=True); trigger_sync(); st.success("Added!"); time.sleep(1); st.rerun()

with tab2:
    if st.session_state.vocab_df.empty: st.info("Empty")
    else:
        c_sort, c_undo = st.columns(2)
        if c_sort.button("🔤 Sort"): st.session_state.vocab_df = st.session_state.vocab_df.sort_values(by="vocab", ignore_index=True); trigger_sync(); st.rerun()
        if c_undo.button("↩️ Undo", disabled=not st.session_state.deleted_rows_history): st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, st.session_state.deleted_rows_history.pop()]).reset_index(drop=True); trigger_sync(); st.rerun()
        search = st.text_input("🔎 Search/Tags").lower().strip()
        disp = st.session_state.vocab_df.copy()
        if search: disp = disp[disp['vocab'].str.contains(search) | disp['tags'].str.contains(search)]
        disp.insert(0, "🗑️", False)
        edited = st.data_editor(disp, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"🗑️": st.column_config.CheckboxColumn(default=False)})
        if st.button("💾 Confirm Edits", type="primary", use_container_width=True):
            deleted = set(edited[edited["🗑️"] == True]["vocab"])
            if deleted: st.session_state.deleted_rows_history.append(st.session_state.vocab_df[st.session_state.vocab_df['vocab'].isin(deleted)].copy()); st.session_state.vocab_df = st.session_state.vocab_df[~st.session_state.vocab_df['vocab'].isin(deleted)]
            for _, r in edited[edited["🗑️"] == False].iterrows(): st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'] == r['vocab'], ['phrase', 'status', 'tags']] = [r['phrase'], r['status'], r['tags']]
            trigger_sync(); st.success("Updated!"); time.sleep(1); st.rerun()

with tab3:
    st.subheader("Generate Anki")
    if not st.session_state.vocab_df.empty:
        st.session_state.deck_name = st.text_input("Deck Name", value=st.session_state.deck_name)
        c1, c2, c3 = st.columns(3)
        only_new = c1.checkbox("Only New", True); inc_audio = c2.checkbox("Audio", True); splt = c3.number_input("Split >", 0, 500, 50)
        if st.button("🚀 Build Deck", type="primary", use_container_width=True):
            sub = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'] if only_new else st.session_state.vocab_df
            if not sub.empty:
                notes = process_anki_data(sub)
                if notes:
                    buf = create_anki_package(notes, st.session_state.deck_name, THEMES["Minimalist Light"], inc_audio, splt)
                    st.download_button("📥 Download", buf, "deck.apkg", use_container_width=True)
                    if only_new: st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin([n['VocabRaw'] for n in notes]), 'status'] = 'Done'; trigger_sync()

with tab4:
    st.subheader("🎮 Study Room")
    valid = st.session_state.vocab_df[st.session_state.vocab_df['phrase'] != ""]
    if len(valid) < 4: st.info("Need 4 words with phrases.")
    else:
        # --- MULTI-MODE LOGIC ---
        game_mode = st.selectbox("🎮 Quiz Mode", ["Phrase (Fill-in-blank)", "Definition (Meanings)"], index=0)
        
        if st.session_state.quiz_active_row is None or st.button("⏭️ Next Question", use_container_width=True):
            st.session_state.quiz_answered = st.session_state.quiz_correct = False
            row = valid.sample(1).iloc[0]
            # Fetch AI data for the question
            with st.spinner("Preparing..."):
                full_data = process_anki_data(pd.DataFrame([row]), 1)[0]
                st.session_state.quiz_active_row = full_data
                wrong = valid[valid['vocab'] != row['vocab']]['vocab'].sample(min(3, len(valid)-1)).tolist()
                st.session_state.quiz_options = random.sample(wrong + [full_data['VocabRaw']], 4)
            st.rerun()
        
        q = st.session_state.quiz_active_row
        if game_mode == "Phrase (Fill-in-blank)":
            st.markdown(f"### {generate_blanked_phrase(q['Text'].split('<br>')[0], q['VocabRaw'])}")
        else:
            st.info(f"📜 {q['Definition']}")
            
        cols = st.columns(2)
        for i, opt in enumerate(st.session_state.quiz_options):
            b_type = "primary" if st.session_state.quiz_answered and opt == q['VocabRaw'] else "secondary"
            if cols[i%2].button(opt.upper(), key=f"q_{i}", use_container_width=True, type=b_type, disabled=st.session_state.quiz_answered):
                st.session_state.quiz_answered = True
                if opt == q['VocabRaw']:
                    st.session_state.quiz_correct = True
                    speak_word(opt) # --- AUTO AUDIO ON CORRECT ---
                st.rerun()
        if st.session_state.quiz_answered:
            if st.session_state.quiz_correct: st.success("Correct!"); st.balloons()
            else: st.error(f"It was **{q['VocabRaw']}**")
