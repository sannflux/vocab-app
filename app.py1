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
import math

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

# --- CSS VARIABLES ---
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

# ========================== PERSISTENT API QUOTA TRACKING (A1+A4 enhanced) ==========================
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
    data = json.dumps({"date": str(date.today()), "rpd_count": count})
    try:
        file = repo.get_contents("usage.json")
        repo.update_file(file.path, "Update API usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage.json", "Init API usage", data)

def load_minute_usage():
    try:
        file = repo.get_contents("usage_minute.json")
        data = json.loads(file.decoded_content.decode('utf-8'))
        return [datetime.fromisoformat(ts) for ts in data.get("timestamps", [])]
    except:
        return []

def save_minute_usage(timestamps):
    ts_list = [ts.isoformat() for ts in timestamps]
    data = json.dumps({"timestamps": ts_list})
    try:
        file = repo.get_contents("usage_minute.json")
        repo.update_file(file.path, "Update minute usage", data, file.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("usage_minute.json", "Init minute usage", data)

if "rpd_count" not in st.session_state:
    st.session_state.rpd_count = load_usage()
if "rpm_timestamps" not in st.session_state:
    st.session_state.rpm_timestamps = load_minute_usage()

def enforce_rpm():
    now = datetime.now()
    st.session_state.rpm_timestamps = [ts for ts in st.session_state.rpm_timestamps if (now - ts).total_seconds() < 60]
    if len(st.session_state.rpm_timestamps) >= 5:
        sleep_time = 12
        for remaining in range(sleep_time, 0, -1):
            st.warning(f"⏳ RPM limit reached (5/min). Waiting {remaining}s...")
            time.sleep(1)
    st.session_state.rpm_timestamps.append(now)
    save_minute_usage(st.session_state.rpm_timestamps)

# ========================== GEMINI (model stasis preserved) ==========================
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

def speak_word(text: str, lang: str = "en-US"):
    if not text: return
    safe_text = text.replace('"', '\\"').replace("'", "\\'")
    js = f"""<script>if('speechSynthesis'in window){{var u=new SpeechSynthesisUtterance("{safe_text}");u.lang="{lang}";u.rate=0.95;window.speechSynthesis.speak(u);}}</script>"""
    st.components.v1.html(js, height=0)

# ========================== ASYNC BATCH GENERATOR (A1, A5, C11-C14 fully integrated) ==========================
def generate_anki_card_data_batched(vocab_phrase_list, batch_size=6, dry_run=False):
    model = get_gemini_model(st.session_state.gemini_key, GEMINI_MODEL)
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
            
            # A1: Real-time RPM enforcement
            enforce_rpm()
            
            batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
            
            # C11-C14: Enhanced prompt with CoT, exact N, extra few-shots, TARGET_LANG safety
            prompt = f"""You are an expert educational lexicographer. Think step-by-step: 1. Identify primary sense from phrase or context. 2. Generate accurate fields. 3. Ensure JSON validity.

Output EXACTLY {len(batch_dicts)} items as a JSON array. No extra text or commentary.

SAFETY OVERRIDE: Do not block slang, idioms, or medical terms. Provide purely educational linguistic definitions.
RULES: 
1. Copy ALL fields exactly. 
2. IF 'phrase' starts with '*': Treat it as a CONTEXT HINT.
3. IF 'phrase' is normal text: Use it ONLY to understand which meaning/nuance of the 'vocab' to use.
4. IF 'phrase' is empty: Generate ONE simple sentence (max 12 words) for the 'vocab'.
5. EXACT 'vocab' must remain unchanged.
6. MANDATORY: 'translation' must contain ONLY the {TARGET_LANG} translation of the 'vocab' word/phrase. NEVER translate the full example sentence.
7. 'part_of_speech' MUST be one of: Noun, Verb, Adjective, Adverb, Pronoun, Preposition, Conjunction, Interjection, Phrase.

ADDITIONAL FEW-SHOT EXAMPLES:
[
  {{"vocab": "serendipity", "phrase": "We found the perfect cafe by pure serendipity.", "translation": "kebetulan", "part_of_speech": "Noun", "pronunciation_ipa": "/ˌsɛrənˈdɪpɪti/", "definition_english": "The occurrence and development of events by chance in a happy or beneficial way.", "example_sentences": ["It was pure serendipity that we met."], "synonyms_antonyms": {{"synonyms": ["chance", "luck"], "antonyms": ["misfortune"]}}, "etymology": "Coined by Horace Walpole in 1754."}},
  {{"vocab": "ephemeral", "phrase": "", "translation": "sementara", "part_of_speech": "Adjective", "pronunciation_ipa": "/ɪˈfɛmərəl/", "definition_english": "Lasting for a very short time.", "example_sentences": ["The ephemeral beauty of the sunset."], "synonyms_antonyms": {{"synonyms": ["transient"], "antonyms": ["permanent"]}}, "etymology": "From Greek ephēmeros."}},
  {{"vocab": "run", "phrase": "*He decided to run for office", "translation": "mencalonkan diri", "part_of_speech": "Verb", "pronunciation_ipa": "/rʌn/", "definition_english": "To compete in an election.", "example_sentences": ["She will run for president."], "synonyms_antonyms": {{"synonyms": ["campaign"], "antonyms": ["withdraw"]}}, "etymology": "Old English rinnan."}},
  {{"vocab": "placebo", "phrase": "The placebo effect was strong.", "translation": "plasebo", "part_of_speech": "Noun", "pronunciation_ipa": "/pləˈsiːboʊ/", "definition_english": "A substance with no therapeutic effect used as a control.", "example_sentences": ["Patients reported improvement due to the placebo effect."], "synonyms_antonyms": {{"synonyms": ["dummy"], "antonyms": []}}, "etymology": "Latin 'I shall please'."}}
]

BATCH INPUT: {json.dumps(batch_dicts, ensure_ascii=False)}"""

            vocab_words = [v[0] for v in batch]
            success = False
            
            if dry_run:
                # A5: Dry-run simulation (no quota)
                st.info(f"🔬 Dry-run simulation for: {', '.join(vocab_words)}")
                mock_data = [{"vocab": v[0], "phrase": v[1], "translation": "mock-" + v[0], "part_of_speech": "Noun", "pronunciation_ipa": "/mock/", "definition_english": "Simulated definition.", "example_sentences": ["Mock example."], "synonyms_antonyms": {"synonyms": ["mock"], "antonyms": []}, "etymology": "Simulated."} for v in batch]
                all_card_data.extend(mock_data)
                success = True
            else:
                for attempt in range(3):
                    try:
                        response = model.generate_content(prompt)
                        st.session_state.rpd_count += 1
                        save_usage(st.session_state.rpd_count)
                        
                        parsed = robust_json_parse(response.text)
                        if isinstance(parsed, list) and len(parsed) == len(batch_dicts):
                            all_card_data.extend(parsed)
                            st.markdown(f"✅ **Processed**: `{', '.join(vocab_words)}`")
                            success = True
                            break
                    except Exception as e:
                        if "429" in str(e): 
                            backoff = 20 + (2 ** attempt) + random.uniform(0, 1)
                            st.warning(f"⚠️ 429 Rate Limit. Backing off for {backoff:.1f}s...")
                            time.sleep(backoff)
                        else:
                            time.sleep(2)
            
            if not success and not dry_run:
                st.error(f"❌ **Failed**: `{', '.join(vocab_words)}` (Skipping to preserve quota)")
            
            progress_bar.progress((idx + 1) / len(batches))

        status_log.update(label=f"✅ AI Generation Complete! ({len(all_card_data)} items)", state="complete", expanded=False)
    
    return all_card_data

# ========================== PROCESS ANKI DATA (D15 caching + B8 validation) ==========================
def process_anki_data(df_subset, batch_size=6, dry_run=False):
    # D15: Cache check
    cache_key = hashlib.sha256(str(df_subset.to_dict()).encode()).hexdigest()
    if "processed_cache" in st.session_state and st.session_state.processed_cache.get("key") == cache_key and (datetime.now() - st.session_state.processed_cache.get("time", datetime.min)).total_seconds() < 300:
        st.info("♻️ Using cached processed notes (no re-generation)")
        return st.session_state.processed_cache["notes"]
    
    df_subset = df_subset[df_subset['vocab'].astype(str).str.strip().str.len() > 0].copy()
    vocab_phrase_list = df_subset[['vocab', 'phrase']].values.tolist()
    all_card_data = generate_anki_card_data_batched(vocab_phrase_list, batch_size=batch_size, dry_run=dry_run)
    processed_notes = []

    for card_data in all_card_data:
        # B8: Runtime validation
        required = ["vocab", "translation", "part_of_speech"]
        if not all(k in card_data and card_data[k] for k in required):
            st.error(f"⚠️ Missing required fields for {card_data.get('vocab', 'unknown')} – skipping")
            continue
        
        vocab_raw = str(card_data.get("vocab", "")).strip().lower()
        if not vocab_raw: continue
        vocab_cap = cap_first(vocab_raw)
        phrase = normalize_spaces(card_data.get("phrase", ""))
        phrase = clean_grammar(phrase)
        phrase = cap_each_sentence(phrase)
        phrase = ensure_trailing_dot(phrase)
        phrase = fix_vocab_casing(phrase, vocab_raw)
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
        
        tags = []
        
        processed_notes.append({
            "VocabRaw": vocab_raw, 
            "Text": text_field, 
            "Pronunciation": pronunciation_field, 
            "Definition": eng_def, 
            "Examples": examples_field, 
            "Synonyms": synonyms_field, 
            "Antonyms": antonyms_field, 
            "Etymology": etymology,
            "Tags": tags
        })
    
    # Cache result
    st.session_state.processed_cache = {"key": cache_key, "notes": processed_notes, "time": datetime.now()}
    return processed_notes

# ========================== AUDIO HELPER ==========================
def generate_audio_file(vocab, temp_dir):
    try:
        clean_vocab = re.sub(r'[^a-zA-Z0-9\s\-\']', '', vocab).strip()
        clean_filename = re.sub(r'[^a-zA-Z0-9]', '', clean_vocab) + ".mp3"
        file_path = os.path.join(temp_dir, clean_filename)
        if clean_vocab:
            tts = gTTS(text=clean_vocab, lang='en', slow=False)
            tts.save(file_path)
            return vocab, clean_filename, file_path
    except Exception as e: print(f"Audio error for {vocab}: {e}")
    return vocab, None, None

# ========================== CSS / PREVIEW (B7 enhanced) ==========================
CYBERPUNK_CSS = f"""
.card {{ font-family: 'Roboto Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; color: {THEME_COLOR}; background-color: {BG_COLOR}; background-image: repeating-linear-gradient(0deg, {BG_STRIPE}, {BG_STRIPE} 1px, {BG_COLOR} 1px, {BG_COLOR} 20px); padding: 30px 20px; text-align: left; }}
.vellum-focus-container {{ background: #0d0d0d; padding: 30px 20px; margin: 0 auto 40px; border: 2px solid {THEME_COLOR}; box-shadow: 0 0 5px {THEME_COLOR}, 0 0 15px {THEME_GLOW}; text-align: center; }}
.prompt-text {{ font-family: 'Electrolize', sans-serif; font-size: 1.8em; font-weight: 900; color: #ffffff; text-shadow: 1px 1px 0 #ff00ff, -1px -1px 0 #00ffff; }}
.cloze {{ color: {BG_COLOR}; background-color: {THEME_COLOR}; padding: 2px 4px; }}
.solved-text .cloze {{ color: #ff00ff; background: none; border-bottom: 3px double #00ffff; text-shadow: 0 0 5px #ff00ff; }}
.vellum-section {{ margin: 15px 0; padding: 10px 0; border-bottom: 1px dashed {THEME_COLOR}; }}
.section-header {{ font-weight: 600; color: #00ffff; border-left: 3px solid {THEME_COLOR}; padding-left: 10px; }}
.content {{ color: {TEXT_COLOR}; padding-left: 13px; }}
/* B7: Anki mobile @media */
@media (max-width: 480px) {{ .card {{ font-size: 16px; padding: 15px; }} .vellum-focus-container {{ padding: 15px; }} }}
"""

# ========================== GENANKI LOGIC (B6, B9, B10 fully integrated) ==========================
def create_anki_package(notes_data, deck_name, generate_audio=True, deck_id=2059400110, include_antonyms=True):
    front_html = """<div class="vellum-focus-container front"><div class="prompt-text">{{cloze:Text}}</div></div>"""
    back_html = """<div class="vellum-focus-container back"><div class="prompt-text solved-text">{{cloze:Text}}</div></div><div class="vellum-detail-container">{{#Definition}}<div class="vellum-section"><div class="section-header">📜 DEFINITION</div><div class="content">{{Definition}}</div></div>{{/Definition}}{{#Pronunciation}}<div class="vellum-section"><div class="section-header">🗣️ PRONUNCIATION</div><div class="content">{{Pronunciation}}</div></div>{{/Pronunciation}}{{#Examples}}<div class="vellum-section"><div class="section-header">🖋️ EXAMPLES</div><div class="content">{{Examples}}</div></div>{{/Examples}}{{#Synonyms}}<div class="vellum-section"><div class="section-header">➕ SYNONYMS</div><div class="content">{{Synonyms}}</div></div>{{/Synonyms}}"""
    if include_antonyms:
        back_html += """{{#Antonyms}}<div class="vellum-section"><div class="section-header">➖ ANTONYMS</div><div class="content">{{Antonyms}}</div></div>{{/Antonyms}}"""
    back_html += """{{#Etymology}}<div class="vellum-section"><div class="section-header">🏛️ ETYMOLOGY</div><div class="content">{{Etymology}}</div></div>{{/Etymology}}<div style='display:none'>{{Audio}}</div></div>{{Audio}}"""
    
    # B6: Session-state model ID
    model_id = st.session_state.get("model_id", 1607392319)
    my_model = genanki.Model(
        model_id, 
        'Cyberpunk Vocab Model', 
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
            # B9: Deterministic GUID with deck_name
            guid_input = note_data['VocabRaw'] + deck_name
            vocab_hash = str(int(hashlib.sha256(guid_input.encode('utf-8')).hexdigest(), 16) % (10**10))
            my_deck.add_note(genanki.Note(
                model=my_model, 
                fields=[note_data['Text'], note_data['Pronunciation'], note_data['Definition'], note_data['Examples'], note_data['Synonyms'], note_data['Antonyms'], note_data['Etymology'], audio_map.get(note_data['VocabRaw'], "")],
                tags=note_data['Tags'],
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

# ========================== LOAD / SAVE WITH SESSION STATE ==========================
@st.cache_data(ttl=600)
def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')), dtype=str)
        df['phrase'] = df['phrase'].fillna(""); 
        df['status'] = df.get('status', 'New')
        df['tags'] = df.get('tags', '') 
        return df.sort_values(by="vocab", ignore_index=True)
    except GithubException as e: 
        return pd.DataFrame(columns=['vocab', 'phrase', 'status', 'tags']) if e.status == 404 else st.stop()
    except: st.stop()

def save_to_github(dataframe):
    dataframe = dataframe[dataframe['vocab'].astype(str).str.strip().str.len() > 0].drop_duplicates(subset=['vocab'], keep='last')
    csv_data = dataframe.to_csv(index=False)
    try:
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "Updated vocab", csv_data, file.sha)
    except GithubException as e:
        if e.status == 404: repo.create_file("vocabulary.csv", "Initial commit", csv_data)
    load_data.clear()
    return True

if "vocab_df" not in st.session_state: st.session_state.vocab_df = load_data().copy()
if "deck_id" not in st.session_state: st.session_state.deck_id = 2059400110
if "bulk_preview_df" not in st.session_state: st.session_state.bulk_preview_df = None
if "apkg_buffer" not in st.session_state: st.session_state.apkg_buffer = None
if "processed_vocabs" not in st.session_state: st.session_state.processed_vocabs = []
if "model_id" not in st.session_state: st.session_state.model_id = 1607392319
if "include_antonyms" not in st.session_state: st.session_state.include_antonyms = True
if "dry_run" not in st.session_state: st.session_state.dry_run = False
if "processed_cache" not in st.session_state: st.session_state.processed_cache = {}

def mark_as_done_callback():
    """Triggered ONLY when the user physically clicks 'Download .apkg'"""
    if "processed_vocabs" in st.session_state and st.session_state.processed_vocabs:
        st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin(st.session_state.processed_vocabs), 'status'] = 'Done'
        save_to_github(st.session_state.vocab_df)
    st.session_state.apkg_buffer = None
    st.session_state.processed_vocabs = []

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
            st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, pd.DataFrame([{"vocab": v, "phrase": p, "status": "New", "tags": ""}])], ignore_index=True)
        save_to_github(st.session_state.vocab_df)
        st.session_state.input_phrase = ""; st.session_state.input_vocab = ""
        st.toast(f"✅ Saved '{v}'!", icon="🚀")
    else: 
        st.error("⚠️ Enter a vocabulary word.")

if "input_phrase" not in st.session_state: st.session_state.input_phrase = ""
if "input_vocab" not in st.session_state: st.session_state.input_vocab = ""

# ========================== SIDEBAR DASHBOARD (A2 quota visuals) ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    total_words = len(st.session_state.vocab_df)
    new_words = len(st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'])
    
    col1, col2 = st.columns(2)
    col1.metric("📖 Total", total_words)
    col2.metric("✨ New", new_words)
    st.metric("🤖 Daily AI Usage", f"{st.session_state.rpd_count}/20 Requests")
    
    # A2: Visual quota dashboard
    rpm_current = len([ts for ts in st.session_state.rpm_timestamps if (datetime.now() - ts).total_seconds() < 60])
    st.progress(rpm_current / 5, text=f"RPM Live: {rpm_current}/5 (last 60s)")
    rpd_progress = st.session_state.rpd_count / 20
    st.progress(rpd_progress, text=f"RPD: {st.session_state.rpd_count}/20")
    
    st.divider()
    TARGET_LANG = st.selectbox("🎯 Definition Language", ["Indonesian", "Spanish", "French", "German", "Japanese", "English (Simple)"], index=0)
    GEMINI_MODEL = st.selectbox("🤖 AI Model", ["gemini-2.5-flash-lite", "gemini-2.0-flash-exp"], index=0)
    st.divider()
    # B6: Model ID control
    if st.button("🔄 Regenerate Note Type Model ID"):
        st.session_state.model_id = random.randrange(1 << 30, 1 << 31)
        st.success(f"New Model ID: {st.session_state.model_id}")
    st.caption(f"Current Model ID: {st.session_state.model_id}")
    if not st.session_state.vocab_df.empty:
        st.download_button("💾 Backup Database (CSV)", st.session_state.vocab_df.to_csv(index=False).encode('utf-8'), f"vocab_backup_{date.today()}.csv", "text/csv")

# ========================== TABS ==========================
tab1, tab2, tab3 = st.tabs(["➕ Add", "✏️ Edit / Review", "📇 Generate Anki"])

with tab1:
    st.subheader("Add new word")
    add_mode = st.radio("Mode", ["Single", "Bulk"], horizontal=True, label_visibility="collapsed")

    if add_mode == "Single":
        p_raw = st.text_input("🔤 Phrase", placeholder="Paste your sentence here...", key="input_phrase")
        v_selected = ""
        if p_raw and p_raw not in ["1", "*"]:
            clean_text = re.sub(r'[^\w\s\-\']', '', p_raw)
            unique_words = list(dict.fromkeys([w.lower() for w in clean_text.split() if w.strip()]))
            if unique_words:
                st.caption("Click words below to extract them as vocabulary:")
                try:
                    selected_pills = st.pills("Select Vocab", unique_words, selection_mode="multi", label_visibility="collapsed")
                    v_selected = " ".join(selected_pills) if selected_pills else ""
                except:
                    selected_words = []
                    cols = st.columns(6)
                    for i, w in enumerate(unique_words):
                        if cols[i % 6].checkbox(w, key=f"chk_{w}"): selected_words.append(w)
                    v_selected = " ".join(selected_words)
        
        if v_selected and v_selected != st.session_state.input_vocab:
            st.session_state.input_vocab = v_selected
            
        st.text_input("📝 Vocab", placeholder="e.g. serendipity", key="input_vocab")
        
        v_check = st.session_state.input_vocab.lower().strip()
        if v_check and not st.session_state.vocab_df.empty and (st.session_state.vocab_df['vocab'] == v_check).any():
            st.warning(f"⚠️ '{v_check}' is already in your database. Saving will overwrite its phrase and reset to 'New'.")
            
        st.button("💾 Save to Cloud", type="primary", use_container_width=True, on_click=save_single_word_callback)

    else: 
        bulk_text = st.text_area("Paste List (word, phrase)", height=150, key="bulk_input")
        if st.button("Preview Bulk Import"):
            lines = [l.strip() for l in bulk_text.split('\n') if l.strip()]
            new_rows = []
            for line in lines:
                parts = line.split(',', 1); bv = parts[0].strip().lower(); bp = parts[1].strip() if len(parts) > 1 else ""
                if bp and bp != "1" and not bp.startswith("*"):
                    if bp.endswith(","): bp = bp[:-1] + "."
                    elif not bp.endswith((".", "!", "?")): bp += "."
                    bp = bp.capitalize()
                if bv: new_rows.append({"vocab": bv, "phrase": bp, "status": "New", "tags": ""})
            if new_rows:
                st.session_state.bulk_preview_df = pd.DataFrame(new_rows)
        
        if st.session_state.bulk_preview_df is not None:
            st.write("### Preview:")
            st.dataframe(st.session_state.bulk_preview_df, hide_index=True)
            if st.button("💾 Confirm & Process Bulk", type="primary"):
                st.session_state.vocab_df = pd.concat([st.session_state.vocab_df, st.session_state.bulk_preview_df]).drop_duplicates(subset=['vocab'], keep='last')
                save_to_github(st.session_state.vocab_df)
                st.success(f"✅ Added {len(st.session_state.bulk_preview_df)} words!")
                st.session_state.bulk_preview_df = None
                time.sleep(0.5)
                st.rerun()

with tab2:
    if st.session_state.vocab_df.empty: st.info("Add words first!")
    else:
        st.subheader(f"✏️ Edit List ({len(st.session_state.vocab_df)} words)")
        search = st.text_input("🔎 Search...", "").lower().strip()
        display_df = st.session_state.vocab_df.copy()
        if search: display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        
        # D16: Paginated view
        page_size = 50
        page = st.number_input("Page", min_value=1, value=1, step=1)
        start = (page - 1) * page_size
        paginated_df = display_df.iloc[start:start + page_size]
        
        edited = st.data_editor(
            paginated_df, 
            num_rows="dynamic", 
            use_container_width=True, 
            hide_index=True, 
            column_config={"status": st.column_config.SelectboxColumn("Status", options=["New", "Done"], required=True)}
        )
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            st.session_state.vocab_df = edited
            save_to_github(st.session_state.vocab_df)
            st.toast("✅ Cloud updated!")
            time.sleep(0.5)
            st.rerun()

with tab3:
    st.subheader("📇 Generate Cyberpunk Anki Deck")
    
    if st.session_state.apkg_buffer is not None:
        st.success("✅ Deck generated successfully! Click below to download and update your database.")
        st.download_button(
            "📥 Download .apkg", 
            data=st.session_state.apkg_buffer, 
            file_name=f"AnkiDeck_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg", 
            mime="application/octet-stream", 
            use_container_width=True,
            on_click=mark_as_done_callback
        )
        if st.button("❌ Cancel / Clear"):
            st.session_state.apkg_buffer = None
            st.session_state.processed_vocabs = []
            st.rerun()

    else:
        if st.session_state.vocab_df.empty: 
            st.info("Add words first!")
        else:
            subset = st.session_state.vocab_df[st.session_state.vocab_df['status'] == 'New'].copy()
            if subset.empty: 
                st.warning("⚠️ No 'New' words to export!")
            else:
                deck_col1, deck_col2 = st.columns([3, 1])
                deck_name_input = deck_col1.text_input("📦 Deck Name", value="-English Learning::Vocabulary")
                if deck_col2.button("🎲 New Deck ID"): 
                    st.session_state.deck_id = random.randrange(1 << 30, 1 << 31)
                deck_col2.caption(f"ID: {st.session_state.deck_id}")

                batch_size = st.slider("⚡ Batch Size (Words per Request)", 1, 15, 10)
                # A3: Auto-adjust
                requests_left = max(0, 20 - st.session_state.rpd_count)
                max_safe_batch = max(1, math.ceil(len(subset) / max(1, requests_left))) if requests_left > 0 else 1
                batch_size = min(batch_size, max_safe_batch)
                st.caption(f"✅ Auto-adjusted effective batch size: {batch_size} (based on remaining quota)")

                include_audio = st.checkbox("🔊 Generate Audio Files", value=True)
                # B10 & A5
                st.session_state.include_antonyms = st.checkbox("➖ Include Antonyms in Card Back", value=st.session_state.include_antonyms)
                st.session_state.dry_run = st.checkbox("🔬 Dry Run Mode (simulate AI, no quota)", value=st.session_state.dry_run)
                
                st.write("**Select words to export:**")
                subset['Export'] = True
                edited_export = st.data_editor(
                    subset, 
                    column_config={"Export": st.column_config.CheckboxColumn("Export?", required=True)}, 
                    hide_index=True,
                    disabled=["vocab", "phrase", "status", "tags"]
                )
                final_export_subset = edited_export[edited_export['Export'] == True]
                
                # D17: Preview table + size estimate
                if not final_export_subset.empty:
                    st.write("### Export Preview")
                    st.dataframe(final_export_subset[['vocab', 'phrase']], hide_index=True)
                    card_count = len(final_export_subset)
                    est_size_kb = card_count * 2.5
                    st.info(f"📊 **{card_count} cards** • Estimated .apkg size: **{est_size_kb:.1f} KB**")
                
                required_requests = math.ceil(len(final_export_subset) / batch_size) if not final_export_subset.empty else 0
                requests_left = max(0, 20 - st.session_state.rpd_count)
                st.info(f"💡 You have **{requests_left}** API requests left today. This batch requires **{required_requests}** requests.")
                
                if final_export_subset.empty:
                    st.warning("Select at least one word to export.")
                elif required_requests > requests_left and not st.session_state.dry_run:
                    st.error("🛑 Exceeds Daily Limit! Reduce your selection or increase batch size.")
                else:
                    if st.button("🚀 Generate Deck", type="primary", use_container_width=True):
                        try:
                            raw_notes = process_anki_data(final_export_subset, batch_size=batch_size, dry_run=st.session_state.dry_run)
                            if raw_notes:
                                apkg_buffer = create_anki_package(raw_notes, deck_name_input, generate_audio=include_audio, deck_id=st.session_state.deck_id, include_antonyms=st.session_state.include_antonyms)
                                st.session_state.apkg_buffer = apkg_buffer.getvalue()
                                st.session_state.processed_vocabs = [n['VocabRaw'] for n in raw_notes]
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Generation error: {e}. Status rolled back to 'New'.")
                            st.session_state.vocab_df.loc[st.session_state.vocab_df['vocab'].isin([n.get('VocabRaw', '') for n in raw_notes]), 'status'] = 'New'
                            save_to_github(st.session_state.vocab_df)
                            if st.button("Retry with smaller batch"):
                                st.rerun()
