import streamlit as st
import pandas as pd
from github import Github
import io

# Setup from Secrets
token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["REPO_NAME"]

st.set_page_config(page_title="Vocab App", layout="centered")
st.title("📚 My Cloud Vocab")

# Connect to GitHub
g = Github(token)
repo = g.get_repo(repo_name)

def load_data():
    try:
        file_content = repo.get_contents("vocabulary.csv")
        return pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
    except:
        return pd.DataFrame(columns=['vocab', 'phrase'])

def save_to_github(dataframe):
    csv_data = dataframe.to_csv(index=False)
    file = repo.get_contents("vocabulary.csv")
    repo.update_file(file.path, "updated vocab", csv_data, file.sha)

df = load_data()

tab1, tab2 = st.tabs(["➕ Add", "✏️ Edit List"])

with tab1:
    with st.form("my_form", clear_on_submit=True):
        v = st.text_input("Vocab:").lower()
        p_raw = st.text_input("Phrase (1 to skip):")
        if st.form_submit_button("Save"):
            p = "" if p_raw == "1" else p_raw.capitalize()
            new_df = pd.concat([df, pd.DataFrame([{"vocab": v, "phrase": p}])], ignore_index=True)
            save_to_github(new_df)
            st.success("Saved!")
            st.rerun()

with tab2:
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if st.button("Save Changes"):
        save_to_github(edited)
        st.success("Cloud Updated!")
        st.rerun()

