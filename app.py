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
        # Try to find the existing file
        file_content = repo.get_contents("vocabulary.csv")
        return pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
    except:
        # If file doesn't exist yet, return a blank table
        return pd.DataFrame(columns=['vocab', 'phrase'])

def save_to_github(dataframe):
    csv_data = dataframe.to_csv(index=False)
    try:
        # Try to find the file to update it
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, "updated vocab", csv_data, file.sha)
    except:
        # If the file is NOT found, create it for the first time
        repo.create_file("vocabulary.csv", "initial commit", csv_data)

df = load_data()

tab1, tab2 = st.tabs(["➕ Add", "✏️ Edit List"])

with tab1:
    with st.form("my_form", clear_on_submit=True):
        v = st.text_input("Vocab:").lower().strip()
        p_raw = st.text_input("Phrase (1 to skip):").strip()
        if st.form_submit_button("Save"):
            if v: # Only save if vocab isn't empty
                p = "" if p_raw == "1" else p_raw.capitalize()
                new_row = pd.DataFrame([{"vocab": v, "phrase": p}])
                # Use pd.concat to add the new row
                updated_df = pd.concat([df, new_row], ignore_index=True)
                save_to_github(updated_df)
                st.success(f"Added '{v}' successfully!")
                st.rerun()
            else:
                st.error("Please enter a word.")

with tab2:
    if df.empty:
        st.info("The list is empty. Add some words first!")
    else:
        st.subheader("Edit or Delete Rows")
        # num_rows="dynamic" allows you to delete rows by selecting them
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("Save Changes"):
            save_to_github(edited)
            st.success("Cloud Updated!")
            st.rerun()
