import streamlit as st
import pandas as pd
from github import Github, GithubException
import io

# ========================== SETUP ==========================
st.set_page_config(page_title="Vocab App", layout="centered", page_icon="📚")
st.title("📚 My Cloud Vocab")

# Secrets (keep these safe!)
token = st.secrets["GITHUB_TOKEN"]
repo_name = st.secrets["REPO_NAME"]

# Connect to GitHub (with better error handling)
try:
    g = Github(token)
    repo = g.get_repo(repo_name)
except GithubException as e:
    st.error(f"❌ GitHub connection failed: {e}")
    st.stop()

# ========================== DATA FUNCTIONS ==========================
def load_data():
    """Load vocabulary.csv from GitHub or return empty DF"""
    try:
        file_content = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(file_content.decoded_content.decode('utf-8')))
        # Always sort alphabetically by vocab (great for a vocab app!)
        df = df.sort_values(by="vocab", ignore_index=True)
        return df
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=['vocab', 'phrase'])
        else:
            st.error(f"❌ Failed to load data: {e}")
            return pd.DataFrame(columns=['vocab', 'phrase'])
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {e}")
        return pd.DataFrame(columns=['vocab', 'phrase'])


def save_to_github(dataframe):
    """Save DF to GitHub (create or update) with proper exception handling"""
    csv_data = dataframe.to_csv(index=False)
    commit_message = "Updated vocabulary list"

    try:
        # Try to update existing file
        file = repo.get_contents("vocabulary.csv")
        repo.update_file(file.path, commit_message, csv_data, file.sha)
    except GithubException as e:
        if e.status == 404:
            # File doesn't exist → create it
            repo.create_file("vocabulary.csv", "Initial vocab commit", csv_data)
        else:
            st.error(f"❌ GitHub save failed: {e}")
            return False
    except Exception as e:
        st.error(f"❌ Unexpected save error: {e}")
        return False
    return True


# ========================== LOAD DATA ==========================
df = load_data()

# ========================== UI ==========================
tab1, tab2 = st.tabs(["➕ Add New Word", "✏️ Edit / Delete"])

# ===================== TAB 1: ADD =====================
with tab1:
    st.subheader("Add a new vocabulary word")
    
    with st.form("add_form", clear_on_submit=True):
        v = st.text_input("📝 Vocab (required)", placeholder="e.g. serendipity").lower().strip()
        p_raw = st.text_input(
            "🔤 Phrase / Example (type 1 to skip)", 
            placeholder="I found it by serendipity!"
        ).strip()
        
        submitted = st.form_submit_button("💾 Save to Cloud", use_container_width=True)
        
        if submitted:
            if not v:
                st.error("❌ Vocab cannot be empty!")
            else:
                # === DUPLICATE CHECK ===
                if v in df['vocab'].values:
                    st.warning(f"⚠️ '{v}' already exists in your list!")
                else:
                    p = "" if p_raw.upper() == "1" else p_raw.capitalize()
                    new_row = pd.DataFrame([{"vocab": v, "phrase": p}])
                    updated_df = pd.concat([df, new_row], ignore_index=True)
                    
                    if save_to_github(updated_df):
                        st.success(f"✅ '{v}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save to GitHub.")

# ===================== TAB 2: EDIT =====================
with tab2:
    if df.empty:
        st.info("📭 Your list is empty. Start adding words in the first tab!")
    else:
        st.subheader(f"✏️ Edit List ({len(df)} words)")
        
        # Optional: Quick search filter (very useful!)
        search = st.text_input("🔎 Search vocab...", "").lower().strip()
        display_df = df.copy()
        if search:
            display_df = display_df[display_df['vocab'].str.contains(search, case=False)]
        
        # Nice column config for better UX
        edited = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "vocab": st.column_config.TextColumn(
                    "Vocab",
                    required=True,
                    help="The word/phrase you want to remember"
                ),
                "phrase": st.column_config.TextColumn(
                    "Phrase / Example (leave blank to skip)",
                    help="Example sentence or meaning"
                ),
            }
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("💾 Save Changes to Cloud", type="primary", use_container_width=True):
                # Re-apply sorting after edit
                edited = edited.sort_values(by="vocab", ignore_index=True)
                if save_to_github(edited):
                    st.success("✅ Cloud updated successfully!")
                    st.rerun()
                else:
                    st.error("Save failed.")
        
        with col2:
            if st.button("📥 Download CSV", use_container_width=True):
                csv = edited.to_csv(index=False).encode()
                st.download_button(
                    label="Download",
                    data=csv,
                    file_name="my_vocabulary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ===================== FOOTER =====================
st.caption("💡 Your vocab is safely stored in a private GitHub repo. Changes sync instantly!")
