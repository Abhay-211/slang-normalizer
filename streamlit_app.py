import streamlit as st
import pandas as pd
import re
import os

# --- Load slang CSV file ---
@st.cache_data
def load_slang_dictionary():
    csv_file_path = os.path.join("strea", "abbreviations.csv")
    if not os.path.exists(csv_file_path):
        st.error(f"Error: Could not find `abbreviations.csv` in the folder: {csv_file_path}")
        return pd.DataFrame(columns=["abbreviation", "expanded form"])
    return pd.read_csv(csv_file_path)

# --- Save slang back to CSV ---
def save_slang(abbr, expanded):
    csv_file_path = os.path.join("strea", "abbreviations.csv")
    df = load_slang_dictionary()
    new_entry = pd.DataFrame([[abbr.lower(), expanded.lower()]], columns=["abbreviation", "expanded form"])
    df = pd.concat([df, new_entry], ignore_index=True).drop_duplicates(subset="abbreviation", keep="last")
    df.to_csv(csv_file_path, index=False)

# --- Normalize slang in text ---
def normalize_slang(text, slang_dict):
    def replace_word(match):
        word = match.group(0).lower()
        return slang_dict.get(word, word)

    if not slang_dict:
        return text

    pattern = r'\b(' + '|'.join(re.escape(k) for k in slang_dict.keys()) + r')\b'
    return re.sub(pattern, replace_word, text, flags=re.IGNORECASE)

# --- Main App ---
def main():
    st.set_page_config(page_title="AI Slang Normalizer", layout="wide")

    # --- CSS Styling ---
    st.markdown("""
        <style>
        body {
            background-color: #000000;
            color: #00ffcc;
        }
        .main {
            background-color: #000000;
        }
        h1, h2, h3, h4 {
            color: #00ffcc;
            text-shadow: 0 0 10px #00ffff;
        }
        .dark-box {
            height: 200px;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 10px;
            background-color: #000000;
            color: #00ffcc;
            font-family: 'Courier New', monospace;
            font-size: 16px;
            overflow-y: auto;
        }
        textarea {
            background-color: #000000 !important;
            color: #00ffcc !important;
            border: 1px solid #333 !important;
            border-radius: 10px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 16px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background-color: #111111;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #00ffcc;
            font-weight: 600;
        }
        .stDataFrame {
            background-color: #111111;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.title("💬 AI Slang Normalizer")
    st.markdown("### Transform Internet Slang into Formal English — Instantly ⚡")

    # --- Load Slang Dictionary ---
    slang_df = load_slang_dictionary()
    slang_dict = dict(zip(slang_df["abbreviation"].str.lower(), slang_df["expanded form"].str.lower()))

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔠 Normalizer", "📘 Dictionary", "➕ Add Slang", "ℹ️ About"])

    # --- TAB 1: NORMALIZER ---
    with tab1:
        st.header("Live Slang Normalizer")
        st.markdown("Type slang text on the left and see its formal version on the right.")

        col1, col2 = st.columns(2)
        with col1:
            user_input = st.text_area(
                "Enter Slang Text:",
                placeholder="e.g., brb lol, that’s gr8. ttyl!",
                height=200,
                label_visibility="collapsed"
            )

        with col2:
            if user_input:
                normalized_text = normalize_slang(user_input, slang_dict)
                st.markdown(
                    f"<div class='dark-box'><strong>{normalized_text}</strong></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='dark-box' style='color:#666;'>Your normalized text will appear here.</div>",
                    unsafe_allow_html=True
                )

    # --- TAB 2: SLANG DICTIONARY ---
    with tab2:
        st.header("Slang Dictionary")
        st.markdown("Browse all slang abbreviations and their meanings below 👇")
        st.dataframe(slang_df, use_container_width=True, hide_index=True)
        st.success(f"✅ Loaded {len(slang_dict)} slang abbreviations successfully!")

    # --- TAB 3: ADD SLANG ---
    with tab3:
        st.header("Add a New Slang Term 🆕")
        st.markdown("Help improve the app by adding your own slang abbreviations!")

        col1, col2 = st.columns(2)
        with col1:
            new_abbr = st.text_input("Slang (Abbreviation)", placeholder="e.g., tbh")
        with col2:
            new_expanded = st.text_input("Expanded Form", placeholder="e.g., to be honest")

        if st.button("Add Slang"):
            if new_abbr and new_expanded:
                save_slang(new_abbr, new_expanded)
                st.success(f"✅ Added '{new_abbr}' → '{new_expanded}' successfully!")
            else:
                st.warning("⚠️ Please fill in both fields before adding.")

    # --- TAB 4: ABOUT PROJECT ---
    with tab4:
        st.header("ℹ️ About the AI Slang Normalizer Project")
        st.markdown("""
        **Project Overview:**
        > The *AI Slang Normalizer* is a smart text preprocessing tool designed to convert internet slang and abbreviations into
        clean, understandable English in real time.

        **Features:**
        - 🔤 Real-time slang-to-formal text conversion  
        - 📖 Slang dictionary viewer and editor  
        - ➕ Add your own custom slang entries  
        - 🖤 Dark neon UI with responsive design  

        **Tech Stack:**
        - 🐍 Python  
        - 🧠 Streamlit  
        - 📄 Pandas  
        - 🗂 CSV for slang dictionary storage  

        **Use Cases:**
        - Data preprocessing for NLP  
        - Social media text cleaning  
        - Chatbot normalization  
        - Fun learning for text slang! 😎

        ---
        **Developed by Abhay Chauhan 💻**
        """)

# --- Run App ---
if __name__ == "__main__":
    main()
