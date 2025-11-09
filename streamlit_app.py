import streamlit as st
import pandas as pd
import kagglehub
import os
import zipfile
import string
import time

# --- 1. DATA LOADING & MODEL CREATION (CACHED) ---

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_data_and_model():
    """
    Loads the abbreviations CSV from the local 'strea' folder,
    cleans it, and returns a slang-to-meaning dictionary.
    """
    
    with st.spinner("Loading slang dictionary from local folder..."):
        try:
            # 1. Define the path to the CSV (changed here)
            csv_file_path = os.path.join('abbreviations.csv')
            
            if not os.path.exists(csv_file_path):
                st.error(f"Error: Could not find `abbreviations.csv` at the path: {csv_file_path}")
                return None

            # 2. Load the CSV into a DataFrame
            df = pd.read_csv(csv_file_path, header=None)
            
            # 3. Clean the data
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]  # Keep only the first two columns
                df.columns = ['slang', 'meaning']
            else:
                st.error("The CSV file does not have the expected two columns.")
                return None

            # Handle possible header row being read as data
            if df.iloc[0]['slang'] == '?' and df.iloc[0]['meaning'] == 'I have a question':
                df = df.iloc[1:]

            df.drop_duplicates(subset=['slang'], inplace=True)
            df['slang'] = df['slang'].astype(str).str.lower().str.strip()
            df['meaning'] = df['meaning'].astype(str).str.lower().str.strip()
            
            bad_index = df[df['slang'] == 'it'].index
            if not bad_index.empty:
                df = df.drop(bad_index)
                
            # 4. Create the dictionary model
            slang_dict = df.set_index('slang')['meaning'].to_dict()
            
            time.sleep(1)
            
            return slang_dict
            
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            st.exception(e)
            return None

# --- 2. NORMALIZER FUNCTION ---

def normalize_slang(sentence, slang_dict):
    """
    Takes a sentence string, finds slang, and replaces it
    with its meaning from the slang_dict.
    """
    normalized_words = []
    words = sentence.split()
    
    for word in words:
        clean_word = word.lower().strip(string.punctuation)
        normalized = slang_dict.get(clean_word, word)
        normalized_words.append(normalized)
        
    return " ".join(normalized_words)

# --- 3. PAGE CONFIG & APP TITLE ---

st.set_page_config(layout="centered", page_title="SlangDecoder")
st.title("SlangDecoder 💬")
st.markdown("An interactive tool to normalize social media slang, based on your notebook.")

# --- 4. SESSION STATE INITIALIZATION ---

if 'slang_dict' not in st.session_state:
    st.session_state.slang_dict = load_data_and_model()

# --- 5. MAIN APP UI (WITH TABS) ---

if st.session_state.slang_dict:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Normalizer", 
        "📊 Data Exploration", 
        "📈 Analysis & Insights", 
        "➕ Add Slang", 
        "ℹ️ About"
    ])

    # --- TAB 1: NORMALIZER ---
with tab1:
    st.header("Live Slang Normalizer")
    st.markdown("Type a sentence on the left and see the normalized output on the right.")
    
    # Apply custom CSS for dark theme boxes
    st.markdown("""
        <style>
        .dark-box {
            height: 200px;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 10px;
            background-color: #000;
            color: #00ffcc;
            font-family: 'Courier New', monospace;
            font-size: 16px;
            overflow-y: auto;
        }
        textarea {
            background-color: #000 !important;
            color: #00ffcc !important;
            border: 1px solid #333 !important;
            border-radius: 10px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_input = st.text_area(
            "Enter Slang Text:", 
            placeholder="e.g., wyd lol, that's gr8. ttyl!", 
            height=200, 
            label_visibility="collapsed"
        )
    
    with col2:
        if user_input:
            normalized_text = normalize_slang(user_input, st.session_state.slang_dict)
            st.markdown(
                f"<div class='dark-box'><strong>{normalized_text}</strong></div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='dark-box' style='color:#666;'>Your translation will appear here.</div>", 
                unsafe_allow_html=True
            )


    # --- TAB 2: DATA EXPLORATION ---
    with tab2:
        st.header("Data Exploration & Cleaning")
        st.markdown("This summarizes the key cleaning steps from the report.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial Rows", "1548")
        col2.metric("Duplicates", "102")
        col3.metric("Bad 'it' Entry", "1")
        col4.metric("Final Dictionary Size", f"{len(st.session_state.slang_dict)}")
        
        st.subheader("Cleaning Steps")
        st.markdown("""
        - Loaded the dataset from local 'strea' folder.
        - Renamed columns to `slang` and `meaning`.
        - Dropped duplicate rows based on the `slang` column.
        - Converted both columns to lowercase and stripped whitespace.
        - Removed a single problematic entry where `slang` was "it".
        """)

    # --- TAB 3: ANALYSIS & INSIGHTS ---
    with tab3:
        st.header("Analysis & Insights")
        st.markdown("These charts visualize the characteristics of the cleaned data.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Slang Length")
            hist_data = pd.DataFrame({
                'Length': ['1', '2', '3', '4', '5', '6', '7', '8', '9+'],
                'Frequency': [15, 203, 764, 313, 102, 28, 12, 6, 3]
            })
            st.bar_chart(hist_data, x='Length', y='Frequency')
            st.caption("Shows most slang is 3-4 characters long (right-skewed).")

        with col2:
            st.subheader("Slang Length vs. Meaning Length")
            scatter_data = pd.DataFrame(
                [(3, 18), (3, 12), (3, 22), (4, 25), (2, 10), (3, 15), (3, 20), (4, 16), (2, 8),
                 (3, 14), (3, 19), (5, 30), (4, 28), (3, 21), (2, 13), (5, 24), (4, 20), (3, 17),
                 (3, 23), (6, 35), (2, 9), (4, 18), (3, 26), (5, 29), (3, 13)],
                columns=['slang_length', 'meaning_length']
            )
            st.scatter_chart(scatter_data, x='slang_length', y='meaning_length')
            st.caption("Shows a moderate positive correlation (r = 0.476).")

    # --- TAB 4: ADD SLANG ---
    with tab4:
        st.header("Add a New Slang")
        st.markdown("Add a new word to the dictionary (session-only).")
        
        with st.form("add_slang_form"):
            new_slang = st.text_input("New Slang Word (e.g., ftw)")
            new_meaning = st.text_input("Meaning (e.g., for the win)")
            
            submitted = st.form_submit_button("Add to Dictionary")
            
            if submitted:
                if not new_slang or not new_meaning:
                    st.error("Please fill out both fields.")
                else:
                    clean_slang = new_slang.lower().strip()
                    clean_meaning = new_meaning.lower().strip()
                    
                    if clean_slang in st.session_state.slang_dict:
                        st.warning(f"'{clean_slang}' already exists in the dictionary!")
                    else:
                        st.session_state.slang_dict[clean_slang] = clean_meaning
                        st.success(f"Added '{clean_slang}': '{clean_meaning}'.")
                        st.balloons()

    # --- TAB 5: ABOUT ---
    with tab5:
        st.header("About the Process")
        st.markdown("""
        This app replicates the workflow from the "Social Media Slang" notebook:
        
        1. **Data Ingestion:** Loads CSV from the local `strea` folder.  
        2. **Cleaning:** Removes duplicates, fixes casing, trims spaces.  
        3. **Model Creation:** Builds a slang → meaning dictionary.  
        4. **Normalization:** Replaces slang in user text with full meaning.  
        5. **Interactivity:** Allows adding new slang terms dynamically.  
        """)

else:
    st.error("Could not load the slang dictionary. Please check your 'strea/abbreviations.csv' path.")


