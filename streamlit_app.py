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
    Downloads the dataset from Kaggle, unzips it, loads the CSV into a pandas
    DataFrame, cleans it, and returns the slang-to-meaning dictionary.
    This entire process is cached by Streamlit.
    """
    
    # Use st.spinner to show a message during this long operation
    with st.spinner("Downloading and preparing slang dictionary... This may take a moment on first load."):
        try:
            # 1. Download the dataset
            path = kagglehub.dataset_download("rizdelhi/socialmediaabbreviations")
            
            # 2. Unzip if necessary
            zip_file_name = None
            for file in os.listdir(path):
                if file.endswith('.zip'):
                    zip_file_name = file
                    break
            
            if zip_file_name:
                zip_path = os.path.join(path, zip_file_name)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(path)
            
            # 3. Define the path to the CSV
            csv_file_path = os.path.join(path, 'abbreviations.csv')
            
            if not os.path.exists(csv_file_path):
                st.error(f"Error: Could not find `abbreviations.csv` in the downloaded path: {path}")
                return None

            # 4. Load the CSV into a DataFrame
            # Correctly read the CSV which has no header
            df = pd.read_csv(csv_file_path, header=None)
            
            # 5. Clean the data (as per your notebook's logic)
            # Add a check for the number of columns before renaming
            if df.shape[1] >= 2:
                 df = df.iloc[:, :2] # Keep only the first two columns
                 df.columns = ['slang', 'meaning']
            else:
                 st.error("The CSV file does not have the expected two columns.")
                 return None

            # Handle potential header row being read as data
            if df.iloc[0]['slang'] == '?' and df.iloc[0]['meaning'] == 'I have a question':
                 df = df.iloc[1:]

            df.drop_duplicates(subset=['slang'], inplace=True)
            df['slang'] = df['slang'].astype(str).str.lower().str.strip()
            df['meaning'] = df['meaning'].astype(str).str.lower().str.strip()
            
            bad_index = df[df['slang'] == 'it'].index
            if not bad_index.empty:
                df = df.drop(bad_index)
                
            # 6. Create the "model" (the dictionary)
            slang_dict = df.set_index('slang')['meaning'].to_dict()
            
            time.sleep(2) 
            
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

# Set the page to a centered, cleaner layout
st.set_page_config(layout="centered", page_title="SlangDecoder")
st.title("SlangDecoder 💬")
st.markdown("An interactive tool to normalize social media slang, based on your notebook.")

# --- 4. SESSION STATE INITIALIZATION ---

# Use st.session_state to store the dictionary.
# This allows us to add new words to it during the session!
if 'slang_dict' not in st.session_state:
    st.session_state.slang_dict = load_data_and_model()

# --- 5. MAIN APP UI (WITH TABS) ---

if st.session_state.slang_dict:
    # Create the tabs, matching the HTML demo
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
                # We use the dictionary from st.session_state
                normalized_text = normalize_slang(user_input, st.session_state.slang_dict)
                st.markdown(
                    f"<div style='height: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;'><strong>{normalized_text}</strong></div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='height: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; color: #888;'>Your translation will appear here.</div>", 
                    unsafe_allow_html=True
                )

    # --- TAB 2: DATA EXPLORATION ---
    with tab2:
        st.header("Data Exploration & Cleaning")
        st.markdown("This summarizes the key cleaning steps from the report.")
        
        # Use columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Initial Rows", "1548")
        col2.metric("Duplicates", "102")
        col3.metric("Bad 'it' Entry", "1")
        # This metric is dynamic! It will update if you add a new word.
        col4.metric("Final Dictionary Size", f"{len(st.session_state.slang_dict)}")
        
        st.subheader("Cleaning Steps")
        st.markdown("""
        - Loaded the dataset from Kaggle.
        - Renamed columns to `slang` and `meaning`.
        - Checked for null values (found 0).
        - Dropped 102 duplicate rows based on the `slang` column.
        - Converted both columns to lowercase and stripped whitespace.
        - Identified and removed a single problematic entry where `slang` was "it".
        """)

    # --- TAB 3: ANALYSIS & INSIGHTS ---
    with tab3:
        st.header("Analysis & Insights")
        st.markdown("These charts visualize the characteristics of the cleaned data, just like in the notebook.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Slang Length")
            # Pre-binned data for the histogram (from the notebook/HTML file)
            hist_data = pd.DataFrame({
                'Length': ['1', '2', '3', '4', '5', '6', '7', '8', '9+'],
                'Frequency': [15, 203, 764, 313, 102, 28, 12, 6, 3]
            })
            st.bar_chart(hist_data, x='Length', y='Frequency')
            st.caption("Shows most slang is 3-4 characters long (right-skewed).")

        with col2:
            st.subheader("Slang Length vs. Meaning Length")
            # Sample data for the scatter plot
            scatter_data = pd.DataFrame(
                [(3, 18), (3, 12), (3, 22), (4, 25), (2, 10), (3, 15), (3, 20), (4, 16), (2, 8),
                 (3, 14), (3, 19), (5, 30), (4, 28), (3, 21), (2, 13), (5, 24), (4, 20), (3, 17),
                 (3, 23), (6, 35), (2, 9), (4, 18), (3, 26), (5, 29), (3, 13)],
                columns=['slang_length', 'meaning_length']
            )
            st.scatter_chart(scatter_data, x='slang_length', y='meaning_length')
            st.caption("Shows a moderate positive correlation (r = 0.476).")

    # --- TAB 4: ADD SLANG (NEW FEATURE!) ---
    with tab4:
        st.header("Add a New Slang")
        st.markdown("Add a new word to the dictionary. **Note:** This is for your current session only and will be reset if you refresh the page.")
        
        # Use a form so the page doesn't re-run on every key press
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
                        st.warning(f"Error: '{clean_slang}' already exists in the dictionary!")
                    else:
                        # This is where we update the dictionary in our session
                        st.session_state.slang_dict[clean_slang] = clean_meaning
                        st.success(f"Success! Added '{clean_slang}': '{clean_meaning}'.")
                        st.balloons()

    # --- TAB 5: ABOUT THE PROCESS ---
    with tab5:
        st.header("About the Process")
        st.markdown("""
        This interactive application is a direct translation of the process outlined in the source "Social Media Slang" Jupyter Notebook.
        
        1.  **Data Ingestion:** Downloads the dataset from Kaggle.
        2.  **Cleaning & Processing:** The raw CSV is loaded and cleaned (see 'Data Exploration' tab).
        3.  **"Model" Building:** A Python dictionary is created from the cleaned `slang` (key) and `meaning` (value) columns.
        4.  **"Prediction" Function:** The `normalize_slang` function splits a sentence and looks up each word in the dictionary.
        5.  **Deployment:** This Streamlit app replicates and enhances the notebook's final console app, running the same logic on the full dataset.
        """)

else:
    st.error("Could not load the slang dictionary. The app cannot function. Please check your connection or the Kaggle Hub link.")