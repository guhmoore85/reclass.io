import streamlit as st
import pandas as pd
import requests
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

# --- START: New NLTK Resource Downloader ---
@st.cache_resource
def download_nltk_resources():
    """Checks for NLTK resources and downloads them if missing."""
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "corpora/wordnet.zip": "wordnet",
        "corpora/omw-1.4.zip": "omw-1.4"
    }
    for path, package_id in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            with st.spinner(f"Downloading required NLTK resource: {package_id}..."):
                nltk.download(package_id)

# Run the downloader function
download_nltk_resources()
# --- END: New NLTK Resource Downloader ---


st.set_page_config(page_title="WTO Tariff Lookup", layout="wide")

API_KEY = "ab5ad8703cd54ffba080cb9554175101" # IMPORTANT: Replace with your actual API key

# --- Initialize NLTK components after ensuring they are downloaded ---
nltk_stopwords_english = set(nltk_stopwords.words('english'))
custom_stop_words_for_extraction = set([
    "parts", "thereof", "subsidiary", "chapter", "whether", "other", "nesoi",
    "product", "hs", "code", "description", "including", "etc"
])
STOP_WORDS_FOR_EXTRACTION = nltk_stopwords_english.union(custom_stop_words_for_extraction)
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---
def lemmatize_string_for_search(text_to_lemmatize):
    """Lemmatizes a string for better search matching. Keeps all words, just lemmatized."""
    if not isinstance(text_to_lemmatize, str) or not text_to_lemmatize.strip(): 
        return ""
    try:
        words = word_tokenize(text_to_lemmatize.lower())
    except Exception: # Fallback if word_tokenize fails
        words = re.findall(r'\b\w+\b', text_to_lemmatize.lower())
    
    lemmatized_list = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') 
                       for word in words if word.isalpha()]
    return " ".join(lemmatized_list)

def create_pivot_table(df, value_columns_list):
    """Create a pivot table with HS6 codes and Description as rows, 
       Years as second level columns, and specified metrics as first level columns."""
    if df.empty or not value_columns_list or "Year" not in df.columns or "HS6" not in df.columns:
        return pd.DataFrame()
        
    df_copy = df.copy()
    df_copy["Year"] = df_copy["Year"].fillna("Unknown_Year").astype(str)
    
    valid_value_columns = []
    for val_col in value_columns_list:
        if val_col not in df_copy.columns:
            continue 
        df_copy[val_col] = pd.to_numeric(df_copy[val_col], errors='coerce')
        valid_value_columns.append(val_col)
    
    if not valid_value_columns: 
        return pd.DataFrame()

    desc_col_name = "Pivot Description"
    if "HTS8 Local Description" in df_copy.columns and df_copy["HTS8 Local Description"].notna().any():
        df_copy[desc_col_name] = df_copy["HTS8 Local Description"].fillna("N/A")
    elif "Description" in df_copy.columns: 
        df_copy[desc_col_name] = df_copy["Description"].fillna("N/A")
    else: 
        return pd.DataFrame() 
        
    columns_to_check_na_for_index = ["HS6", desc_col_name, "Year"]
    df_copy.dropna(subset=columns_to_check_na_for_index, inplace=True)
    df_copy.dropna(subset=valid_value_columns, how='all', inplace=True)

    if df_copy.empty:
        return pd.DataFrame()

    try:
        pivot = df_copy.pivot_table(index=["HS6", desc_col_name], columns="Year", 
                                    values=valid_value_columns, aggfunc="first")
        if pivot.empty: return pd.DataFrame()
        if isinstance(pivot.columns, pd.MultiIndex) and len(pivot.columns.levels) > 1:
            try:
                year_level_values = pivot.columns.levels[1].astype(str)
                sorted_years = sorted(year_level_values, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
                
                new_column_order = []
                for metric in valid_value_columns: 
                    for year in sorted_years:
                        if (metric, year) in pivot.columns:
                            new_column_order.append((metric, year))
                if new_column_order: pivot = pivot[new_column_order]
            except: pass 

        pivot = pivot.reset_index().rename(columns={desc_col_name: "Description (Local or API)"})
        return pivot
    except Exception: return pd.DataFrame()

@st.cache_data
def load_hts_data(file_path="hts8.csv"):
    try:
        # Force the hts8 column to be read as a string to preserve leading zeros
        df = pd.read_csv(file_path, dtype={'hts8': str}) 
        if "hts8" not in df.columns or "brief_description" not in df.columns:
            st.error(f"HTS data file ('{file_path}') missing 'hts8' or 'brief_description'.")
            return pd.DataFrame(columns=["hts8", "hs6", "brief_description", "searchable_description"])
        df["hts8"] = df["hts8"].astype(str).str.strip().str.zfill(8)
        df["hs6"] = df["hts8"].str[:6]
        df["brief_description"] = df["brief_description"].astype(str) 
        df = df.drop_duplicates(subset=["hts8", "brief_description"]) 
        
        if not df.empty:
            df['searchable_description'] = df['brief_description'].apply(lemmatize_string_for_search)
        else:
            df['searchable_description'] = pd.Series(dtype='str')
            
        if df.empty: st.warning(f"HTS data file ('{file_path}') loaded but empty/became empty.")
        return df
    except FileNotFoundError:
        st.error(f"\u26a0\ufe0f HTS data file ('{file_path}') not found. Comparisons unavailable.")
        return pd.DataFrame(columns=["hts8", "hs6", "brief_description", "searchable_description"])
    except Exception as e: 
        st.error(f"Error loading HTS data: {e}")
        return pd.DataFrame(columns=["hts8", "hs6", "brief_description", "searchable_description"])

@st.cache_data
def load_country_list():
    fallback_countries = [
        {"name": "Afghanistan", "code": "004"}, {"name": "Albania", "code": "008"}, 
        {"name": "Algeria", "code": "012"}, {"name": "Angola", "code": "024"}, {"name": "Antigua and Barbuda", "code": "028"},
        {"name": "Argentina", "code": "032"}, {"name": "Armenia", "code": "051"}, {"name": "Australia", "code": "036"},
        {"name": "Austria", "code": "040"}, {"name": "Bahrain", "code": "048"}, {"name": "Bangladesh", "code": "050"},
        {"name": "Barbados", "code": "052"}, {"name": "Belgium", "code": "056"}, {"name": "Belize", "code": "084"},
        {"name": "Benin", "code": "204"}, {"name": "Bolivia", "code": "068"}, {"name": "Botswana", "code": "072"},
        {"name": "Brazil", "code": "076"}, {"name": "Brunei Darussalam", "code": "096"}, {"name": "Bulgaria", "code": "100"},
        {"name": "Burkina Faso", "code": "854"}, {"name": "Burundi", "code": "108"}, {"name": "Cambodia", "code": "116"},
        {"name": "Cameroon", "code": "120"}, {"name": "Canada", "code": "124"}, {"name": "Cape Verde", "code": "132"},
        {"name": "Central African Republic", "code": "140"}, {"name": "Chad", "code": "148"}, {"name": "Chile", "code": "152"},
        {"name": "China", "code": "156"}, {"name": "Colombia", "code": "170"}, {"name": "Congo", "code": "178"},
        {"name": "Costa Rica", "code": "188"}, {"name": "Côte d'Ivoire", "code": "384"}, {"name": "Croatia", "code": "191"},
        {"name": "Cuba", "code": "192"}, {"name": "Cyprus", "code": "196"}, {"name": "Czech Republic", "code": "203"},
        {"name": "Democratic Republic of the Congo", "code": "180"}, {"name": "Denmark", "code": "208"}, {"name": "Djibouti", "code": "262"},
        {"name": "Dominica", "code": "212"}, {"name": "Dominican Republic", "code": "214"}, {"name": "Ecuador", "code": "218"},
        {"name": "Egypt", "code": "818"}, {"name": "El Salvador", "code": "222"}, {"name": "Estonia", "code": "233"},
        {"name": "Eswatini", "code": "748"}, {"name": "European Union", "code": "918"}, {"name": "Fiji", "code": "242"},
        {"name": "Finland", "code": "246"}, {"name": "France", "code": "250"}, {"name": "Gabon", "code": "266"},
        {"name": "Gambia", "code": "270"}, {"name": "Georgia", "code": "268"}, {"name": "Germany", "code": "276"},
        {"name": "Ghana", "code": "288"}, {"name": "Greece", "code": "300"}, {"name": "Grenada", "code": "308"},
        {"name": "Guatemala", "code": "320"}, {"name": "Guinea", "code": "324"}, {"name": "Guinea-Bissau", "code": "624"},
        {"name": "Guyana", "code": "328"}, {"name": "Haiti", "code": "332"}, {"name": "Honduras", "code": "340"},
        {"name": "Hong Kong, China", "code": "344"}, {"name": "Hungary", "code": "348"}, {"name": "Iceland", "code": "352"},
        {"name": "India", "code": "356"}, {"name": "Indonesia", "code": "360"}, {"name": "Ireland", "code": "372"},
        {"name": "Israel", "code": "376"}, {"name": "Italy", "code": "380"}, {"name": "Jamaica", "code": "388"},
        {"name": "Japan", "code": "392"}, {"name": "Jordan", "code": "400"}, {"name": "Kazakhstan", "code": "398"},
        {"name": "Kenya", "code": "404"}, {"name": "Korea, Republic of", "code": "410"}, {"name": "Kuwait", "code": "414"},
        {"name": "Kyrgyz Republic", "code": "417"}, {"name": "Lao People's Democratic Republic", "code": "418"}, {"name": "Latvia", "code": "428"},
        {"name": "Lesotho", "code": "426"}, {"name": "Liberia", "code": "430"}, {"name": "Liechtenstein", "code": "438"},
        {"name": "Lithuania", "code": "440"}, {"name": "Luxembourg", "code": "442"}, {"name": "Macao, China", "code": "446"},
        {"name": "Madagascar", "code": "450"}, {"name": "Malawi", "code": "454"}, {"name": "Malaysia", "code": "458"},
        {"name": "Maldives", "code": "462"}, {"name": "Mali", "code": "466"}, {"name": "Malta", "code": "470"},
        {"name": "Mauritania", "code": "478"}, {"name": "Mauritius", "code": "480"}, {"name": "Mexico", "code": "484"},
        {"name": "Moldova", "code": "498"}, {"name": "Mongolia", "code": "496"}, {"name": "Montenegro", "code": "499"},
        {"name": "Morocco", "code": "504"}, {"name": "Mozambique", "code": "508"}, {"name": "Myanmar", "code": "104"},
        {"name": "Namibia", "code": "516"}, {"name": "Nepal", "code": "524"}, {"name": "Netherlands", "code": "528"},
        {"name": "New Zealand", "code": "554"}, {"name": "Nicaragua", "code": "558"}, {"name": "Niger", "code": "562"},
        {"name": "Nigeria", "code": "566"}, {"name": "North Macedonia", "code": "807"}, {"name": "Norway", "code": "578"},
        {"name": "Oman", "code": "512"}, {"name": "Pakistan", "code": "586"}, {"name": "Panama", "code": "591"},
        {"name": "Papua New Guinea", "code": "598"}, {"name": "Paraguay", "code": "600"}, {"name": "Peru", "code": "604"},
        {"name": "Philippines", "code": "608"}, {"name": "Poland", "code": "616"}, {"name": "Portugal", "code": "620"},
        {"name": "Qatar", "code": "634"}, {"name": "Romania", "code": "642"}, {"name": "Russian Federation", "code": "643"},
        {"name": "Rwanda", "code": "646"}, {"name": "Saint Kitts and Nevis", "code": "659"}, {"name": "Saint Lucia", "code": "662"},
        {"name": "Saint Vincent and the Grenadines", "code": "670"}, {"name": "Samoa", "code": "882"}, {"name": "Saudi Arabia", "code": "682"},
        {"name": "Senegal", "code": "686"}, {"name": "Seychelles", "code": "690"}, {"name": "Sierra Leone", "code": "694"},
        {"name": "Singapore", "code": "702"}, {"name": "Slovak Republic", "code": "703"}, {"name": "Slovenia", "code": "705"},
        {"name": "Solomon Islands", "code": "090"}, {"name": "South Africa", "code": "710"}, {"name": "Spain", "code": "724"},
        {"name": "Sri Lanka", "code": "144"}, {"name": "Suriname", "code": "740"}, {"name": "Sweden", "code": "752"},
        {"name": "Switzerland", "code": "756"}, {"name": "Chinese Taipei", "code": "158"}, {"name": "Tajikistan", "code": "762"},
        {"name": "Tanzania", "code": "834"}, {"name": "Thailand", "code": "764"}, {"name": "Togo", "code": "768"},
        {"name": "Tonga", "code": "776"}, {"name": "Trinidad and Tobago", "code": "780"}, {"name": "Tunisia", "code": "788"},
        {"name": "Turkey", "code": "792"}, {"name": "Uganda", "code": "800"}, {"name": "Ukraine", "code": "804"},
        {"name": "United Arab Emirates", "code": "784"}, {"name": "United Kingdom", "code": "826"}, {"name": "United States of America", "code": "840"},
        {"name": "Uruguay", "code": "858"}, {"name": "Vanuatu", "code": "548"}, {"name": "Venezuela", "code": "862"},
        {"name": "Viet Nam", "code": "704"}, {"name": "Yemen", "code": "887"}, {"name": "Zambia", "code": "894"},
        {"name": "Zimbabwe", "code": "716"}
    ]
    try: 
        url = "https://api.wto.org/timeseries/v1/reporters"
        headers = {"Ocp-Apim-Subscription-Key": API_KEY}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            countries_data = response.json()
            if countries_data and isinstance(countries_data, list): 
                countries = [{"name": str(c.get("name","")).strip(), "code": str(c.get("code","")).strip()} 
                             for c in countries_data if c.get("name") and c.get("code")]
                unique_countries_dict = {c['name']: c for c in reversed(countries)}
                countries = sorted(list(unique_countries_dict.values()), key=lambda x: ["name"])
                return countries if countries else fallback_countries
            return fallback_countries
        return fallback_countries
    except Exception as e:
        return fallback_countries

@st.cache_data 
def fetch_tariff_data(indicator_code, hs6_list_str, year_range, reporter_code, partner_code=None):
    hs6_list = [hs6.strip() for hs6 in hs6_list_str.split(',') if hs6.strip()]
    if not hs6_list: return pd.DataFrame()
        
    params = {
        "i": indicator_code, "pc": ",".join(hs6_list), "ps": year_range,
        "r": reporter_code, "subscription-key": API_KEY,
    }
    if partner_code: params["p"] = partner_code
    
    try:
        response = requests.get(API_URL, params=params, timeout=60) 
        response.raise_for_status() 
        data = response.json()
        if "Dataset" in data and data["Dataset"]:
            records = []
            for item in data["Dataset"]:
                record = {
                    "Reporter": item.get("ReportingEconomy"), "Reporter Code": item.get("ReportingEconomyCode"),
                    "HS6": item.get("ProductOrSectorCode"), "Description": item.get("ProductOrSector"), 
                    "Year": item.get("TimeDimensionValue"), indicator_code: item.get("Value") 
                }
                if "PartnerEconomy" in item or "Partner" in item : 
                    record["Partner"] = item.get("PartnerEconomy", item.get("Partner"))
                    record["Partner Code"] = item.get("PartnerEconomyCode", item.get("PartnerCode"))
                records.append(record)
            return pd.DataFrame(records)
        return pd.DataFrame()
    except requests.exceptions.HTTPError:
        return pd.DataFrame()
    except requests.exceptions.Timeout: return pd.DataFrame()
    except Exception: return pd.DataFrame()

def extract_keywords_nltk(text, num_keywords=3, min_len=4):
    """Extracts a few key NLTK-processed keywords for finding comparison products."""
    if not isinstance(text, str) or not text.strip(): 
        return []
    try: 
        words = word_tokenize(text.lower())
    except Exception: 
        words = re.findall(r'\b\w+\b', text.lower())
        
    lemmatized_keywords = []
    for word in words:
        if word.isalpha() and len(word) >= min_len and word not in STOP_WORDS_FOR_EXTRACTION: 
            lemma = lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') 
            if lemma not in STOP_WORDS_FOR_EXTRACTION and len(lemma) >= min_len:
                 lemmatized_keywords.append(lemma)
    
    distinct_keywords = []
    for k in lemmatized_keywords:
        if k not in distinct_keywords:
            distinct_keywords.append(k)
            if len(distinct_keywords) == num_keywords:
                break
    return distinct_keywords

def display_product_data_tab(data_df, tab_title_suffix, search_context_info=None):
    """Helper function to display product data in a consistent way across tabs."""
    st.subheader(f"Data for {tab_title_suffix}")
    if search_context_info: st.caption(search_context_info)

    # This Reclassification Helper section is unchanged and correct.
    with st.expander("Reclassification Helper"):
        st.write("Select a product from the raw data table below and click the button to start a reclassification request.")
        selected_hs_for_reclass = st.text_input("Enter the HS6 code of the product you wish to reclassify:", key=f"reclass_hs_{tab_title_suffix.replace(' ', '_')}")

        if st.button(f"Start Reclassification for {selected_hs_for_reclass}", key=f"reclass_btn_{tab_title_suffix.replace(' ', '_')}"):
            if selected_hs_for_reclass:
                product_row = data_df[data_df['HS6'] == selected_hs_for_reclass]
                if not product_row.empty:
                    product_row = product_row.iloc[0]
                    st.session_state.reclassification_data = {
                        "current_hs": product_row['HS6'],
                        "current_desc": product_row.get("HTS8 Local Description") or product_row.get("Description") or "N/A"
                    }
                    st.session_state.app_page = "Reclassification Helper"
                    st.rerun()
                else:
                    st.warning(f"HS6 code {selected_hs_for_reclass} not found in this table.")
            else:
                st.warning("Please enter an HS6 code from the table above.")
    
    st.markdown("---") # Divider
    
    if not data_df.empty:
        # MODIFIED: Add "Preferential Rate (%)" to this list to include it in the main pivot table.
        rate_metrics_to_pivot = [col for col in ["Applied Rate (%)", "Bound Rate (%)", "Preferential Rate (%)", "Overhang (%)"] if col in data_df.columns]
        
        if rate_metrics_to_pivot:
            pivot_title = "Tariff Rates & Overhang"
            
            # ADDED: A help tooltip to explain the different tariff rates.
            help_text = """
            - **Applied Rate:** The standard rate actually charged on imports.
            - **Bound Rate:** The maximum rate a country has committed to at the WTO.
            - **Preferential Rate:** A lower rate (often 0%) available under a trade agreement.
            - **Overhang:** The difference between the Bound and Applied rates.
            """
            st.write(f"**{pivot_title}:**", help=help_text)

            pivot_rates_df = create_pivot_table(data_df.copy(), rate_metrics_to_pivot)
            if not pivot_rates_df.empty:
                s = pivot_rates_df.style
                
                # MODIFIED: Add a color map for the new "Preferential Rate" column.
                col_map = {"Applied Rate (%)": "Blues", "Bound Rate (%)": "Greens", "Preferential Rate (%)": "Purples", "Overhang (%)": "coolwarm"}
                highlight_min_map = {"Applied Rate (%)": '#a2f7a2', "Bound Rate (%)": '#add8e6'}
                
                year_sub_cols = []
                if isinstance(pivot_rates_df.columns, pd.MultiIndex) and len(pivot_rates_df.columns.levels) > 1:
                    year_sub_cols = pivot_rates_df.columns.levels[1].tolist()
                elif not pivot_rates_df.empty: 
                    year_sub_cols = [col for col in pivot_rates_df.columns if col not in ['HS6', 'Description (Local or API)']]
                
                for metric, cmap in col_map.items():
                    if metric not in rate_metrics_to_pivot: continue
                    cols_to_style = []
                    if isinstance(pivot_rates_df.columns, pd.MultiIndex):
                        cols_to_style = [(metric, yr) for yr in year_sub_cols if (metric, yr) in pivot_rates_df.columns]
                    elif metric in pivot_rates_df.columns:
                        cols_to_style = [metric] if not year_sub_cols else [yr for yr in year_sub_cols if yr in pivot_rates_df.columns]

                    valid_cols = [c for c in cols_to_style if c in pivot_rates_df.columns]
                    if valid_cols:
                        s = s.background_gradient(cmap=cmap, subset=valid_cols)
                        if metric in highlight_min_map and tab_title_suffix != "Your Selected Products":
                            s = s.highlight_min(axis=0, color=highlight_min_map[metric], subset=valid_cols)
                st.dataframe(s, use_container_width=True)
        
        # REMOVED: The separate pivot table for "FTA Tariff Rate" is no longer needed.
        
        with st.expander(f"View Raw Data for {tab_title_suffix}", expanded=False):
            st.dataframe(data_df)
    else:
        st.info(f"No API data found for {tab_title_suffix.lower()}.")

# --- Main App Logic ---

# Initialize session state for multi-page navigation and storing results
if 'app_page' not in st.session_state:
    st.session_state.app_page = "Search"
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = {}
if 'reclassification_data' not in st.session_state:
    st.session_state.reclassification_data = {}

# --- Sidebar Navigation ---
st.sidebar.title("App Navigation")

def set_page():
    """Callback function to update page state from the radio button."""
    st.session_state.app_page = st.session_state.sidebar_radio_key

page_options = ["Search", "Compare Results", "Reclassification Helper", "News & Overview"]
try:
    current_page_index = page_options.index(st.session_state.app_page)
except ValueError:
    current_page_index = 0

st.sidebar.radio("Go to", page_options, 
                 key="sidebar_radio_key", 
                 on_change=set_page,
                 index=current_page_index
                ) 
st.sidebar.markdown("---") 

# Load data once at the top
hts_df = load_hts_data() 
countries = load_country_list()
country_names = [country["name"] for country in countries if country.get("name")]
# --- Page 1: Search ---
if st.session_state.app_page == "Search":
    st.title("🔎 Search & Compare WTO Tariffs")
    st.subheader("Use the filters in the sidebar to find and select products.")
    
    st.sidebar.header("🔍 Query Filters")
    search_term = st.sidebar.text_input("Search Product Description (in local HTS data)", help="e.g., motorcycles, steel pipes, rice")
    code_prefix = st.sidebar.text_input("Filter by HTS or HS Code Prefix (in local HTS data)", help="e.g., 1006, 7210")

    current_actual_year = pd.Timestamp.now().year
    min_hist_year = 2010 
    available_years = sorted(list(set(range(min_hist_year, current_actual_year + 1))), reverse=True)

    use_year_range = st.sidebar.checkbox("Use Year Range", value=True)
    if use_year_range:
        default_start_year = current_actual_year - 3 
        default_end_year = current_actual_year   
        if default_start_year < min(available_years): default_start_year = min(available_years)
        if default_end_year > max(available_years): default_end_year = max(available_years)
        year_range_input = st.sidebar.text_input("Year Range (e.g., 2019-2023 or 2020)", 
            value=f"{default_start_year}-{default_end_year}",
            help=f"Single year or range. Latest available data might be for {current_actual_year-1} or earlier.")
        year_range_to_send = year_range_input.strip()
    else:
        selected_year = st.sidebar.selectbox("Select Year", available_years, index=0)
        year_range_to_send = str(selected_year)

    default_reporter_name = "United States of America"
    reporter_idx = country_names.index(default_reporter_name) if default_reporter_name in country_names else 0
    reporter_name = st.sidebar.selectbox("Select Reporter Country", country_names, index=reporter_idx)
    reporter_code = next((c["code"] for c in countries if c["name"] == reporter_name), None)

    partner_options = ["All (Worldwide)"] + country_names 
    partner_name = st.sidebar.selectbox("Select Partner Country (Optional)", partner_options, index=0)
    partner_code = None
    if partner_name != "All (Worldwide)":
        partner_code = next((c["code"] for c in countries if c["name"] == partner_name), None)

    st.header("📍 Product Selection (from local HTS data)")
    # Lemmatized Sidebar Keyword Search Logic
    filtered_hts_df_for_multiselect = hts_df.copy() 
    if not filtered_hts_df_for_multiselect.empty:
        if search_term:
            processed_search_term = lemmatize_string_for_search(search_term)
            search_query_words = [word for word in processed_search_term.split() if word] 
            
            if search_query_words and 'searchable_description' in filtered_hts_df_for_multiselect.columns:
                # Apply AND logic: all keywords must be present
                for kw in search_query_words:
                    filtered_hts_df_for_multiselect = filtered_hts_df_for_multiselect[
                        filtered_hts_df_for_multiselect['searchable_description'].str.contains(re.escape(kw), case=False, na=False, regex=True)
                    ]
            
        if code_prefix: 
            filtered_hts_df_for_multiselect = filtered_hts_df_for_multiselect[
                filtered_hts_df_for_multiselect["hts8"].astype(str).str.startswith(code_prefix) |
                filtered_hts_df_for_multiselect["hs6"].astype(str).str.startswith(code_prefix)
            ]
    else:
        st.warning("HTS data ('hts8.xlsx') not loaded or empty. Product selection from local file is unavailable.")

    selected_hts8_codes = []
    if not filtered_hts_df_for_multiselect.empty: 
        selected_hts8_codes = st.multiselect(
            label="Select HTS8 Codes (max 10 recommended for performance)",
            options=filtered_hts_df_for_multiselect["hts8"].unique().tolist(), 
            default=filtered_hts_df_for_multiselect["hts8"].unique().tolist()[:min(3, len(filtered_hts_df_for_multiselect["hts8"].unique()))],
            format_func=lambda x: f"{x} – {hts_df[hts_df['hts8'] == x]['brief_description'].iloc[0] if not hts_df[hts_df['hts8'] == x].empty else 'N/A'}" 
        )
        if len(selected_hts8_codes) > 10: 
            st.info("Querying many HTS8 codes (and their comparisons) can be slow.")
    elif not hts_df.empty and (search_term or code_prefix):
        st.info("No products found in the HTS data matching your search/filter criteria.")

    if st.button("🚀 Fetch Tariff Data & Comparisons", type="primary", key="fetch_data_button"):
        user_selected_hs6 = list(set(hts_df[hts_df["hts8"].isin(selected_hts8_codes)]["hs6"].tolist())) if selected_hts8_codes and not hts_df.empty else []
        
        if not selected_hts8_codes: st.warning("Please select at least one HTS8 product code.")
        elif not user_selected_hs6: st.warning("No valid HS6 codes derived from your HTS8 selection.")
        elif not reporter_code: st.error("Please select a Reporter Country.")
        elif not year_range_to_send: st.error("Please specify a Year or Year Range.")
        else:
            with st.spinner("Querying WTO API... (This may take a moment)"):
                hs4_prefixes = sorted(list(set([hs6[:4] for hs6 in user_selected_hs6])))
                hs4_comparison_hs6_codes = []
                if hs4_prefixes and not hts_df.empty:
                    for prefix in hs4_prefixes:
                        matches = hts_df[hts_df["hs6"].str.startswith(prefix) & (~hts_df["hs6"].isin(user_selected_hs6))]["hs6"].unique().tolist()
                        hs4_comparison_hs6_codes.extend(matches)
                    hs4_comparison_hs6_codes = sorted(list(set(hs4_comparison_hs6_codes)))[:20]
                
                extracted_keywords_for_search = []
                keyword_comparison_hs6_codes = []
                if selected_hts8_codes and not hts_df.empty:
                    desc_to_search = hts_df[hts_df["hts8"] == selected_hts8_codes[0]]["brief_description"].iloc[0]
                    extracted_keywords_for_search = extract_keywords_nltk(desc_to_search, num_keywords=2, min_len=4) 
                    if extracted_keywords_for_search:
                        matched_hs6_by_keyword = set()
                        for kw in extracted_keywords_for_search:
                            pattern = r'\b' + re.escape(kw) + r'\b' 
                            matched_hs6_by_keyword.update(hts_df[hts_df["searchable_description"].str.contains(pattern, case=False, regex=True, na=False)]["hs6"].unique().tolist())
                        keyword_comparison_hs6_codes = sorted(list(matched_hs6_by_keyword - set(user_selected_hs6) - set(hs4_comparison_hs6_codes)))[:15]

                all_hs6_to_query = sorted(list(set(user_selected_hs6 + hs4_comparison_hs6_codes + keyword_comparison_hs6_codes)))
                if not all_hs6_to_query: 
                    st.error("No HS6 codes identified for querying.")
                    st.stop()

                all_hs6_to_query_str = ",".join(all_hs6_to_query)
                api_calls = [("HS_A_0010", "Applied Rate (%)"), ("HS_A_0020", "Bound Rate (%)"), ("TRF_PREF_APPLIED", "Preferential Rate (%)")]
                fetched_dfs = {} # <-- Add this line to initialize the dictionary

                for indicator, col_name in api_calls:
                    df = fetch_tariff_data(indicator, all_hs6_to_query_str, year_range_to_send, reporter_code, partner_code)
                    if not df.empty: fetched_dfs[col_name] = df.rename(columns={indicator: col_name})

                # Store results in session state
                st.session_state.results_ready = True
                st.session_state.results_data = {
                    "fetched_dfs": fetched_dfs,
                    "user_selected_hs6": user_selected_hs6,
                    "hs4_comparison_hs6_codes": hs4_comparison_hs6_codes,
                    "keyword_comparison_hs6_codes": keyword_comparison_hs6_codes,
                    "extracted_keywords_for_search": extracted_keywords_for_search,
                    # Store query context for display on results page
                    "reporter_name": reporter_name,
                    "partner_name": partner_name,
                    "year_range_to_send": year_range_to_send,
                    "all_hs6_queried_count": len(all_hs6_to_query)
                }
                # Programmatically change the page state and rerun to display the results page
                st.session_state.app_page = "Compare Results"
                st.rerun()
# --- Page 2: Compare Results ---
elif st.session_state.app_page == "Compare Results":
    st.title("📊 Comparison Results")

    if not st.session_state.results_ready:
        st.info("Please perform a search on the 'Search' page first.")
        if st.button("New Search"):
            st.session_state.app_page = "Search"
            st.session_state.results_ready = False # Reset results flag
            st.rerun()
        st.stop()
    
    results_data = st.session_state.results_data
    fetched_dfs = results_data.get("fetched_dfs", {})
    user_selected_hs6 = results_data.get("user_selected_hs6", [])
    hs4_comparison_hs6_codes = results_data.get("hs4_comparison_hs6_codes", [])
    keyword_comparison_hs6_codes = results_data.get("keyword_comparison_hs6_codes", [])
    extracted_keywords_for_search = results_data.get("extracted_keywords_for_search", [])

    if not fetched_dfs:
        st.warning(f"No data was returned from the last WTO API query. Reporter: {results_data.get('reporter_name')}, Partner: {results_data.get('partner_name', 'All')}, Years: {results_data.get('year_range_to_send')}, HS6#: {results_data.get('all_hs6_queried_count')}. Please try different parameters on the Search page.")
    else:
        # Data Merging and Processing
        merged_df_all_data = pd.DataFrame()
        df_list = list(fetched_dfs.values())
        if df_list:
            merged_df_all_data = df_list[0].copy()
            base_keys = ["Reporter", "Reporter Code", "HS6", "Description", "Year"]
            if any("Partner" in df.columns for df in df_list): base_keys.extend(["Partner", "Partner Code"])
            
            for df_add in df_list[1:]:
                on_keys = [k for k in base_keys if k in merged_df_all_data.columns and k in df_add.columns]
                if not all(ek in on_keys for ek in ["HS6", "Year"]): continue
                try: merged_df_all_data = pd.merge(merged_df_all_data, df_add, on=on_keys, how="outer")
                except: continue 
        
        if not hts_df.empty and not merged_df_all_data.empty and "HS6" in merged_df_all_data.columns:
            if all(c in hts_df.columns for c in ["hs6", "hts8", "brief_description"]):
                matches = hts_df[hts_df["hs6"].isin(merged_df_all_data["HS6"].unique())][["hs6", "hts8", "brief_description"]].drop_duplicates(subset=['hs6','hts8'])
                if not matches.empty:
                    merged_df_all_data = pd.merge(merged_df_all_data, matches, left_on="HS6", right_on="hs6", how="left")
                    if 'hs6' in merged_df_all_data.columns: merged_df_all_data = merged_df_all_data.drop(columns="hs6")
                    merged_df_all_data = merged_df_all_data.rename(columns={"brief_description": "HTS8 Local Description", "hts8":"Matching HTS8"})
        
        st.success(f"Displaying results for {len(merged_df_all_data)} total records.")

        if "Applied Rate (%)" in merged_df_all_data.columns and "Bound Rate (%)" in merged_df_all_data.columns:
            app_num = pd.to_numeric(merged_df_all_data["Applied Rate (%)"], errors='coerce')
            bnd_num = pd.to_numeric(merged_df_all_data["Bound Rate (%)"], errors='coerce')
            merged_df_all_data["Overhang (%)"] = bnd_num - app_num
        
        sel_prod_data = merged_df_all_data[merged_df_all_data["HS6"].isin(user_selected_hs6)].copy()
        hs4_data = merged_df_all_data[merged_df_all_data["HS6"].isin(hs4_comparison_hs6_codes)].copy()
        kw_data = merged_df_all_data[merged_df_all_data["HS6"].isin(keyword_comparison_hs6_codes)].copy()
        
        tab_titles = [f"Selected ({len(sel_prod_data)})"]
        if not hs4_data.empty: tab_titles.append(f"HS4 Comp. ({len(hs4_data)})")
        if not kw_data.empty: tab_titles.append(f"Keyword Comp. ({len(kw_data)})")
        tabs = st.tabs(tab_titles)
        
        with tabs[0]: display_product_data_tab(sel_prod_data, "Your Selected Products")
        
        idx_offset = 1
        if not hs4_data.empty and len(tabs) > idx_offset:
            with tabs[idx_offset]: display_product_data_tab(hs4_data, "HS4 Comparison Products"); idx_offset += 1
        if not kw_data.empty and len(tabs) > idx_offset:
            with tabs[idx_offset]: display_product_data_tab(kw_data, "Keyword Comparison Products", 
                                                          f"NLTK keywords: `{', '.join(extracted_keywords_for_search)}`" if extracted_keywords_for_search else None)
        
        st.markdown("---")
        st.subheader("Download Options")
        if not merged_df_all_data.empty:
            csv = merged_df_all_data[merged_df_all_data.columns.tolist()].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Queried Data (CSV)", csv, f"wto_data_comparisons.csv", "text/csv", key="dl_full_data")

# --- Page 3: Reclassification Helper ---
elif st.session_state.app_page == "Reclassification Helper":
    st.title("📝 Reclassification Letter Helper")

    reclass_data = st.session_state.get('reclassification_data', {})
    
    current_hs = reclass_data.get('current_hs', '')
    current_desc = reclass_data.get('current_desc', '')

    st.markdown("This tool helps you draft a letter to request a product reclassification. Fill in the details below.")

    st.text_input("Current HTS Code", value=current_hs, key="current_hs_reclass")
    st.text_area("Current Product Description", value=current_desc, key="current_desc_reclass", height=100)

    st.markdown("---")
    
    st.subheader("Proposed Reclassification")
    proposed_hs = st.text_input("Proposed HTS Code (e.g., 9503.00)", help="Enter the HTS code you believe is correct.")
    
    # Simple search for the proposed HTS code's description
    proposed_desc = ""
    if proposed_hs and not hts_df.empty:
        # Search by 6-digit prefix first
        proposed_hs6 = proposed_hs.replace('.', '')[:6]
        match = hts_df[hts_df['hs6'] == proposed_hs6]
        if not match.empty:
            proposed_desc = match.iloc[0]['brief_description']
    
    st.text_area("Proposed Product Description", value=proposed_desc, key="proposed_desc_reclass", height=100, help="Description from local HTS data will appear here if the code is found.")

    justification = st.text_area("Your Justification", 
                                 height=200, 
                                 placeholder="Explain why the product fits the proposed classification better. For example, focus on its primary material, function, or characteristics as defined by the tariff schedule's chapter notes.")

    if st.button("Generate Letter Draft"):
        if not st.session_state.current_hs_reclass or not proposed_hs or not justification:
            st.error("Please fill in the Current HTS Code, Proposed HTS Code, and your Justification.")
        else:
            letter_template = f"""
[Your Name/Company Name]
[Your Address]
[City, State, Zip]
[Date]

[Recipient Name/Title, e.g., Director, National Commodity Specialist Division]
[Recipient Agency, e.g., U.S. Customs and Border Protection]
[Agency Address]
[City, State, Zip]

**Subject: Request for Reconsideration of HTS Classification for Product**

Dear [Mr./Ms./Director Last Name],

This letter is a formal request for a binding ruling or reconsideration of the Harmonized Tariff Schedule (HTS) classification for the product detailed below.

**Current Classification:**
- **HTS Code:** {st.session_state.current_hs_reclass}
- **Description:** {st.session_state.current_desc_reclass}

**Proposed Classification:**
- **HTS Code:** {proposed_hs}
- **Description:** {st.session_state.proposed_desc_reclass}

**Justification for Reclassification:**

{justification}

We believe that the proposed classification under HTS {proposed_hs} is more appropriate due to the product's primary characteristics and function, in accordance with the General Rules of Interpretation and the relevant Section and Chapter Notes of the HTSUS.

We have attached detailed product specifications, images, and any other relevant documentation for your review. We would be pleased to provide samples upon request.

Thank you for your time and consideration of this matter. We look forward to your response.

Sincerely,

[Your Name]
[Your Title]
"""
            st.subheader("Generated Letter Draft")
            st.info("You can now copy the text below and paste it into your own document for further editing.")
            st.code(letter_template, language=None)

# --- Page 4: News & Overview ---
elif st.session_state.app_page == "News & Overview":
    st.title("📰 News & Overview")
    st.info("This section is under development and will feature news articles and official announcements related to tariff changes and trade policy, as per the product requirements document.")
    st.markdown("Future features will include:")
    st.markdown("- Real-time news feeds from sources like the WTO.")
    st.markdown("- Curated announcements on major trade policy shifts.")
