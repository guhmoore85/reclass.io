import streamlit as st
import pandas as pd
import requests
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

# --- START: NLTK Resource Downloader ---
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
# --- END: NLTK Resource Downloader ---

st.set_page_config(page_title="WTO Tariff Lookup", layout="wide")

# IMPROVEMENT: Use Streamlit Secrets for your API Key in deployment
try:
    API_KEY = st.secrets["API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local development if secrets aren't set
    API_KEY = "ab5ad8703cd54ffba080cb9554175101"

# --- Initialize NLTK components ---
nltk_stopwords_english = set(nltk_stopwords.words('english'))
custom_stop_words_for_extraction = set([
    "parts", "thereof", "subsidiary", "chapter", "whether", "other", "nesoi",
    "product", "hs", "code", "description", "including", "etc"
])
STOP_WORDS_FOR_EXTRACTION = nltk_stopwords_english.union(custom_stop_words_for_extraction)
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---
def lemmatize_string_for_search(text_to_lemmatize):
    """Lemmatizes a string for better search matching."""
    if not isinstance(text_to_lemmatize, str) or not text_to_lemmatize.strip(): 
        return ""
    words = word_tokenize(text_to_lemmatize.lower())
    lemmatized_list = [lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='v'), pos='n') for word in words if word.isalpha()]
    return " ".join(lemmatized_list)

def create_pivot_table(df, value_columns_list):
    """Create a pivot table with HS6 codes and Description as rows."""
    if df.empty or not value_columns_list or "Year" not in df.columns or "HS6" not in df.columns:
        return pd.DataFrame()
    df_copy = df.copy()
    df_copy["Year"] = df_copy["Year"].fillna("Unknown_Year").astype(str)
    valid_value_columns = [col for col in value_columns_list if col in df_copy.columns]
    if not valid_value_columns: 
        return pd.DataFrame()
    for col in valid_value_columns: 
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    desc_col_name = "Pivot Description"
    if "HTS8 Local Description" in df_copy.columns and df_copy["HTS8 Local Description"].notna().any():
        df_copy[desc_col_name] = df_copy["HTS8 Local Description"].fillna("N/A")
    elif "Description" in df_copy.columns:
        df_copy[desc_col_name] = df_copy["Description"].fillna("N/A")
    else:
        df_copy[desc_col_name] = "N/A"

    df_copy.dropna(subset=["HS6", desc_col_name, "Year"], inplace=True)
    df_copy.dropna(subset=valid_value_columns, how='all', inplace=True)
    if df_copy.empty: 
        return pd.DataFrame()

    try:
        pivot = df_copy.pivot_table(index=["HS6", desc_col_name], columns="Year", values=valid_value_columns, aggfunc="first")
        if pivot.empty: 
            return pd.DataFrame()
        if isinstance(pivot.columns, pd.MultiIndex) and len(pivot.columns.levels) > 1:
            try:
                year_level_values = pivot.columns.levels[1].astype(str)
                sorted_years = sorted(year_level_values, key=lambda x: (int(x) if x.isdigit() else float('inf'), x))
                new_column_order = [(metric, year) for metric in valid_value_columns for year in sorted_years if (metric, year) in pivot.columns]
                if new_column_order: 
                    pivot = pivot[new_column_order]
            except: 
                pass
        return pivot.reset_index().rename(columns={desc_col_name: "Description (Local or API)"})
    except Exception: 
        return pd.DataFrame()

@st.cache_data
def load_hts_data(file_path="hts8.csv"):
    try:
        df = pd.read_csv(file_path, dtype={'hts8': str})
        if "hts8" not in df.columns or "brief_description" not in df.columns:
            st.error(f"HTS data file ('{file_path}') is missing required columns 'hts8' or 'brief_description'.")
            return pd.DataFrame()
        df["hts8"] = df["hts8"].astype(str).str.strip().str.zfill(8)
        df["hs6"] = df["hts8"].str[:6]
        df["brief_description"] = df["brief_description"].astype(str)
        df = df.drop_duplicates(subset=["hts8", "brief_description"])
        df['searchable_description'] = df['brief_description'].apply(lemmatize_string_for_search)
        return df
    except FileNotFoundError:
        st.error(f"\u26a0\ufe0f HTS data file ('{file_path}') not found. Please upload it to your GitHub repository.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading HTS data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_country_list():
    fallback_countries = [
        {"name": "Canada", "code": "124"}, 
        {"name": "China", "code": "156"}, 
        {"name": "European Union", "code": "918"}, 
        {"name": "Mexico", "code": "484"}, 
        {"name": "United States of America", "code": "840"}
    ]
    try:
        response = requests.get("https://api.wto.org/timeseries/v1/reporters", headers={"Ocp-Apim-Subscription-Key": API_KEY}, timeout=10)
        if response.status_code == 200 and isinstance(response.json(), list):
            countries = [{"name": str(c.get("name","")).strip(), "code": str(c.get("code","")).strip()} for c in response.json() if c.get("name") and c.get("code")]
            unique_countries = {c['name']: c for c in countries}
            # FIX: Correctly sort the list of dictionaries by name
            return sorted(list(unique_countries.values()), key=lambda x: x['name'])
        return sorted(fallback_countries, key=lambda x: x['name'])
    except Exception:
        return sorted(fallback_countries, key=lambda x: x['name'])

@st.cache_data
def fetch_tariff_data(indicator_code, hs6_list_str, year_range, reporter_code, partner_code=None):
    if not hs6_list_str: 
        return pd.DataFrame()
    params = {"i": indicator_code, "pc": hs6_list_str, "ps": year_range, "r": reporter_code, "subscription-key": API_KEY}
    if partner_code: 
        params["p"] = partner_code
    try:
        response = requests.get("https://api.wto.org/timeseries/v1/data", params=params, timeout=60)
        response.raise_for_status() # Will raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if "Dataset" in data and data["Dataset"]:
            records = [{"Reporter": item.get("ReportingEconomy"), "Reporter Code": item.get("ReportingEconomyCode"), "Partner": item.get("PartnerEconomy"), "Partner Code": item.get("PartnerEconomyCode"), "HS6": item.get("ProductOrSectorCode"), "Description": item.get("ProductOrSector"), "Year": item.get("TimeDimensionValue"), indicator_code: item.get("Value")} for item in data["Dataset"]]
            return pd.DataFrame(records)
        return pd.DataFrame()
    # FIX: Make API errors visible to the user instead of failing silently.
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}", icon="🚨")
        return pd.DataFrame()
    except requests.exceptions.Timeout:
        st.error("API Error: The request timed out. The WTO server might be busy. Please try again later.", icon="⏱️")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="⚙️")
        return pd.DataFrame()

def extract_keywords_nltk(text, num_keywords=3, min_len=4):
    if not isinstance(text, str) or not text.strip(): 
        return []
    words = word_tokenize(text.lower())
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
    st.subheader(f"Data for {tab_title_suffix}")
    if search_context_info: 
        st.caption(search_context_info)
    with st.expander("Reclassification Helper"):
        st.write("Enter an HS6 code from the raw data table below and click the button to start a reclassification request.")
        selected_hs_for_reclass = st.text_input("Enter HS6 code:", key=f"reclass_hs_{tab_title_suffix.replace(' ', '_')}")
        if st.button(f"Start Reclassification for {selected_hs_for_reclass}", key=f"reclass_btn_{tab_title_suffix.replace(' ', '_')}"):
            if selected_hs_for_reclass and not data_df[data_df['HS6'] == selected_hs_for_reclass].empty:
                product_row = data_df[data_df['HS6'] == selected_hs_for_reclass].iloc[0]
                st.session_state.reclassification_data = {"current_hs": product_row['HS6'], "current_desc": product_row.get("HTS8 Local Description") or product_row.get("Description") or "N/A"}
                st.session_state.app_page = "Reclassification Helper"
                st.rerun()
            else:
                st.warning("Please enter a valid HS6 code found in this table.")
    st.markdown("---")
    if not data_df.empty:
        rate_metrics_to_pivot = [col for col in ["Applied Rate (%)", "Bound Rate (%)", "Preferential Rate (%)", "Overhang (%)"] if col in data_df.columns]
        if rate_metrics_to_pivot:
            help_text = "- **Applied Rate:** The standard rate actually charged on imports.\n- **Bound Rate:** The maximum rate a country has committed to at the WTO.\n- **Preferential Rate:** A lower rate (often 0%) available under a trade agreement.\n- **Overhang:** The difference between the Bound and Applied rates."
            st.write(f"**Tariff Rates & Overhang:**", help=help_text)
            pivot_rates_df = create_pivot_table(data_df.copy(), rate_metrics_to_pivot)
            if not pivot_rates_df.empty:
                s = pivot_rates_df.style
                col_map = {"Applied Rate (%)": "Blues", "Bound Rate (%)": "Greens", "Preferential Rate (%)": "Purples", "Overhang (%)": "coolwarm"}
                for metric, cmap in col_map.items():
                    if isinstance(pivot_rates_df.columns, pd.MultiIndex) and metric in pivot_rates_df.columns.get_level_values(0):
                        s = s.background_gradient(cmap=cmap, subset=[col for col in pivot_rates_df.columns if col[0] == metric])
                st.dataframe(s, use_container_width=True)
        with st.expander(f"View Raw Data for {tab_title_suffix}", expanded=False):
            st.dataframe(data_df)
    else:
        st.info(f"No API data found for {tab_title_suffix.lower()}.")

# --- Main App Logic ---
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
page_options = ["Search", "Compare Results", "Reclassification Helper", "News & Overview"]
try: 
    current_page_index = page_options.index(st.session_state.app_page)
except ValueError: 
    current_page_index = 0
st.sidebar.radio("Go to", page_options, key="sidebar_radio_key", on_change=lambda: setattr(st.session_state, 'app_page', st.session_state.sidebar_radio_key), index=current_page_index)
st.sidebar.markdown("---")

# IMPROVEMENT: Load data once at the top level, not inside the page logic.
hts_df = load_hts_data()
countries = load_country_list()

# --- Page Routing ---
if st.session_state.app_page == "Search":
    st.title("🔎 Search & Compare WTO Tariffs")
    st.subheader("Use the filters in the sidebar to find and select products.")
    st.sidebar.header("🔍 Query Filters")

    # IMPROVEMENT: Create a curated country list for better UX.
    PREFERRED_REPORTERS = ["United States of America", "European Union", "China", "Mexico", "Canada", "Japan", "Korea, Republic of", "United Kingdom"]
    preferred_list = [c for c in countries if c["name"] in PREFERRED_REPORTERS]
    other_list = [c for c in countries if c["name"] not in PREFERRED_REPORTERS]
    separator = "──────────────────"
    country_names = [c['name'] for c in preferred_list] + ([separator] if other_list else []) + [c['name'] for c in other_list]

    # --- Filter Widgets ---
    search_term = st.sidebar.text_input("Search Product Description", help="e.g., motorcycles, steel pipes, rice")
    code_prefix = st.sidebar.text_input("Filter by HTS or HS Code Prefix", help="e.g., 1006, 7210")

    current_year = pd.Timestamp.now().year
    # FIX: Default year range is now safer and less likely to find no data.
    default_start_year = current_year - 5
    default_end_year = current_year - 3
    year_range_input = st.sidebar.text_input("Year Range (e.g., 2018-2022)", value=f"{default_start_year}-{default_end_year}", help=f"Single year or range. Latest available data is often 2-3 years old.")
    year_range_to_send = year_range_input.strip()

    reporter_name = st.sidebar.selectbox("Select Reporter Country", country_names, index=country_names.index("United States of America") if "United States of America" in country_names else 0)
    if reporter_name == separator:
        st.sidebar.warning("Please select a valid country.")
        st.stop()
    reporter_code = next((c["code"] for c in countries if c["name"] == reporter_name), None)

    partner_options = ["All (Worldwide)"] + [c for c in country_names if c != separator]
    partner_name = st.sidebar.selectbox("Select Partner Country (Optional)", partner_options, index=0)
    partner_code = next((c["code"] for c in countries if c["name"] == partner_name), None) if partner_name != "All (Worldwide)" else None

    # --- Product Selection ---
    st.header("📍 Product Selection")
    if not hts_df.empty:
        filtered_hts = hts_df.copy()
        if search_term:
            query_parts = [f"`searchable_description`.str.contains(r'\\b{re.escape(kw)}\\b', case=False, na=False)" for kw in lemmatize_string_for_search(search_term).split() if kw]
            if query_parts: 
                filtered_hts = filtered_hts.query(" and ".join(query_parts))
        if code_prefix:
            filtered_hts = filtered_hts[filtered_hts["hts8"].str.startswith(code_prefix) | filtered_hts["hs6"].str.startswith(code_prefix)]

        if not filtered_hts.empty:
            options = filtered_hts["hts8"].unique().tolist()
            selected_hts8_codes = st.multiselect("Select HTS8 Codes (max 10 recommended)", options=options, default=options[:min(3, len(options))], format_func=lambda x: f"{x} – {hts_df.loc[hts_df['hts8'] == x, 'brief_description'].iloc[0]}")
        else:
            st.info("No products found in the HTS data matching your search/filter criteria.")
            selected_hts8_codes = []
    else:
        selected_hts8_codes = []
        st.warning("HTS data file not loaded. Product search is disabled.")

    # --- Fetch Button Logic ---
    if st.button("🚀 Fetch Tariff Data & Comparisons", type="primary"):
        if not selected_hts8_codes:
            st.warning("Please select at least one HTS8 product code.")
        elif not reporter_code:
            st.error("Please select a Reporter Country.")
        else:
            with st.spinner("Querying WTO API... (This may take a moment)"):
                user_selected_hs6 = hts_df[hts_df["hts8"].isin(selected_hts8_codes)]["hs6"].unique().tolist()
                hs4_prefixes = tuple(set(hs6[:4] for hs6 in user_selected_hs6))
                hs4_comp_codes = hts_df[hts_df["hs6"].str.startswith(hs4_prefixes) & (~hts_df["hs6"].isin(user_selected_hs6))]["hs6"].unique().tolist()[:20]

                desc_to_search = hts_df.loc[hts_df["hts8"] == selected_hts8_codes[0], "brief_description"].iloc[0]
                keywords = extract_keywords_nltk(desc_to_search, num_keywords=2)
                kw_comp_codes = []
                if keywords:
                    kw_query = ' or '.join([f"`searchable_description`.str.contains(r'\\b{re.escape(kw)}\\b', case=False)" for kw in keywords])
                    kw_matches = hts_df.query(kw_query)
                    kw_comp_codes = kw_matches[~kw_matches["hs6"].isin(user_selected_hs6 + hs4_comp_codes)]["hs6"].unique().tolist()[:15]
                
                all_hs6_to_query = sorted(list(set(user_selected_hs6 + hs4_comp_codes + kw_comp_codes)))
                if not all_hs6_to_query:
                    st.error("No HS6 codes identified for querying.")
                    st.stop()

                api_calls = [("HS_A_0010", "Applied Rate (%)"), ("HS_A_0020", "Bound Rate (%)"), ("TRF_PREF_APPLIED", "Preferential Rate (%)")]
                fetched_dfs = {col: fetch_tariff_data(ind, ",".join(all_hs6_to_query), year_range_to_send, reporter_code, partner_code).rename(columns={ind: col}) for ind, col in api_calls}
                fetched_dfs = {k: v for k, v in fetched_dfs.items() if not v.empty}

                st.session_state.results_ready = True
                st.session_state.results_data = {
                    "fetched_dfs": fetched_dfs,
                    "user_selected_hs6": user_selected_hs6,
                    "hs4_comparison_hs6_codes": hs4_comp_codes,
                    "keyword_comparison_hs6_codes": kw_comp_codes,
                    "extracted_keywords_for_search": keywords,
                    "reporter_name": reporter_name,
                    "partner_name": partner_name,
                    "year_range_to_send": year_range_to_send,
                    "all_hs6_queried_count": len(all_hs6_to_query)
                }
                st.session_state.app_page = "Compare Results"
                st.rerun()

elif st.session_state.app_page == "Compare Results":
    st.title("📊 Comparison Results")
    if not st.session_state.results_ready:
        st.info("Please perform a search on the 'Search' page first.")
        if st.button("New Search"):
            st.session_state.app_page = "Search"
            st.session_state.results_ready = False
            st.rerun()
        st.stop()
    
    data = st.session_state.results_data
    if not data.get("fetched_dfs"):
        st.warning(f"No data was returned from the last WTO API query. Reporter: {data.get('reporter_name')}, Partner: {data.get('partner_name', 'All')}, Years: {data.get('year_range_to_send')}, HS6 Codes Queried: {data.get('all_hs6_queried_count')}.")
        if st.button("New Search"):
            st.session_state.app_page = "Search"
            st.session_state.results_ready = False
            st.rerun()
    else:
        df_list = list(data["fetched_dfs"].values())
        merged_df = df_list[0]
        if len(df_list) > 1:
            merge_keys = ["Reporter", "Reporter Code", "HS6", "Description", "Year", "Partner", "Partner Code"]
            for df_add in df_list[1:]:
                on_keys = [k for k in merge_keys if k in merged_df.columns and k in df_add.columns]
                if not on_keys: 
                    continue
                merged_df = pd.merge(merged_df, df_add, on=on_keys, how="outer")
        
        if not hts_df.empty:
            matches = hts_df[hts_df["hs6"].isin(merged_df["HS6"].unique())][["hs6", "hts8", "brief_description"]].drop_duplicates(subset=['hs6', 'hts8'])
            if not matches.empty:
                merged_df = pd.merge(merged_df, matches, left_on="HS6", right_on="hs6", how="left").drop(columns="hs6").rename(columns={"brief_description": "HTS8 Local Description", "hts8": "Matching HTS8"})
        
        st.success(f"Displaying results for {len(merged_df)} total records.")
        if "Applied Rate (%)" in merged_df.columns and "Bound Rate (%)" in merged_df.columns:
            merged_df["Overhang (%)"] = pd.to_numeric(merged_df["Bound Rate (%)"], errors='coerce') - pd.to_numeric(merged_df["Applied Rate (%)"], errors='coerce')

        sel_data = merged_df[merged_df["HS6"].isin(data["user_selected_hs6"])].copy()
        hs4_data = merged_df[merged_df["HS6"].isin(data["hs4_comparison_hs6_codes"])].copy()
        kw_data = merged_df[merged_df["HS6"].isin(data["keyword_comparison_hs6_codes"])].copy()
        
        tab_titles = [f"Selected ({len(sel_data)})"]
        if not hs4_data.empty: 
            tab_titles.append(f"HS4 Comp. ({len(hs4_data)})")
        if not kw_data.empty: 
            tab_titles.append(f"Keyword Comp. ({len(kw_data)})")
        tabs = st.tabs(tab_titles)
        with tabs[0]: 
            display_product_data_tab(sel_data, "Your Selected Products")
        if not hs4_data.empty and len(tabs) > 1:
            with tabs[1]: 
                display_product_data_tab(hs4_data, "HS4 Comparison Products")
        if not kw_data.empty and len(tabs) > 2:
            with tabs[2]: 
                display_product_data_tab(kw_data, "Keyword Comparison Products", f"NLTK keywords: `{', '.join(data['extracted_keywords_for_search'])}`")
        
        st.markdown("---")
        st.subheader("Download Options")
        csv = merged_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Queried Data (CSV)", csv, "wto_data_comparisons.csv", "text/csv")

elif st.session_state.app_page == "Reclassification Helper":
    st.title("📝 Reclassification Letter Helper")
    data = st.session_state.get('reclassification_data', {})
    st.text_input("Current HTS Code", value=data.get('current_hs', ''), key="current_hs_reclass")
    st.text_area("Current Product Description", value=data.get('current_desc', ''), key="current_desc_reclass", height=100)
    st.markdown("---")
    st.subheader("Proposed Reclassification")
    proposed_hs = st.text_input("Proposed HTS Code")
    match = hts_df[hts_df['hts8'].str.startswith(proposed_hs.replace('.', '')[:8])] if proposed_hs and not hts_df.empty else pd.DataFrame()
    st.text_area("Proposed Product Description", value=match.iloc[0]['brief_description'] if not match.empty else "", key="proposed_desc_reclass", height=100)
    justification = st.text_area("Your Justification", height=200, placeholder="Explain why the product fits the proposed classification better...")
    if st.button("Generate Letter Draft"):
        if not all([st.session_state.current_hs_reclass, proposed_hs, justification]):
            st.error("Please fill in all fields.")
        else:
            letter = f"""
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
            st.code(letter)

elif st.session_state.app_page == "News & Overview":
    st.title("📰 News & Overview")
    st.info("This section is under development.")
