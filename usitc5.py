import streamlit as st
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK and Data Setup ---

# One-time setup for NLTK resources.
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading necessary NLTK resources...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
    st.success("NLTK resources downloaded. The app should now run correctly.")

# Initialize NLTK components and define custom stop words for better filtering
lemmatizer = WordNetLemmatizer()
custom_stop_words = {'nesoi', 'thereof', 'other', 'whether', 'textile', 'materials', 'articles'}
stop_words = set(nltk_stopwords.words('english')).union(custom_stop_words)

@st.cache_data
def load_and_prepare_hts_data(filepath: str) -> pd.DataFrame:
    """Loads HTS codes from a CSV and prepares them for searching."""
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(subset=['hts8', 'brief_description'])
        df = df.drop_duplicates(subset=['hts8'], keep='first')
        df['hts8'] = df['hts8'].astype(str).str.zfill(8)
        df['hs6'] = df['hts8'].str[:6]

        def lemmatize_text(text):
            tokens = re.findall(r'\b\w+\b', text.lower())
            lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
            return ' '.join(lemmatized_tokens)

        df['lemmatized_desc'] = df['brief_description'].apply(lemmatize_text)
        df = df.reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in the same directory as the app.")
        return pd.DataFrame()

@st.cache_data
def load_rulings_data(filepath: str) -> pd.DataFrame:
    """Loads and prepares the cross-rulings data from a CSV."""
    try:
        df = pd.read_csv(filepath)
        df = df.rename(columns={"Found_HTS8": "hts8"})
        df = df.dropna(subset=['hts8', 'RULING', 'RULING_DATE'])
        df['hts8'] = df['hts8'].astype(str).str.replace(r'\D', '', regex=True).str.zfill(8)
        df = df.drop_duplicates(subset=['hts8', 'RULING'])
        return df
    except FileNotFoundError:
        st.error(f"Error: The rulings file '{filepath}' was not found. Please make sure it's in the same directory as the app.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the rulings file: {e}")
        return pd.DataFrame()

# --- Recommendation Engine Setup ---
@st.cache_resource
def create_tfidf_matrix(_df: pd.DataFrame):
    """Creates and caches the TF-IDF vectorizer and matrix."""
    if _df.empty or 'lemmatized_desc' not in _df.columns:
        return None, None
    # FIX: Tune vectorizer for better performance with specialized vocabulary
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), min_df=1, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(_df['lemmatized_desc'])
    return vectorizer, tfidf_matrix

def get_hierarchical_recommendations(hts_code: str, _df: pd.DataFrame, num_recs: int = 5) -> pd.DataFrame:
    """Finds other products with the same 6-digit HTS prefix."""
    if _df.empty or len(hts_code) < 6:
        return pd.DataFrame()
    prefix = hts_code[:6]
    recommendations = _df[(_df['hs6'] == prefix) & (_df['hts8'] != hts_code)]
    return recommendations[['hts8', 'brief_description']].head(num_recs)

def get_semantic_recommendations(hts_code: str, _df: pd.DataFrame, _tfidf_matrix, num_recs: int = 5) -> pd.DataFrame:
    """Finds products with the most similar descriptions using TF-IDF."""
    if _df.empty or _tfidf_matrix is None:
        return pd.DataFrame()
    try:
        idx = _df.index[_df['hts8'] == hts_code].tolist()[0]
        cosine_similarities = cosine_similarity(_tfidf_matrix[idx:idx+1], _tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-num_recs-2:-1]
        recommendations = [i for i in related_docs_indices if i != idx]
        return _df.iloc[recommendations][['hts8', 'brief_description']].head(num_recs)
    except (IndexError, TypeError):
        return pd.DataFrame()

# --- Configuration & Styling ---
st.set_page_config(page_title="Tariff Data Explorer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>""", unsafe_allow_html=True)

# --- Core API Logic ---
@st.cache_data(ttl=3600)
def get_tariff_data(api_token: str, hts_code: str, year: str) -> Optional[Dict[str, Any]]:
    """Queries the USITC DataWeb Tariff API (v2) for detailed tariff info."""
    base_url = "https://datawebws.usitc.gov/dataweb/api/v2/tariff/currentTariffDetails"
    params = {"year": year, "hts8": hts_code}
    headers = {"Authorization": f"Bearer {api_token}", "Accept": "application/json", "User-Agent": "Mozilla/5.0 (Streamlit Tariff App v1.0)"}
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        if response.status_code != 404: st.error(f"HTTP Error for {hts_code} in {year}: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"A critical network error occurred: {req_err}")
    return None

# --- Data Parsing & Display Functions ---
def parse_key_rates(data: Dict[str, Any]) -> pd.DataFrame:
    """Parses the main tariff rates (Applied vs Bound) from the JSON."""
    rates = []
    try:
        treatment_section = next(s for s in data.get('sections', []) if s.get('id') == 'tariff_treatment')
        mfn_section = next(c for c in treatment_section.get('children', []) if c.get('id') == 'mfn')
        mfn_rate = next(r.get('value') for r in mfn_section.get('children', []) if r.get('id') == 'mfn_text')
        rates.append({"Rate Type": "General (Applied MFN)", "Rate": mfn_rate or "N/A"})
        col2_section = next(c for c in treatment_section.get('children', []) if c.get('id') == 'col2')
        col2_rate = next(r.get('value') for r in col2_section.get('children', []) if r.get('id') == 'col2_text_rate')
        rates.append({"Rate Type": "Column 2 (Bound)", "Rate": col2_rate or "N/A"})
    except (StopIteration, TypeError):
        st.warning("Could not parse key rate data.")
    return pd.DataFrame(rates)

def parse_rate_for_graph(data: Dict[str, Any]) -> Optional[float]:
    """Parses the MFN rate into a single float value for graphing."""
    try:
        treatment_section = next(s for s in data.get('sections', []) if s.get('id') == 'tariff_treatment')
        mfn_section = next(c for c in treatment_section.get('children', []) if c.get('id') == 'mfn')
        adv_rate_str = next((r.get('value') for r in mfn_section.get('children', []) if r.get('id') == 'adv_rate_comp'), "0%")
        if adv_rate_str and '%' in adv_rate_str:
            return float(adv_rate_str.replace('%', '').strip())
        mfn_rate_text = next((r.get('value') for r in mfn_section.get('children', []) if r.get('id') == 'mfn_text'), "")
        if mfn_rate_text and "free" in mfn_rate_text.lower(): return 0.0
        if isinstance(mfn_rate_text, str):
            numbers = re.findall(r"(\d+\.?\d*)", mfn_rate_text)
            if numbers: return float(numbers[0])
    except (StopIteration, TypeError, ValueError):
        return None
    return None

def parse_rate_for_calculation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parses the MFN rate components for duty calculation."""
    calculation_details = {"ad_valorem_percent": 0.0, "specific_rate_usd": 0.0, "unit": "N/A", "error": None}
    try:
        treatment_section = next(s for s in data.get('sections', []) if s.get('id') == 'tariff_treatment')
        mfn_section = next(c for c in treatment_section.get('children', []) if c.get('id') == 'mfn')
        calculation_details['unit'] = next((c.get('value') for c in treatment_section.get('children', []) if c.get('id') == 'uoq1'), "N/A")
        adv_rate_str = next((r.get('value') for r in mfn_section.get('children', []) if r.get('id') == 'adv_rate_comp'), "0%")
        if adv_rate_str and '%' in adv_rate_str:
            calculation_details['ad_valorem_percent'] = float(adv_rate_str.replace('%', '').strip()) / 100.0
        spec_rate_str = next((r.get('value') for r in mfn_section.get('children', []) if r.get('id') == 'spec_rate_comp'), "$0.0")
        if spec_rate_str and '$' in spec_rate_str:
            calculation_details['specific_rate_usd'] = float(spec_rate_str.replace('$', '').strip())
    except (StopIteration, TypeError, ValueError) as e:
        calculation_details['error'] = f"Could not parse rate components for calculation: {e}"
    return calculation_details

def parse_trade_agreements(data: Dict[str, Any]) -> pd.DataFrame:
    """Parses the preferential trade agreement statuses from the JSON."""
    agreements = []
    try:
        program_section = next(s for s in data.get('sections', []) if s.get('id') == 'tariff_program')
        for child in program_section.get('children', []):
            program_name = child.get('id', 'N/A').replace('_', ' ').title()
            status = child.get('value')
            if status is None and child.get('children'):
                 status_child = next((sc for sc in child.get('children') if sc.get('id') == 'status'), None)
                 if status_child: status = status_child.get('value')
            agreements.append({"Trade Program": program_name, "Eligibility": status or "N/A"})
    except (StopIteration, TypeError):
        st.warning("Could not parse trade agreement data.")
    return pd.DataFrame(agreements)

# --- Streamlit App UI ---
st.title("📊 Tariff Data Explorer")
st.write("An MVP application to search and compare U.S. tariff data, based on the USITC API.")

try:
    API_TOKEN = st.secrets["usitc_api_token"]
except FileNotFoundError:
    st.error("API token not found. Please add it to your Streamlit secrets.")
    st.info("Create a file `.streamlit/secrets.toml` with the content:\n`usitc_api_token = \"YOUR_TOKEN_HERE\"`")
    st.stop()

# Load local data and create recommendation engine components
hts_df = load_and_prepare_hts_data('hts8.csv')
rulings_df = load_rulings_data('reconciled_rulings_with_all_hts8.csv')
vectorizer, tfidf_matrix = create_tfidf_matrix(hts_df)

page = st.sidebar.radio("Navigation", ["Search", "Compare", "Reclassification Helper", "News & Overview"])

if page == "Search":
    st.header("Search for a Product Tariff")
    if hts_df.empty: st.stop()

    search_type = st.radio("Search Type", ["By HTS Code", "By Keyword"], horizontal=True)

    if search_type == "By HTS Code":
        hts_input = st.text_input("Enter HTS Code (e.g., 08044000 for Avocados)", "08044000")
    else:
        keyword_input = st.text_input("Enter a keyword (e.g., coffee, computer)", "avocado")
        if keyword_input:
            lemmatized_query = ' '.join([lemmatizer.lemmatize(w) for w in re.findall(r'\b\w+\b', keyword_input.lower()) if w.isalpha()])
            results = hts_df[hts_df['lemmatized_desc'].str.contains(lemmatized_query, na=False)]
            if not results.empty:
                product_options = {f"{row['hts8']} - {row['brief_description']}": row['hts8'] for index, row in results.iterrows()}
                selected_product = st.selectbox("Select a product from the results", options=product_options.keys())
                hts_input = product_options.get(selected_product)
            else:
                st.warning("No products found for that keyword.")
                hts_input = None
    
    st.subheader("Select Time Range for Historical Graph")
    col1, col2 = st.columns(2)
    with col1: start_year = st.number_input("Start Year", 2000, 2025, 2020)
    with col2: end_year = st.number_input("End Year", 2000, 2025, 2023)

    if st.button("Get Tariff Details & History"):
        if not hts_input: st.warning("Please enter an HTS Code or select a product from a keyword search.")
        elif start_year > end_year: st.warning("Start Year cannot be after End Year.")
        else:
            with st.spinner(f"Fetching latest data for HTS {hts_input}..."):
                latest_tariff_data = get_tariff_data(API_TOKEN, hts_input, str(end_year))
            
            if latest_tariff_data:
                st.session_state.latest_tariff_data = latest_tariff_data
                st.success(f"Displaying details for HTS Code: **{hts_input}** in **{end_year}**")
                desc = latest_tariff_data.get('desc')
                if desc: st.subheader(f"Description: {desc}")
                
                if not rulings_df.empty:
                    matching_rulings = rulings_df[rulings_df['hts8'] == hts_input]
                    if not matching_rulings.empty:
                        st.markdown("#### 📝 Associated Rulings")
                        st.dataframe(matching_rulings[['RULING', 'RULING_DATE']], use_container_width=True)

                st.markdown("#### Key Tariff Rates")
                st.table(parse_key_rates(latest_tariff_data))
                st.markdown("#### Preferential Trade Agreement Status")
                agreements_df = parse_trade_agreements(latest_tariff_data)
                if not agreements_df.empty: st.dataframe(agreements_df, use_container_width=True)
                else: st.info("No specific trade agreement data found.")
            else:
                st.error(f"Failed to retrieve data for {end_year}.")
                st.session_state.latest_tariff_data = None

            st.markdown("#### Historical MFN Rate (Ad Valorem %)")
            with st.spinner(f"Fetching historical data from {start_year} to {end_year}..."):
                historical_rates = []
                for year in range(start_year, end_year + 1):
                    data = get_tariff_data(API_TOKEN, hts_input, str(year))
                    if data:
                        rate = parse_rate_for_graph(data)
                        if rate is not None: historical_rates.append({"Year": str(year), "Rate (%)": rate})
            if historical_rates:
                history_df = pd.DataFrame(historical_rates).set_index("Year")
                st.line_chart(history_df)
            else: st.info("No historical rate data could be found.")

            # --- NEW: Similar Products Section ---
            st.markdown("---")
            st.markdown("#### 💡 Similar Product Recommendations")
            
            hier_recs = get_hierarchical_recommendations(hts_input, hts_df)
            sem_recs = get_semantic_recommendations(hts_input, hts_df, tfidf_matrix)

            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                st.markdown("**Other Products in this Subheading**")
                if not hier_recs.empty: st.table(hier_recs)
                else: st.info("No other products found with the same 6-digit prefix.")
            
            with col_rec2:
                st.markdown("**Semantically Similar Products**")
                if not sem_recs.empty: st.table(sem_recs)
                else: st.info("No semantically similar products found.")

    # --- Duty Calculator Section ---
    if 'latest_tariff_data' in st.session_state and st.session_state.latest_tariff_data:
        st.markdown("---")
        st.markdown("#### 🧮 Duty Calculator (Estimate)")
        rate_components = parse_rate_for_calculation(st.session_state.latest_tariff_data)
        if rate_components.get("error"): st.warning(rate_components["error"])
        else:
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1: shipment_value = st.number_input("Shipment Value (USD)", 0.0, value=1000.0, step=100.0)
            shipment_quantity = 0
            if rate_components.get("specific_rate_usd", 0) > 0:
                 with col_calc2: shipment_quantity = st.number_input(f"Shipment Quantity ({rate_components.get('unit', 'units')})", 0.0, value=100.0, step=10.0)
            if st.button("Calculate Duty"):
                ad_valorem_duty = shipment_value * rate_components.get("ad_valorem_percent", 0)
                specific_duty = shipment_quantity * rate_components.get("specific_rate_usd", 0)
                total_duty = ad_valorem_duty + specific_duty
                st.subheader("Estimated Duty Calculation:")
                st.metric("Ad Valorem Duty", f"${ad_valorem_duty:,.2f}")
                if specific_duty > 0: st.metric("Specific Duty", f"${specific_duty:,.2f}")
                st.metric("Total Estimated Duty", f"${total_duty:,.2f}")

elif page == "Compare":
    st.header("Compare Product Tariffs")
    if 'comparison_results' not in st.session_state: st.session_state.comparison_results = None
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: hts1 = st.text_input("Product 1 HTS Code", "08044000")
    with col2: hts2 = st.text_input("Product 2 HTS Code", "09012100")
    with col3: compare_year = st.text_input("Year", "2023")
    if st.button("Compare"):
        with st.spinner("Fetching data for comparison..."):
            data1, data2 = get_tariff_data(API_TOKEN, hts1, compare_year), get_tariff_data(API_TOKEN, hts2, compare_year)
        if data1 and data2:
            st.success("Comparison data loaded successfully.")
            st.session_state.comparison_results = {"data1": data1, "data2": data2, "hts1": hts1, "hts2": hts2}
        else:
            st.error("Could not fetch data for one or both products.")
            st.session_state.comparison_results = None
    if st.session_state.comparison_results:
        res = st.session_state.comparison_results
        desc1, desc2 = res['data1'].get('desc', 'N/A'), res['data2'].get('desc', 'N/A')
        rates1, rates2 = parse_key_rates(res['data1']), parse_key_rates(res['data2'])
        mfn1, mfn2 = rates1[rates1['Rate Type'] == 'General (Applied MFN)']['Rate'].values[0], rates2[rates2['Rate Type'] == 'General (Applied MFN)']['Rate'].values[0]
        col2_rate1, col2_rate2 = rates1[rates1['Rate Type'] == 'Column 2 (Bound)']['Rate'].values[0], rates2[rates2['Rate Type'] == 'Column 2 (Bound)']['Rate'].values[0]
        comparison_df = pd.DataFrame({"Feature": ["Description", "Applied MFN Rate", "Bound Rate (Col. 2)"], f"Product 1 ({res['hts1']})": [desc1, mfn1, col2_rate1], f"Product 2 ({res['hts2']})": [desc2, mfn2, col2_rate2]})
        st.table(comparison_df.set_index('Feature'))
        st.markdown("---")
        st.markdown("#### 🧮 Duty Comparison Calculator")
        rate_comp1, rate_comp2 = parse_rate_for_calculation(res['data1']), parse_rate_for_calculation(res['data2'])
        col_calc1, col_calc2 = st.columns(2)
        with col_calc1: shipment_value = st.number_input("Shipment Value (USD)", 0.0, value=1000.0, step=100.0, key="compare_val")
        shipment_quantity = 0
        if rate_comp1.get("specific_rate_usd", 0) > 0 or rate_comp2.get("specific_rate_usd", 0) > 0:
            unit1, unit2 = rate_comp1.get('unit', 'units'), rate_comp2.get('unit', 'units')
            with col_calc2: shipment_quantity = st.number_input(f"Shipment Quantity ({unit1}/{unit2})", 0.0, value=100.0, step=10.0, key="compare_qty")
        if st.button("Calculate Duty Difference"):
            duty1 = (shipment_value * rate_comp1.get("ad_valorem_percent", 0)) + (shipment_quantity * rate_comp1.get("specific_rate_usd", 0))
            duty2 = (shipment_value * rate_comp2.get("ad_valorem_percent", 0)) + (shipment_quantity * rate_comp2.get("specific_rate_usd", 0))
            difference = duty1 - duty2
            st.subheader("Estimated Duty Comparison:")
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric(f"Product 1 ({res['hts1']}) Duty", f"${duty1:,.2f}")
            col_res2.metric(f"Product 2 ({res['hts2']}) Duty", f"${duty2:,.2f}")
            col_res3.metric("Potential Savings", f"${difference:,.2f}", delta_color="inverse")
        if st.button("Prepare Reclassification Letter"):
            st.session_state.reclassification_data = {'current_hs': res['hts1'], 'current_desc': desc1, 'proposed_hs': res['hts2'], 'proposed_desc': desc2}
            st.success("Data sent to Reclassification Helper. Please navigate to that page from the sidebar.")

elif page == "Reclassification Helper":
    st.title("📝 Reclassification Letter Helper")
    reclass_data = st.session_state.get('reclassification_data', {})
    if not reclass_data: st.warning("Please compare two products on the 'Compare' page first to pre-fill this form.")
    st.markdown("This tool helps you draft a letter to request a product reclassification. Fill in the details below.")
    st.text_input("Current HTS Code", value=reclass_data.get('current_hs', ''), key="current_hs_reclass")
    st.text_area("Current Product Description", value=reclass_data.get('current_desc', ''), key="current_desc_reclass", height=100)
    st.markdown("---")
    st.subheader("Proposed Reclassification")
    proposed_hs = st.text_input("Proposed HTS Code", value=reclass_data.get('proposed_hs', ''), help="Enter the HTS code you believe is correct.")
    proposed_desc_val = reclass_data.get('proposed_desc', '')
    if proposed_hs and not hts_df.empty:
        match = hts_df[hts_df['hts8'] == proposed_hs.replace('.', '')]
        if not match.empty: proposed_desc_val = match.iloc[0]['brief_description']
    st.text_area("Proposed Product Description", value=proposed_desc_val, key="proposed_desc_reclass", height=100, help="Description from local HTS data will appear here if the code is found.")
    justification = st.text_area("Your Justification", height=200, placeholder="Explain why the product fits the proposed classification better...")
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

elif page == "News & Overview":
    st.header("News & Overview")
    st.info("This section is a placeholder for the News & Overview feature outlined in the PRD.")
    st.write("As per the PRD (Section 4.2), this section will eventually display news and announcements.")

st.sidebar.markdown("---")
st.sidebar.subheader("Future Considerations (Post-MVP)")
st.sidebar.markdown("- **Save Favorites:** Requires a backend database.\n- **Notifications:** Requires a backend and user authentication.\n- **Historical Data Charts:** Implemented for MFN Rate.\n- **Duty Calculator:** Added to Search page.\n- **Associated Rulings:** Added to Search page.")
