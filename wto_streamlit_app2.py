import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="WTO Tariff Lookup", layout="wide")

API_URL = "https://api.wto.org/timeseries/v1/data"
API_KEY = "ab5ad8703cd54ffba080cb9554175101"

# Debug mode in sidebar
# debug_mode = st.sidebar.checkbox("Debug Mode")

def create_pivot_table(df, value_column):
    """Create a pivot table with HS6 codes as rows and years as columns"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    df_copy["Year"] = df_copy["Year"].astype(str)
    
    # Create pivot table
    pivot = df_copy.pivot_table(
        index=["HS6", "Description"],
        columns="Year",
        values=value_column,
        aggfunc="first"
    ).reset_index()
    
    # Sort columns chronologically
    year_cols = [col for col in pivot.columns if col not in ["HS6", "Description"]]
    year_cols = sorted(year_cols)
    all_cols = ["HS6", "Description"] + year_cols
    
    return pivot[all_cols]

@st.cache_data
def load_hts_data():
    try:
        df = pd.read_excel("hts8.xlsx")
        df["hts8"] = df["hts8"].astype(str).str.zfill(8)
        df["hs6"] = df["hts8"].str[:6]
        df = df.drop_duplicates(subset=["hts8", "brief_description"])
        return df
    except Exception as e:
        # if debug_mode:
        #     st.error(f"Error loading HTS data: {e}")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=["hts8", "hs6", "brief_description"])

@st.cache_data
def load_country_list():
    fallback_countries = [
        {"name": "Albania", "code": "008"},
        {"name": "Australia", "code": "036"},
        {"name": "United States of America", "code": "840"},
        {"name": "China", "code": "156"},
        {"name": "European Union", "code": "918"}
    ]
    
    try:
        url = "https://api.wto.org/timeseries/v1/metadata/reporters?i=HS_A_0010"
        headers = {"subscription-key": API_KEY}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            countries = response.json()
            return sorted([{"name": c["text"], "code": c["id"]} for c in countries], key=lambda x: x["name"])
        else:
            # if debug_mode:
            #     st.info("Using fallback country list (API returned non-200 status)")
            return fallback_countries
    except Exception as e:
        # if debug_mode:
        #     st.info(f"Using fallback country list (Exception: {str(e)})")
        return fallback_countries
    
hts_df = load_hts_data()
countries = load_country_list()
country_names = [country["name"] for country in countries]

st.title("🌍 WTO Tariff Lookup by Product or Code")

# Input section
search_term = st.text_input("Search product description (e.g., motorcycles, cheese, rice)")
code_prefix = st.text_input("Filter by HTS or HS code prefix (e.g., 1006, 1701)")

years = [2023, 2022, 2021, 2020, 2019, 2018, 2024, 2025]
years.sort()
min_year = years[0]
max_year = years[-1]
default_range = f"{min_year}-{max_year}"

use_year_range = st.checkbox("Use year range instead of single year", value=True)
if use_year_range:
    year_range = st.text_input("Year Range (e.g., 2018-2025)", value=default_range)
else:
    year = st.selectbox("Select Year", years)
    year_range = str(year)

reporter = st.selectbox("Select Reporter Country", country_names)
reporter_code = next((c["code"] for c in countries if c["name"] == reporter), None)

partner = st.selectbox("Select Partner Country", ["All"] + country_names)
partner_code = next((c["code"] for c in countries if c["name"] == partner), None) if partner != "All" else None

# Filter products based on search
filtered_df = hts_df.copy()
if search_term:
    filtered_df = filtered_df[filtered_df["brief_description"].str.contains(search_term, case=False, na=False)]
if code_prefix:
    filtered_df = filtered_df[
        filtered_df["hts8"].str.startswith(code_prefix) | filtered_df["hs6"].str.startswith(code_prefix)
    ]

# Select HTS8 codes
selected_hts8 = []
if not filtered_df.empty:
    selected_hts8 = st.multiselect(
        label="Select HTS8 Codes",
        options=filtered_df["hts8"].tolist(),
        default=filtered_df["hts8"].tolist()[:min(5, len(filtered_df))],  # Limit default selection
        format_func=lambda x: f"{x} – {filtered_df[filtered_df['hts8'] == x]['brief_description'].iloc[0]}"
    )

# Get unique HS6 codes
selected_hs6 = list(set(
    filtered_df[filtered_df["hts8"].isin(selected_hts8)]["hs6"].tolist()
)) if selected_hts8 else []

if selected_hs6:
    st.markdown("#### 📦 HS6 codes being queried:")
    st.code(", ".join(selected_hs6))

def fetch_tariff_data(indicator_code, hs6_list, year_range, reporter_code, partner_code=None):
    """
    Fetch tariff data from the WTO API with improved response handling
    """
    if not hs6_list:
        return pd.DataFrame()
        
    params = {
        "i": indicator_code,
        "pc": ",".join(hs6_list),
        "ps": year_range,
        "r": reporter_code,
        "subscription-key": API_KEY
    }
    
    if partner_code:
        params["p"] = partner_code
    
    # if debug_mode:
    #     st.write(f"API Request URL: {API_URL}?{requests.compat.urlencode(params)}")
    
    try:
        response = requests.get(API_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # if debug_mode:
            #     with st.expander(f"Raw API Response for {indicator_code}", expanded=False):
            #         st.json(data)
            
            if "Dataset" in data and data["Dataset"]:
                records = []
                for item in data["Dataset"]:
                    record = {
                        "Reporter": item.get("ReportingEconomy"),
                        "Reporter Code": item.get("ReportingEconomyCode"),
                        "HS6": item.get("ProductOrSectorCode"),
                        "Description": item.get("ProductOrSector"),
                        "Year": item.get("Year"),
                        indicator_code: item.get("Value")
                    }
                    
                    # Add partner country information if available in the response
                    if "Partner" in item:
                        record["Partner"] = item.get("Partner")
                        record["Partner Code"] = item.get("PartnerCode")
                        
                    records.append(record)
                return pd.DataFrame(records)
            else:
                # if debug_mode:
                #     st.info(f"API returned successfully but no data found for {indicator_code}")
                return pd.DataFrame()
        elif response.status_code == 204:
            # if debug_mode:
            #     st.info(f"API returned status 204 (No Content) for {indicator_code}")
            return pd.DataFrame()
        else:
            st.error(f"API error {response.status_code} for {indicator_code}")
            # if debug_mode:
            #     st.write(response.text)
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Exception occurred: {str(e)}")
        return pd.DataFrame()

if st.button("Fetch Tariff Data"):
    if not selected_hts8:
        st.warning("Please select at least one product code.")
    else:
        with st.spinner("Querying WTO API..."):
            # Fetch tariff rates (applied and bound) and import data
            applied_df = fetch_tariff_data("HS_A_0010", selected_hs6, year_range, reporter_code, partner_code)
            bound_df = fetch_tariff_data("HS_A_0020", selected_hs6, year_range, reporter_code, partner_code)
            import_df = fetch_tariff_data("HS_P_0070", selected_hs6, year_range, reporter_code, partner_code)

            if applied_df.empty and bound_df.empty and import_df.empty:
                st.warning("No data found for the selected items and year(s).")
                st.info("Try different products, years, or reporter countries.")
            else:
                dfs_to_merge = []
                
                if not applied_df.empty:
                    dfs_to_merge.append(applied_df.rename(columns={"HS_A_0010": "Applied Rate (%)"}))
                
                if not bound_df.empty:
                    dfs_to_merge.append(bound_df.rename(columns={"HS_A_0020": "Bound Rate (%)"}))
                
                if not import_df.empty:
                    dfs_to_merge.append(import_df.rename(columns={"HS_P_0070": "Import Value (1000 USD)"}))
                
                merge_cols = ["Reporter", "Reporter Code", "HS6", "Description", "Year"]
                if partner_code:
                    # Include partner columns if we used a partner country
                    merge_cols.extend(["Partner", "Partner Code"])
                
                if len(dfs_to_merge) >= 2:
                    # Start with the first dataframe
                    merged = dfs_to_merge[0]
                    
                    for i in range(1, len(dfs_to_merge)):
                        merged = pd.merge(
                            merged,
                            dfs_to_merge[i],
                            on=merge_cols,
                            how="outer"
                        )
                else:
                    merged = dfs_to_merge[0] if dfs_to_merge else pd.DataFrame()
                
                if not hts_df.empty:
                    hts_matches = hts_df[hts_df["hs6"].isin(merged["HS6"])]
                    if not hts_matches.empty:
                        merged = pd.merge(
                            merged,
                            hts_matches[["hs6", "hts8", "brief_description"]].drop_duplicates(),
                            left_on="HS6",
                            right_on="hs6",
                            how="left"
                        ).drop(columns="hs6")

                total_records = len(merged)
                st.success(f"Data loaded successfully! Found {total_records} records.")
                
                if "Applied Rate (%)" in merged.columns:
                    st.subheader("Applied Rates (HS_A_0010)")
                    pivot_applied = create_pivot_table(merged, "Applied Rate (%)")
                    st.dataframe(pivot_applied, use_container_width=True)
                
                if "Bound Rate (%)" in merged.columns:
                    st.subheader("Bound Rates (HS_A_0020)")
                    pivot_bound = create_pivot_table(merged, "Bound Rate (%)")
                    st.dataframe(pivot_bound, use_container_width=True)
                
                if "Import Value (1000 USD)" in merged.columns:
                    st.subheader("Import Values (HS_P_0070)")
                    pivot_import = create_pivot_table(merged, "Import Value (1000 USD)")
                    st.dataframe(pivot_import, use_container_width=True)
                
                with st.expander("View Original Data", expanded=True):
                    st.dataframe(merged)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Detailed Data (CSV)",
                        data=merged.to_csv(index=False),
                        file_name=f"wto_data_detailed_{year_range}.csv",
                        mime="text/csv"
                    )
                
                # Create combined pivot data for download
                pivot_data = []
                if "Applied Rate (%)" in merged.columns:
                    pivot_data.append((pivot_applied, "Applied_"))
                if "Bound Rate (%)" in merged.columns:
                    pivot_data.append((pivot_bound, "Bound_"))
                if "Import Value (1000 USD)" in merged.columns:
                    pivot_data.append((pivot_import, "Import_"))
                
                if len(pivot_data) >= 2:
                    with col2:
                        pivot_combined = pivot_data[0][0].add_prefix(pivot_data[0][1])
                        
                        for i in range(1, len(pivot_data)):
                            current_prefix = pivot_data[i][1]
                            previous_prefix = pivot_data[i-1][1]
                            
                            pivot_combined = pd.merge(
                                pivot_combined,
                                pivot_data[i][0].add_prefix(current_prefix),
                                left_on=[f"{previous_prefix}HS6", f"{previous_prefix}Description"],
                                right_on=[f"{current_prefix}HS6", f"{current_prefix}Description"],
                                how="outer"
                            )
                        
                        st.download_button(
                            label="Download Pivoted Data (CSV)",
                            data=pivot_combined.to_csv(index=False),
                            file_name=f"wto_data_pivot_{year_range}.csv",
                            mime="text/csv"
                        )