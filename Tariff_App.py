import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set page config
st.set_page_config(page_title="Tariff Database Query App", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_tariff_data(url):
    """Loads tariff data from a CSV URL and creates an HS chapter column."""
    try:
        df = pd.read_csv(url)
        if 'hts8' in df.columns:
            df['hs_chapter'] = df['hts8'].astype(str).str[:2].astype(int)
        return df
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return pd.DataFrame()

DATA_URL = 'https://drive.google.com/uc?export=download&id=1OvJqH7kMdmqeK6ldDnRNLs-9U3QhhDDl'
tariff_df = load_tariff_data(DATA_URL)

if tariff_df.empty:
    st.stop()

# --- Sidebar Configuration ---
st.sidebar.header('Filters')

# Filter by years
years = sorted(tariff_df['year'].unique().tolist())
selected_years = st.sidebar.multiselect('Select years to analyze', options=years, default=years)

# Agreement selection
all_agreement_columns = [col for col in tariff_df.columns if '_ad_val_rate' in col]
if not all_agreement_columns:
    st.error("No agreement columns found in the dataset. Please check your data.")
    st.stop()

agreement_mapping = {col: col.split('_')[0].upper() for col in all_agreement_columns}
agreement_options = {v: k for k, v in agreement_mapping.items()}

if not agreement_options:
    st.error("No agreement options could be created. Please check your data.")
    st.stop()

selected_agreement_display = st.sidebar.selectbox('Select Agreement', options=list(agreement_options.keys()))
selected_agreement_column = agreement_options[selected_agreement_display]

st.sidebar.success(f"Selected agreement: {selected_agreement_display} (Column: {selected_agreement_column})")

# Search product description
search_text = st.sidebar.text_input('Search Product Description', '')

# Select HS Chapter
if 'hs_chapter' in tariff_df.columns:
    hs_chapters = sorted(tariff_df['hs_chapter'].unique().tolist())
    selected_hs_chapter = st.sidebar.selectbox('Select HS Chapters',
                                                options=[None] + hs_chapters,
                                                format_func=lambda x: f"Chapter {x}" if x else "All Chapters")

# --- Data Filtering Functions ---
def filter_dataframe_by_year(df, years):
    """Filters the DataFrame by the selected years."""
    if years:
        return df[df['year'].isin(years)]
    return df

def filter_dataframe_by_description(df, search_term):
    """Filters the DataFrame by product description."""
    if search_term:
        try:
            escaped_search = re.escape(search_term)
            mask = df['brief_description'].fillna('').astype(str).str.contains(escaped_search, case=False)
            return df[mask]
        except Exception as e:
            st.error(f"Error in search: {e}")
            st.write("Debug info: Search text =", search_term)
    return df

def filter_dataframe_by_hs_chapter(df, hs_chapter):
    """Filters the DataFrame by the selected HS chapter."""
    if hs_chapter is not None:
        return df[df['hs_chapter'] == hs_chapter]
    return df

# Apply filters
filtered_tariff_df = tariff_df.copy()
filtered_tariff_df = filter_dataframe_by_year(filtered_tariff_df, selected_years)
filtered_tariff_df = filter_dataframe_by_description(filtered_tariff_df, search_text)
filtered_tariff_df = filter_dataframe_by_hs_chapter(filtered_tariff_df, selected_hs_chapter)

# --- Display Filtered Results ---
st.write(f"Filtered Results for {selected_agreement_display}: {len(filtered_tariff_df)} items found.")

# Display the first 50 rows of the filtered dataframe, showing only relevant columns
columns_to_display = ['year', 'brief_description', 'hts8', selected_agreement_column]
st.dataframe(filtered_tariff_df[columns_to_display].head(50))

# --- Charting Functions ---
def create_rate_distribution_plot(df, agreement_column, agreement_name):
    """Creates a histogram of ad valorem rates for the selected agreement."""
    if agreement_column not in df.columns:
        st.error(f"Column '{agreement_column}' not found in the dataframe.")
        st.write("Available columns:", df.columns.tolist())
        st.warning("Cannot create rate distribution chart.")
        return

    rate_df = df[df[agreement_column].notna()]

    if not rate_df.empty:
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        sns.histplot(rate_df[agreement_column], bins=20, kde=True, ax=ax1, color='#766CDB')
        ax1.set_title(f'{agreement_name} Ad Valorem Rate Distribution',
                            fontsize=20, pad=15, color='#222222')
        ax1.set_xlabel('Ad Valorem Rate', fontsize=16, labelpad=10, color='#333333')
        ax1.set_ylabel('Count', fontsize=16, labelpad=10, color='#333333')
        ax1.tick_params(axis='both', labelsize=14, colors='#555555')
        ax1.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig1)
    else:
        st.warning(f'No data available for {agreement_name} with the current filters')

def create_rate_trend_plot(df, agreement_column, agreement_name, hts_code):
    """Creates a line chart showing the rate trend for a specific HS code."""
    hts_df = df[df['hts8'] == hts_code].sort_values('year')

    if not hts_df.empty:
        rate_table = hts_df[['year', agreement_column]].rename(columns={agreement_column: 'Ad Valorem Rate'})
        st.write("Ad Valorem Rate by Year:")
        st.table(rate_table)

        if hts_df['year'].nunique() >= 2:
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            ax2.plot(hts_df['year'], hts_df[agreement_column], marker='o', color='#766CDB', linewidth=2)
            ax2.set_title(f'{agreement_name} Rate Trend for HS Code {hts_code}',
                                fontsize=20, pad=15, color='#222222')
            ax2.set_xlabel('Year', fontsize=16, labelpad=10, color='#333333')
            ax2.set_ylabel('Ad Valorem Rate', fontsize=16, labelpad=10, color='#333333')
            ax2.tick_params(axis='both', labelsize=14, colors='#555555')
            ax2.grid(True, linestyle='--', alpha=0.7)

            for _, row in hts_df.iterrows():
                ax2.annotate(f"{row[agreement_column]}",
                                    (row['year'], row[agreement_column]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center',
                                    fontsize=12)
            st.pyplot(fig2)
        else:
            st.warning('Not enough data over multiple years for this HTS code')
    else:
        st.warning(f'No data available for HTS code {hts_code}')

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader(f'Distribution of {selected_agreement_display} Ad Valorem Rates')
    create_rate_distribution_plot(filtered_tariff_df, selected_agreement_column, selected_agreement_display)

with col2:
    st.subheader('Change Over Time for a Specific HS Code')

    selection_method = st.radio('Select HTS Code via:', ['Dropdown', 'Manual Entry'])

    if selection_method == 'Dropdown':
        hts_options = filtered_tariff_df['hts8'].unique().tolist()
        selected_hts = st.selectbox('Select HTS Code', options=[None] + hts_options) if hts_options else None
    else:
        selected_hts = st.text_input('Enter HTS Code', '')

    if selected_hts:
        hts_desc = filtered_tariff_df[filtered_tariff_df['hts8'] == selected_hts]['brief_description'].iloc[0] if not filtered_tariff_df[filtered_tariff_df['hts8'] == selected_hts].empty else 'Description not available'
        st.write(f"Description: {hts_desc}")
        create_rate_trend_plot(filtered_tariff_df, selected_agreement_column, selected_agreement_display, selected_hts)
    else:
        st.info('Please select or enter an HTS code to view trend data')

# --- Significant Rate Changes ---
st.header('Significant Ad Valorem Rate Changes')

@st.cache_data
def find_significant_changes(df, agreement_column, agreement_name, min_change=0.02):
    """Finds significant ad valorem rate changes for the selected agreement."""
    significant_changes = []
    for hts_code in df['hts8'].unique():
        hts_data = df[df['hts8'] == hts_code].sort_values('year')
        if len(hts_data) >= 2 and not hts_data[agreement_column].isna().all():
            rates_by_year = {}
            for _, row in hts_data.iterrows():
                if not pd.isna(row[agreement_column]):
                    rates_by_year[row['year']] = row[agreement_column]

            if rates_by_year:
                min_rate = min(rates_by_year.values())
                max_rate = max(rates_by_year.values())
                change = max_rate - min_rate

                if change >= min_change:
                    description = hts_data['brief_description'].iloc[0]
                    bound_rate = hts_data['wto_binding_code'].iloc[0] if 'wto_binding_code' in hts_data.columns else None
                    breakdown = [{'year': year, 'rate': rate} for year, rate in rates_by_year.items()]
                    significant_changes.append({
                        'hts8': hts_code,
                        'brief_description': description,
                        'agreement': agreement_name,
                        'bound_rate': bound_rate,
                        'min_rate': min_rate,
                        'max_rate': max_rate,
                        'change': change,
                        'breakdown': breakdown
                    })
    significant_changes.sort(key=lambda x: x['change'], reverse=True)
    return significant_changes

if st.button('Find Significant Rate Changes'):
    with st.spinner('Analyzing rate changes...'):
        significant_changes = find_significant_changes(filtered_tariff_df, selected_agreement_column, selected_agreement_display)
        if significant_changes:
            st.write(f"Found {len(significant_changes)} significant rate changes for {selected_agreement_display}")
            top_changes = significant_changes[:10]
            change_data = []
            for i, change in enumerate(top_changes):
                breakdown_str = ', '.join([f"{item['year']}: {item['rate']:.2%}" for item in change['breakdown']])
                change_data.append({
                    'number': i+1,
                    'hts8': change['hts8'],
                    'description': change['brief_description'],
                    'agreement': change['agreement'],
                    'bound_rate': change['bound_rate'],
                    'breakdown': breakdown_str,
                    'change': f"{change['change']:.2f}"
                })
            change_df = pd.DataFrame(change_data)
            st.table(change_df)

            if top_changes:
                top_change = top_changes[0]
                st.subheader(f"Visualization of Largest Rate Change for {top_change['hts8']} ({top_change['agreement']})")
                years_viz = [item['year'] for item in top_change['breakdown']]
                rates_viz = [item['rate'] for item in top_change['breakdown']]
                fig_sig, ax_sig = plt.subplots(figsize=(9, 6))
                ax_sig.plot(years_viz, rates_viz, marker='o', color='#766CDB', linewidth=2)
                ax_sig.set_title(f"{top_change['agreement']} Rate Change for {top_change['hts8']}", fontsize=20, pad=15, color='#222222')
                ax_sig.set_xlabel('Year', fontsize=16, labelpad=10, color='#333333')
                ax_sig.set_ylabel('Ad Valorem Rate', fontsize=16, labelpad=10, color='#333333')
                ax_sig.tick_params(axis='both', labelsize=14, colors='#555555')
                ax_sig.grid(True, linestyle='--', alpha=0.7)
                for x, y in zip(years_viz, rates_viz):
                    ax_sig.annotate(f"{y:.2%}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=12)
                st.pyplot(fig_sig)
        else:
            st.warning(f'No significant rate changes found for {selected_agreement_display} with current filters')

# --- Debug Info (Moved to the end for better flow) ---
st.sidebar.expander("Debug Info", expanded=False).write({
    "Data loaded": "Yes" if 'tariff_df' in locals() and not tariff_df.empty else "No",
    "DataFrame shape": tariff_df.shape if 'tariff_df' in locals() and not tariff_df.empty else None,
    "DataFrame columns": tariff_df.columns.tolist() if 'tariff_df' in locals() and not tariff_df.empty else None,
    "Selected agreement column": selected_agreement_column,
})
