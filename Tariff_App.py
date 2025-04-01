#
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import re

# Set page config
st.set_page_config(page_title="Tariff Database Query App", layout="wide")

# Load the combined dataset
@st.cache_data
def load_data():
    df = pd.read_csv('compressed_tariff_database.csv.zip', compression='zip')
    # Create HS chapter from first 2 digits of hts8
    if 'hts8' in df.columns:
        df['hs_chapter'] = df['hts8'].astype(str).str[:2].astype(int)
    return df

df = load_data()


# Add debugging information
st.sidebar.expander("Debug Info", expanded=False).write({
    "Data loaded": "Yes" if 'df' in locals() else "No",
    "DataFrame shape": df.shape if 'df' in locals() else None,
    "DataFrame columns": df.columns.tolist() if 'df' in locals() else None,
    "Agreement columns": [col for col in df.columns if '_ad_val_rate' in col] if 'df' in locals() else None
})
st.title('Tariff Database Query App')

# Sidebar filters
st.sidebar.header('Filters')

# Filter by years
years = sorted(df['year'].unique().tolist())
selected_years = st.sidebar.multiselect('Select years to analyze', options=years, default=years)

# Agreement selection
agreement_cols = [col for col in df.columns if '_ad_val_rate' in col]
if not agreement_cols:
    st.error("No agreement columns found in the dataset. Please check your data.")
    st.stop()

agreement_mapping = {col: col.split('_')[0].upper() for col in agreement_cols}
agreement_options = {v: k for k, v in agreement_mapping.items()}

if not agreement_options:
    st.error("No agreement options could be created. Please check your data.")
    st.stop()

selected_agreement = st.sidebar.selectbox('Select Agreement', options=list(agreement_options.keys()))

# Add error handling for agreement column selection
try:
    agreement_col = agreement_options[selected_agreement]
    st.sidebar.success(f"Selected agreement column: {agreement_col}")
except Exception as e:
    st.sidebar.error(f"Error selecting agreement: {e}")
    st.sidebar.write("Debug info: Selected agreement =", selected_agreement)
    st.sidebar.write("Available options:", list(agreement_options.keys()))
    # Fallback to first available agreement
    if agreement_options:
        selected_agreement = list(agreement_options.keys())[0]
        agreement_col = agreement_options[selected_agreement]
        st.sidebar.warning(f"Falling back to: {selected_agreement}")
    else:
        st.error("No agreements available. Cannot continue.")
        st.stop()

# Search product description
search_text = st.sidebar.text_input('Search Product Description', '')

# Select HS Chapter
if 'hs_chapter' in df.columns:
    hs_chapters = sorted(df['hs_chapter'].unique().tolist())
    selected_hs_chapter = st.sidebar.selectbox('Select HS Chapters', 
                                              options=[None] + hs_chapters,
                                              format_func=lambda x: f"Chapter {x}" if x else "All Chapters")

# Apply filters
filtered_df = df.copy()

# Filter by years
if selected_years:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]

# Filter by description
if search_text:
    try:
        # Escape special regex characters in search text
        escaped_search = re.escape(search_text)
        # Handle NaN values explicitly
        mask = filtered_df['brief_description'].fillna('').astype(str).str.contains(escaped_search, case=False)
        filtered_df = filtered_df[mask]
    except Exception as e:
        st.error(f"Error in search: {e}")
        st.write("Debug info: Search text =", search_text)

# Filter by HS chapter
if selected_hs_chapter:
    filtered_df = filtered_df[filtered_df['hs_chapter'] == selected_hs_chapter]

# Display filtered results
st.write(f"Filtered Results: {len(filtered_df)} items found.")

# Display the first 50 rows of the filtered dataframe
st.dataframe(filtered_df.head(50))

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader(f'Distribution of {selected_agreement} Ad Valorem Rates')
    
    try:
        # Check if agreement_col exists in the dataframe
        if agreement_col not in filtered_df.columns:
            st.error(f"Column '{agreement_col}' not found in the dataframe.")
            st.write("Available columns:", filtered_df.columns.tolist())
            st.warning("Cannot create rate distribution chart.")
        else:
            # Filter out NaN values for the selected agreement column
            rate_df = filtered_df[filtered_df[agreement_col].notna()]
            
            if not rate_df.empty:
                fig1, ax1 = plt.subplots(figsize=(9, 6))
                sns.histplot(rate_df[agreement_col], bins=20, kde=True, ax=ax1, color='#766CDB')
                ax1.set_title(f'{selected_agreement} Ad Valorem Rate Distribution', 
                             fontsize=20, pad=15, color='#222222')
                ax1.set_xlabel('Ad Valorem Rate', fontsize=16, labelpad=10, color='#333333')
                ax1.set_ylabel('Count', fontsize=16, labelpad=10, color='#333333')
                ax1.tick_params(axis='both', labelsize=14, colors='#555555')
                ax1.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig1)
            else:
                st.warning(f'No data available for {selected_agreement} with the current filters')
    except Exception as e:
        st.error(f"Error creating rate distribution chart: {e}")
        st.write("Debug info: agreement_col =", agreement_col)

with col2:
    st.subheader('Change Over Time for a Specific HS Code')
    
    # Option to select HTS code via dropdown or manual entry
    selection_method = st.radio('Select HTS Code via:', ['Dropdown', 'Manual Entry'])
    
    if selection_method == 'Dropdown':
        # Get unique HTS codes from filtered data
        hts_options = filtered_df['hts8'].unique().tolist()
        if hts_options:
            selected_hts = st.selectbox('Select HTS Code', options=hts_options)
        else:
            st.warning('No HTS codes available with current filters')
            selected_hts = None
    else:
        selected_hts = st.text_input('Enter HTS Code', '')
    
    if selected_hts:
        # Get description for the selected HTS code
        hts_desc = filtered_df[filtered_df['hts8'] == selected_hts]['brief_description'].iloc[0] if not filtered_df[filtered_df['hts8'] == selected_hts].empty else 'Description not available'
        st.write(f"Description: {hts_desc}")
        
        # Filter data for the selected HTS code
        hts_df = filtered_df[filtered_df['hts8'] == selected_hts].sort_values('year')
        
        # Display table of rates by year
        if not hts_df.empty:
            rate_table = hts_df[['year', agreement_col]].rename(columns={agreement_col: 'Ad Valorem Rate'})
            st.write("Ad Valorem Rate by Year:")
            st.table(rate_table)
            
            # Create line chart if we have data for multiple years
            if hts_df['year'].nunique() >= 2:
                fig2, ax2 = plt.subplots(figsize=(9, 6))
                ax2.plot(hts_df['year'], hts_df[agreement_col], marker='o', color='#766CDB', linewidth=2)
                ax2.set_title(f'{selected_agreement} Rate Trend for HS Code {selected_hts}', 
                             fontsize=20, pad=15, color='#222222')
                ax2.set_xlabel('Year', fontsize=16, labelpad=10, color='#333333')
                ax2.set_ylabel('Ad Valorem Rate', fontsize=16, labelpad=10, color='#333333')
                ax2.tick_params(axis='both', labelsize=14, colors='#555555')
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Annotate data points
                for _, row in hts_df.iterrows():
                    ax2.annotate(f"{row[agreement_col]}", 
                                (row['year'], row[agreement_col]), 
                                textcoords="offset points", 
                                xytext=(0, 10), 
                                ha='center', 
                                fontsize=12)
                st.pyplot(fig2)
            else:
                st.warning('Not enough data over multiple years for this HTS code')
        else:
            st.warning(f'No data available for HTS code {selected_hts}')
    else:
        st.info('Please select or enter an HTS code to view trend data')

# Add a section for significant rate changes
st.header('Significant Ad Valorem Rate Changes')

# Function to find significant changes
@st.cache_data
def find_significant_changes(df, agreement_col, min_change=0.02):
    # Group by HTS code and calculate min and max rates
    significant_changes = []
    
    for hts_code in df['hts8'].unique():
        hts_data = df[df['hts8'] == hts_code].sort_values('year')
        
        if len(hts_data) >= 2 and not hts_data[agreement_col].isna().all():
            # Get rates for each year
            rates_by_year = {}
            for _, row in hts_data.iterrows():
                if not pd.isna(row[agreement_col]):
                    rates_by_year[row['year']] = row[agreement_col]
            
            if rates_by_year:
                min_rate = min(rates_by_year.values())
                max_rate = max(rates_by_year.values())
                change = max_rate - min_rate
                
                if change >= min_change:
                    # Get description and bound rate
                    description = hts_data['brief_description'].iloc[0]
                    bound_rate = None
                    if 'wto_binding_code' in hts_data.columns:
                        bound_rate = hts_data['wto_binding_code'].iloc[0]
                    
                    # Create breakdown of rates by year
                    breakdown = []
                    for year, rate in rates_by_year.items():
                        breakdown.append({'year': year, 'rate': rate})
                    
                    significant_changes.append({
                        'hts8': hts_code,
                        'brief_description': description,
                        'agreement': selected_agreement,
                        'bound_rate': bound_rate,
                        'min_rate': min_rate,
                        'max_rate': max_rate,
                        'change': change,
                        'breakdown': breakdown
                    })
    
    # Sort by change magnitude (descending)
    significant_changes.sort(key=lambda x: x['change'], reverse=True)
    return significant_changes

# Find significant changes
if st.button('Find Significant Rate Changes'):
    with st.spinner('Analyzing rate changes...'):
        significant_changes = find_significant_changes(filtered_df, agreement_col)
        
        if significant_changes:
            st.write(f"Found {len(significant_changes)} significant rate changes")
            
            # Display top 10 changes
            top_changes = significant_changes[:10]
            
            # Create a table for display
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
            
            # Visualize the top change
            if len(top_changes) > 0:
                top_change = top_changes[0]
                st.subheader(f"Visualization of Largest Rate Change: {top_change['hts8']}")
                
                # Create visualization
                years = [item['year'] for item in top_change['breakdown']]
                rates = [item['rate'] for item in top_change['breakdown']]
                
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.plot(years, rates, marker='o', color='#766CDB', linewidth=2)
                ax.set_title(f"{selected_agreement} Rate Change for {top_change['hts8']}", 
                           fontsize=20, pad=15, color='#222222')
                ax.set_xlabel('Year', fontsize=16, labelpad=10, color='#333333')
                ax.set_ylabel('Ad Valorem Rate', fontsize=16, labelpad=10, color='#333333')
                ax.tick_params(axis='both', labelsize=14, colors='#555555')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Annotate data points
                for x, y in zip(years, rates):
                    ax.annotate(f"{y:.2%}", 
                              (x, y), 
                              textcoords="offset points", 
                              xytext=(0, 10), 
                              ha='center', 
                              fontsize=12)
                
                st.pyplot(fig)
        else:
            st.warning('No significant rate changes found with current filters')
"""

# Save the Streamlit app to a file
with open('tariff_app.py', 'w') as f:
    f.write(streamlit_code)

print("Streamlit app code has been saved to 'tariff_app.py'")
print("To run the app, use the command: streamlit run tariff_app.py")

# Display a sample of what the app will look like
print("\
App Features:")
print("1. Filters for years, agreements, HS chapters, and product descriptions")
print("2. Table display of filtered results")
print("3. Distribution chart of ad valorem rates")
print("4. Line chart showing rate changes over time for specific HS codes")
print("5. Analysis of significant rate changes with visualizations")
print("\
The app uses the combined dataset with data from 2020-2025")"""
