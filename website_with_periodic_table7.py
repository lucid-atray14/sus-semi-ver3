import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
import streamlit as st
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.methods import TOPSIS
from io import BytesIO
from bokeh.transform import jitter, factor_cmap
from streamlit_bokeh import streamlit_bokeh
from bokeh.palettes import Category10, Category20
import re
import matplotlib.pyplot as plt
import random

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Lighten sidebar background */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Dark text for light sidebar */
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            color: #333333 !important;
        }
        
        /* Sidebar hover effects */
        [data-testid="stSidebar"] .stRadio > div:hover {
            background-color: #e9ecef;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_composition_database():
    df1 = pd.read_excel("8_material_properties_cleaned.xlsx")
    return df1

@st.cache_data
def load_high_confidence_database():
    df2 = pd.read_excel("materials_high_confidence_cleaned.xlsx")
    return df2.iloc[:, 1:]

def filter_dataframe(df, filters, selected_names=None):
    """Filter dataframe based on provided filters and optional names"""
    filtered = df.copy()
    
    # Apply each filter dynamically
    for filter_name, filter_range in filters.items():
        if filter_name in df.columns:
            filtered = filtered[
                filtered[filter_name].between(filter_range[0], filter_range[1], inclusive='both')
            ]
    
    if selected_names is not None:
        filtered = filtered[filtered["Name"].isin(selected_names)]
    
    return filtered

def run_topsis(matrix, weights, criteria_types):
    topsis = TOPSIS()
    return topsis(matrix, weights, criteria_types)

def run_promethee(matrix, weights, criteria_types):
    promethee = PROMETHEE_II('usual')
    return promethee(matrix, weights, criteria_types)

@st.cache_data
def prepare_plot_data(df, x_col, y_col, log_x=False, log_y=False):
    df_plot = df.copy()
    if log_x:
        df_plot[x_col] = np.log10(df_plot[x_col].clip(lower=1e-10))
    if log_y:
        df_plot[y_col] = np.log10(df_plot[y_col].clip(lower=1e-10))
    return df_plot

@st.cache_data
def create_full_output(filtered_df, results_df, weights_df):
    """Create Excel output with all MCDM analysis results"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Prepare full data with MCDM results
        full_data = filtered_df.copy()
        
        # Reset index of results_df to get Rank as a column
        results_reset = results_df.reset_index()
        
        # Merge MCDM scores/ranks with full data
        if 'Score' in results_reset.columns:
            # TOPSIS results
            score_map = dict(zip(results_reset['Material'], results_reset['Score']))
            rank_map = dict(zip(results_reset['Material'], results_reset['Rank']))
            full_data['TOPSIS_Score'] = full_data['Name'].map(score_map)
            full_data['TOPSIS_Rank'] = full_data['Name'].map(rank_map)
        else:
            # PROMETHEE results
            flow_map = dict(zip(results_reset['Material'], results_reset['Net Flow']))
            rank_map = dict(zip(results_reset['Material'], results_reset['Rank']))
            full_data['PROMETHEE_Net_Flow'] = full_data['Name'].map(flow_map)
            full_data['PROMETHEE_Rank'] = full_data['Name'].map(rank_map)
        
        # Write sheets
        full_data.to_excel(writer, sheet_name='Full Data', index=False)
        results_reset.to_excel(writer, sheet_name='Rankings', index=False)
        weights_df.reset_index().to_excel(writer, sheet_name='Weights', index=False)
        
        # Filter settings
        if 'filters' in st.session_state and st.session_state.filters:
            filter_settings = pd.DataFrame([
                {'Filter': k, 'Min': v[0], 'Max': v[1]} 
                for k, v in st.session_state.filters.items()
            ])
            filter_settings.to_excel(writer, sheet_name='Filter Settings', index=False)
    
    return output.getvalue()

def create_professional_plot(df, x_col, y_col, title, x_label, y_label, log_x=False, log_y=False):
    # Create a copy to avoid modifying the original dataframe
    df_plot = df.copy()
    
    # Professional color palette
    primary_color = "#3498db"
    highlight_color = "#c5301f"
    
    # Create the figure with dynamic axis types
    p = figure(
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        x_axis_label=f"log({x_label})" if log_x else x_label,
        y_axis_label=f"log({y_label})" if log_y else y_label,
        x_axis_type="log" if log_x else "linear",
        y_axis_type="log" if log_y else "linear",
        width=800,
        height=500,
        tooltips=[("Name", "@Name")],
        toolbar_location="above",
        sizing_mode="stretch_width"
    )
    
    # Handle negative/zero values for log scales
    if log_x:
        df_plot[x_col] = df_plot[x_col].clip(lower=1e-10)
    if log_y:
        df_plot[y_col] = df_plot[y_col].clip(lower=1e-10)
    
    # Plot all points
    source = ColumnDataSource(df_plot)
    p.circle(
        x=x_col,
        y=y_col,
        source=source,
        size=8,
        color=primary_color,
        alpha=0.6,
        legend_label="All Materials"
    )
    
    # Highlight exactly 10 random materials
    num_highlight = min(10, len(df_plot))
    highlight_df = df_plot.sample(n=num_highlight, random_state=42)
    highlight_source = ColumnDataSource(highlight_df)
    
    p.circle(
        x=x_col,
        y=y_col,
        source=highlight_source,
        size=12,
        color=highlight_color,
        alpha=1.0,
        legend_label="Highlighted Materials"
    )
    
    # Add labels to highlighted points
    labels = LabelSet(
        x=x_col,
        y=y_col,
        text="Name",
        source=highlight_source,
        text_font_size="10pt",
        text_color=highlight_color,
        y_offset=8,
        text_align='center'
    )
    p.add_layout(labels)
    
    # Professional legend styling
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.7
    p.legend.label_text_font_size = "12pt"
    
    # Grid and axis styling
    p.xgrid.grid_line_color = "#e0e0e0"
    p.ygrid.grid_line_color = "#e0e0e0"
    p.axis.minor_tick_line_color = None
    
    return p

def format_tons(value):
    """Format large numbers with appropriate units (M/B)"""
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.1f}TB tons"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B tons"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M tons"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K tons"
    else:
        return f"{value:.0f} tons"

def filter_by_excluded_elements(df, excluded_elements):
    """
    Filter out materials that contain any of the excluded elements.
    
    Parameters:
    - df: DataFrame with Element_1 through Element_7 columns
    - excluded_elements: List of element symbols to exclude
    
    Returns:
    - Filtered DataFrame with materials that don't contain any excluded elements
    """
    if not excluded_elements:
        return df.copy()
    
    # Create a mask for rows to keep (rows that don't contain excluded elements)
    element_columns = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 
                       'Element_5', 'Element_6', 'Element_7']
    
    # Start with all rows included
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Normalize excluded elements (strip whitespace, handle case)
    excluded_elements_normalized = [str(elem).strip() for elem in excluded_elements]
    
    # Check each element column
    for col in element_columns:
        if col in df.columns:
            # Mark rows as False if they contain any excluded element
            # Handle potential NaN and string comparison issues
            column_values = df[col].fillna('').astype(str).str.strip()
            mask &= ~column_values.isin(excluded_elements_normalized)
    
    return df[mask].copy()

def main():
    set_custom_style()
    df1 = load_composition_database()
    df2 = load_high_confidence_database()

    # Sidebar navigation
    st.sidebar.title("Material Analysis")
    st.sidebar.markdown("---")
    selected_page = st.sidebar.radio(
        "Navigation Menu", 
        ["Home", "Bandgap Information", "Decision-making Assistant"],
        captions=["Welcome page", "Commonly researched semiconductors", "Multi-criteria decision making tool"]
    )
    
    # Add footer
    st.markdown("""
    <div class="footer">
        Semiconductor Database © 2025 | v3.0 | Developed by HERAWS
    </div>
    """, unsafe_allow_html=True)

    if selected_page == "Home":
        st.title("Semiconductor Database")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ### 🔍 About This Tool
            This interactive platform enables comprehensive analysis of environmental impacts and sustainability of semiconductors with:
            - **Extensive database** on ESG scores, CO₂ footprints, and more
            - **Visualizations** to explore relationships between parameters
            - **Multi-criteria** decision making tools (TOPSIS, PROMETHEE)
            - **Export capabilities** for further analysis
            """)
            
        with cols[1]:
            st.markdown("""
            ### 🚀 Getting Started
            1. Select an analysis page from the sidebar
            2. Configure your filters and parameters
            3. Visualize the relationships
            4. Download results for further use
            
            **Pro Tip:** Use the MCDM analysis for ranking the most promising semiconductors.
            """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Database Information")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Total Materials", len(df1))
            prod_min = df1['Production (ton)'].min()
            prod_max = df1['Production (ton)'].max()
            st.metric("Production Range", f"{format_tons(prod_min)} - {format_tons(prod_max)}")
        with cols[1]:
            st.metric("Bandgap Range", f"{df1['Bandgap'].min():.1f} - {df1['Bandgap'].max():.1f} eV")

        
    elif selected_page == "Bandgap Information":
        st.title("Bandgap Information")
        st.markdown("Most commonly researched semiconductors and their band gap range.")

        # Session state initialization

        if "included_elements" not in st.session_state:
            st.session_state.included_elements = []
        if "filters_applied" not in st.session_state:
            st.session_state.filters_applied = False

        # ELEMENT INCLUSION SECTION

        st.markdown("### Element Inclusion")
        
        # Using a URL for periodic table image
        periodic_table_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Simple_Periodic_Table_Chart-en.svg/1200px-Simple_Periodic_Table_Chart-en.svg.png"
        st.image(periodic_table_url, caption="Periodic Table of Elements", use_container_width=True)

        # Get all unique elements from df2 (element columns are from index 3 to -2)
        element_cols_df2 = df2.columns[3:-2].tolist()
        
        # For df1, element columns are Element_1 through Element_7
        element_cols_df1 = [f'Element_{i}' for i in range(1, 8)]
        
        # Copy df1 for filtering
        df = df1.copy()
        
        # Get all unique elements from both datasets
        all_elements = set(element_cols_df2)
        
        # Also get elements from df1 if available
        for col in element_cols_df1:
            if col in df.columns:
                elements = df[col].dropna().unique()
                all_elements.update(elements)
        
        # Remove empty strings and sort
        all_elements = {elem for elem in all_elements if elem and str(elem).strip()}
        all_elements = sorted(list(all_elements))
        
        # Helper function for filtering df1
        def filter_df1_by_included_elements(dataframe, included_list, element_cols):
            """Filter df1 to only include rows with ALL elements in the included list."""
            if not included_list:
                return dataframe.iloc[0:0].copy()
            
            if not element_cols:
                return dataframe.copy()
            
            def _is_empty(x):
                return pd.isna(x) or str(x).strip() == ""
            
            mask = pd.Series(True, index=dataframe.index)
            included_set = set(included_list)
            
            for c in element_cols:
                if c in dataframe.columns:
                    mask &= dataframe[c].isin(included_set) | dataframe[c].apply(_is_empty)
            
            return dataframe[mask].copy()
        
        # Helper function for filtering df2
        def filter_df2_by_included_elements(dataframe, included_list, element_cols):
            """Filter df2 to only include rows with elements present (>0) only in the included list."""
            if not included_list:
                return dataframe.iloc[0:0].copy()
            
            if not element_cols:
                return dataframe.copy()
            
            included_set = set(included_list)
            mask = pd.Series(True, index=dataframe.index)
            
            # For each element column
            for elem_col in element_cols:
                if elem_col in dataframe.columns:
                    # If element is NOT in included list, it must have value 0
                    if elem_col not in included_set:
                        mask &= (dataframe[elem_col] == 0)
            
            return dataframe[mask].copy()
        
        # Apply current filters to show impact on df1
        df1_filtered = filter_df1_by_included_elements(df, st.session_state.included_elements, element_cols_df1)
        
        # Apply current filters to show impact on df2
        df2_filtered_preview = filter_df2_by_included_elements(df2, st.session_state.included_elements, element_cols_df2)
        
        # Show info about element filtering
        if st.session_state.included_elements:
            removed_count_df1 = len(df) - len(df1_filtered)
            removed_count_df2 = len(df2) - len(df2_filtered_preview)
            st.info(f"🔬 Element filter active: Included {', '.join(sorted(st.session_state.included_elements))} | "
                    f"Showing {len(df1_filtered)} of {len(df)} materials")
        
        st.markdown("**Enter element symbols to include (separated by commas)**")

        # Text input for manual entry
        element_text_input = st.text_input(
            "Element symbols:",
            value=", ".join(st.session_state.included_elements) if st.session_state.included_elements else "",
            key="element_text_input",
            placeholder="e.g., Ti, O, Zn",
            help="Enter element symbols separated by commas. Spaces are optional. Only materials containing these elements will be shown."
        )
        
        # Parse the input
        if element_text_input.strip():
            selected_elements = [elem.strip() for elem in element_text_input.split(',') if elem.strip()]
        else:
            selected_elements = []
        
        # Show count of currently included elements
        if st.session_state.included_elements:
            st.markdown(f"**Currently Included Elements:** {len(st.session_state.included_elements)}")
        
        # Show impact preview for element inclusion
        if selected_elements != st.session_state.included_elements:
            preview_df1 = filter_df1_by_included_elements(df, selected_elements, element_cols_df1)
            preview_df2 = filter_df2_by_included_elements(df2, selected_elements, element_cols_df2)
            
            if len(preview_df1) > 0 or len(preview_df2) > 0:
                st.success(f"✅ Preview: {len(preview_df1)} materials ({len(preview_df1)/len(df)*100:.1f}%)")
            else:
                st.warning(f"⚠️ Preview: No materials contain only these elements.")
        
        # Update session state with selected elements
        st.session_state.included_elements = selected_elements
        
        # Apply filters button
        colA, colB = st.columns([1, 3])
        with colA:
            apply_clicked = st.button("Apply Filters", key="apply_initial_filters")
            if apply_clicked:
                st.session_state.filters_applied = True
                st.rerun()
        
        # Use filtered dataframe for rest of the page
        df_filtered = df1_filtered.copy()

        # Identify the bandgap column name
        bandgap_col = None
        possible_names = ['Bandgap', 'bandgap', 'Band_gap', 'band_gap', 'Value', 'value', 'BandGap']
        for col_name in possible_names:
            if col_name in df_filtered.columns:
                bandgap_col = col_name
                break
        # =========================
        # Utility functions
        # =========================
        def pick_palette(n: int):
            if n <= 10:
                return Category10[10][:n]
            return Category20[20][:min(n, 20)]

        _element_pat = re.compile(r"[A-Z][a-z]?")

        # Aggregate data for top materials
        df_agg = (
            df_filtered.groupby("Name")
            .size()
            .reset_index(name="Count")
            .sort_values(by="Count", ascending=False)
        ).head(9)

        st.session_state.df_agg = df_agg

        # Top 9 names
        top_names = df_agg["Name"].head(9).tolist()
        st.markdown(
            """
            <h3 style='
                font-size:20px;
                font-weight:600; 
                color:#222;
                margin-top:10px;
                margin-bottom:20px;
                font-family:Arial;
            '>
            Exploratory Data Analysis of Database
            </h3>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("***The scatterplot provides a broad visual overview of the relative bandgap ranges across the materials. It shows individual data points and overall range of bandgaps per semiconductor. Moreover, it provides visual spread and clustering trends.***")

        if bandgap_col is None or not top_names:
            st.warning(f"⚠️ No bandgap column found or no top names available.")
        else:
            filtered_df_top = df_filtered[df_filtered["Name"].isin(top_names)].copy()

            if filtered_df_top.empty:
                st.info("No data to display for the current selection of elements.")
            else:
                source = ColumnDataSource(filtered_df_top)
                y_factors = sorted(filtered_df_top["Name"].unique().tolist())
                palette = pick_palette(len(y_factors))

                p = figure(
                    y_range=y_factors,
                    width=900, height=500,
                    toolbar_location=None,
                    title=None
                )

                p.circle(
                    x=bandgap_col,
                    y=jitter("Name", width=0.3, range=p.y_range),
                    source=source,
                    size=10,
                    alpha=0.9,
                    color=factor_cmap("Name", palette=palette, factors=y_factors)
                )

                p.add_tools(HoverTool(tooltips=[("Material", "@Name"), ("Bandgap (eV)", f"@{bandgap_col}")]))
                p.xaxis.axis_label = "Bandgap (eV)"
                p.yaxis.axis_label = "Semiconductor"
                p.xgrid.visible = True
                p.ygrid.visible = False
                p.outline_line_color = None

                streamlit_bokeh(p, key="professional_graph")

        st.markdown("***Histogram plot shows the frequency distribution of bandgaps. It offers quantitative information regarding central tendency and data concentration.***")

        if bandgap_col is None:
            st.warning(f"⚠️ No bandgap column found. Available columns: {', '.join(df_filtered.columns.tolist())}")
            st.info("Cannot display KDE plots without bandgap data.")
        else:
            df_hist = df_filtered[df_filtered["Name"].isin(top_names)].copy()
            df_hist = df_hist[pd.to_numeric(df_hist[bandgap_col], errors="coerce").notna()]
            df_hist[bandgap_col] = df_hist[bandgap_col].astype(float)

            if df_hist.empty:
                st.info("No data to plot after filtering. Try including more materials.")
            else:
                # Shared x-range for fair comparison
                x_min, x_max = df_hist[bandgap_col].min(), 10
                # Ensure we only render up to 9 panels
                top9 = top_names[:9]

                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), sharex=True, sharey=True)
                axes = axes.flatten()

                for i, name in enumerate(top9):
                    ax = axes[i]
                    sub = df_hist.loc[df_hist["Name"] == name, bandgap_col].dropna().values

                    if sub.size >= 1:
                        ax.hist(sub, bins='auto', density=False, alpha=0.85)
                        if sub.size >= 2:
                            # Median guide
                            ax.axvline(np.median(sub), linestyle="--", linewidth=1)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=ax.transAxes, fontsize=9, alpha=0.7)

                    ax.set_title(f"{name} (n={sub.size})", fontsize=11)
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
                    ax.set_xlim(x_min, x_max)
                    if i % 3 != 0: ax.set_ylabel("")
                    if i < 6: ax.set_xlabel("")

                # Hide any unused slots if fewer than 9 materials
                for j in range(len(top9), 9):
                    axes[j].axis("off")

                fig.supylabel("Count")
                fig.supxlabel("Bandgap (eV)")
                #fig.suptitle("Bandgap Histograms — Top 9 Materials", fontsize=13)
                fig.tight_layout(rect=[0, 0.02, 1, 0.95])

                st.pyplot(fig, clear_figure=True)

        # --- Section: Scatter plot of recently researched materials ---
        st.markdown("***The scatter plot visualizes the trends of recently researched materials and their corresponding bandgap values.***")

        # --- Materials to keep (from df_agg) ---
        unique_list = df_agg['Name'].dropna().unique().tolist()

        # --- Filter df2 by Name (not by all columns) ---
        if 'Name' not in df2.columns:
            st.warning("Missing required column: Name")
            st.stop()

        # Copy and ensure 'Date' column is datetime
        df2_set = df2.copy()
        df2_set['Date'] = pd.to_datetime(df2_set['Date'], errors='coerce')

        # Define minimum and maximum dates in the dataset
        min_date = df2_set['Date'].min().date()
        max_date = df2_set['Date'].max().date()

        # Create a date range selector
        start_date, end_date = st.date_input(
            "Select start and end date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # --- Filter DataFrame ---
        mask = (df2_set['Date'].dt.date >= start_date) & (df2_set['Date'].dt.date <= end_date)
        df_filtered = df2_set.loc[mask].copy()
        #df2_filtered = df2_filtered.sort_values(by='Date', ascending=True).reset_index(drop=True)

        df2_doi = df_filtered.copy()
        df2_doi = df2_doi[df2_doi['Name'].isin(unique_list)].copy()
        df2_doi = df2_doi.drop(columns=['index','Composition','Confidence','Publisher'], errors= 'ignore')
        df2_filtered = df2_doi.groupby(['Date','Name','Value']).size().reset_index(name='Frequency').reset_index()
        #st.write(f"✅ From Elsevier, there are {len(df2_filtered)} results for {len(unique_list)} materials filtered: {', '.join(unique_list)}.")
        #st.write(f"✅ From Elsevier, there are {len(df2_filtered)} results for the materials filtered. Sample 5 results were shown below.")"
        
         # --- Controls row ---
        #col1, col2 = st.columns([1, 2])

        #with col1:
            # Random label fraction slider
        #    label_fraction = st.slider(
        #        "Label fraction",
        #        min_value=0.0, max_value=1.0, value=0.1, step=0.1,
        #        help="0.0 = no labels, 1.0 = label all points"
        #    )

        #with col2:
        #    st.write("")

        # --- Handle empty or missing data ---
        required_cols = {'Date', 'Value', 'Frequency', 'Name'}
        missing_cols = required_cols - set(df2_filtered.columns)
        if missing_cols:
            st.warning(f"Missing required columns: {', '.join(sorted(missing_cols))}")
            st.stop()

        if df2_filtered.empty:
            st.info("No materials found for the selected names.")
        else:
            # --- Clean dtypes ---
            df2_filtered['Date'] = pd.to_datetime(df2_filtered['Date'], errors='coerce')
            df2_filtered['Value'] = pd.to_numeric(df2_filtered['Value'], errors='coerce')
            df2_filtered['Frequency'] = pd.to_numeric(df2_filtered['Frequency'], errors='coerce')
            df2_filtered = df2_filtered.dropna(subset=['Date', 'Value', 'Frequency', 'Name'])

            # --- Prepare for plotting ---
            df2_plot = df2_filtered.sort_values('Date').copy()
            df2_plot['Label'] = df2_plot['Name'].astype(str)
            rng = np.random.default_rng(42)
            #df2_plot['show_label'] = rng.random(len(df2_plot)) < label_fraction
            df2_plot['show_label'] = rng.random(len(df2_plot)) < 0.25

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(9, 6))

            # Spectral regions
            ax.axhspan(0, 1.6,  color='yellow', alpha=0.10, label='Infrared (0–1.6 eV)')
            ax.axhspan(1.6, 3.26, color='green',  alpha=0.10, label='Visible (1.6–3.26 eV)')
            ax.axhspan(3.26, df2_plot['Value'].max(), color='red',    alpha=0.10, label='Ultraviolet (3.26–4.0 eV)')

            # Colors per material
            groups = df2_plot['Label'].unique().tolist()
            cmap = plt.cm.get_cmap('tab10' if len(groups) <= 10 else 'tab20', len(groups))

            for idx, g in enumerate(groups):
                gdata = df2_plot[df2_plot['Label'] == g]
                if gdata.empty:
                    continue
                ax.scatter(
                    gdata['Date'], gdata['Value'],
                    s=(gdata['Frequency'].clip(lower=1) * 20),
                    c=[cmap(idx)], alpha=0.75,
                    edgecolors='white', linewidth=0.5, label=g
                )
                for _, row in gdata[gdata['show_label']].iterrows():
                    ax.text(row['Date'] + pd.Timedelta(days=10),
                            row['Value'] + 0.05,
                            str(row['Name']),
                            fontsize=8, color='black', ha='left', va='center')

            # Style
            ax.set_xlabel("Publication Date")
            ax.set_ylabel("Bandgap Energy (eV)")
            ax.grid(True, linewidth=0.3, alpha=0.4)
            ax.legend(title='Material', loc='upper right', frameon=False, fontsize=8)

            st.pyplot(fig, clear_figure=True)

            #downloading the csv file
            #st.table(df2_doi.sample(5))
            # Safety in case df is small
            # --- Section: Scatter plot of recently researched materials ---
            st.markdown("***The table displays ten(10) sampled journals relating to the filtered semiconductors.***")

            n = min(10, len(df2_doi))

            # --- Shuffle button to refresh the sample ---
            if "sample_seed" not in st.session_state:
                st.session_state.sample_seed = 42

            if st.button("🔀 Shuffle sample"):
                st.session_state.sample_seed = random.randint(0, 10**9)

            #st.subheader("Random sample (5 rows)")
            st.table(df2_doi.sample(n=n, random_state=st.session_state.sample_seed))

            # --- Download full table as CSV ---
            @st.cache_data
            def to_csv_bytes(df: pd.DataFrame) -> bytes:
                return df.to_csv(index=False).encode("utf-8")

            csv_bytes = to_csv_bytes(df2_doi)

            st.download_button(
                label="⬇️ Download excel file as CSV",
                data=csv_bytes,
                file_name="bandgap-filtered.csv",
                mime="text/csv",
            )

    elif selected_page == "Decision-making Assistant":
            st.title("Decision-making Assistant")
            st.markdown("Facilitate semiconductor selection with advanced filtering and visualization")
            
            # Initialize session state
            if 'filters' not in st.session_state:
                st.session_state.filters = {}
            if 'initial_filter_name' not in st.session_state:
                st.session_state.initial_filter_name = None
            if 'initial_filters_only' not in st.session_state:
                st.session_state.initial_filters_only = {}
            if 'plot_x_col' not in st.session_state:
                st.session_state.plot_x_col = 'Bandgap'
            if 'plot_y_col' not in st.session_state:
                st.session_state.plot_y_col = 'Reserve (ton)'
            if 'excluded_elements' not in st.session_state:
                st.session_state.excluded_elements = []
            if 'additional_dynamic_filters' not in st.session_state:
                st.session_state.additional_dynamic_filters = []
            if 'filters_applied' not in st.session_state:
                st.session_state.filters_applied = False

            # APPLY ELEMENT EXCLUSION FILTER (if any)
            df_after_element_exclusion = filter_by_excluded_elements(df1, st.session_state.excluded_elements)
            
            # SECTION 1: INITIAL FILTERS & ELEMENT EXCLUSION
            # ELEMENT EXCLUSION SECTION
            st.markdown("### 1. Element Exclusion")
            
            # Using a URL for periodic table image
            periodic_table_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Simple_Periodic_Table_Chart-en.svg/1200px-Simple_Periodic_Table_Chart-en.svg.png"
            st.image(periodic_table_url, caption="Periodic Table of Elements", use_container_width=True)
 
            # Show info about element filtering (only if elements are excluded)
            if st.session_state.excluded_elements:
                removed_count = len(df1) - len(df_after_element_exclusion)
                st.info(f"🔬 Element filter active: Excluded {', '.join(sorted(st.session_state.excluded_elements))} | "
                        f"Removed {removed_count} materials | Showing {len(df_after_element_exclusion)} of {len(df1)} materials")
            
            # Get all unique elements from Element_1 through Element_7 columns
            all_elements = set()
            element_columns = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 
                            'Element_5', 'Element_6', 'Element_7']
            
            for col in element_columns:
                if col in df1.columns:
                    # Get unique elements from this column, excluding NaN values
                    elements = df1[col].dropna().unique()
                    all_elements.update(elements)
            
            # Remove empty strings if any
            all_elements = {elem for elem in all_elements if elem and str(elem).strip()}
            all_elements = sorted(list(all_elements))
            
            st.markdown("**Enter element symbols to exclude (separated by commas)**")

            # Text input for manual entry
            element_text_input = st.text_input(
                "Element symbols:",
                value=", ".join(st.session_state.excluded_elements) if st.session_state.excluded_elements else "",
                key="element_text_input",
                placeholder="e.g., Au, Ag, Si, Pb",
                help="Enter element symbols separated by commas. Spaces are optional."
            )
            
            # Parse the input
            if element_text_input.strip():
                # Split by comma and clean up
                selected_elements = [elem.strip() for elem in element_text_input.split(',') if elem.strip()]
            else:
                selected_elements = []
            
            # Show count of currently excluded elements
            if st.session_state.excluded_elements:
                st.markdown(f"**Currently Excluded Elements:** {len(st.session_state.excluded_elements)}")
            
            # Show impact preview for element exclusion
            if selected_elements != st.session_state.excluded_elements:
                preview_filtered = filter_by_excluded_elements(df1, selected_elements)
                would_remove = len(df1) - len(preview_filtered)
                if would_remove > 0:
                    st.warning(f"⚠️ Preview: This will remove {would_remove} materials ({would_remove/len(df1)*100:.1f}%) from the dataset.")
            
            # INITIAL FILTERS SECTION
            st.markdown("### 2. Initial Filters")
            cols = st.columns(2)

            with cols[0]:
                st.markdown("#### Bandgap Selection")
                col1, col2 = st.columns(2)
                with col1:
                    bandgap_min = st.number_input(
                        "Min (eV)",
                        min_value=0.0,
                        max_value=35.0,
                        value=0.0,
                        step=0.1,
                        key="bandgap_min"
                    )
                with col2:
                    bandgap_max = st.number_input(
                        "Max (eV)",
                        min_value=0.0,
                        max_value=35.0,
                        value=3.0,
                        step=0.1,
                        key="bandgap_max"
                    )

                # Validation
                if bandgap_min > bandgap_max:
                    st.error("Minimum bandgap must be less than or equal to maximum bandgap")
                    
                bandgap_range = (bandgap_min, bandgap_max)

            with cols[1]:
                st.markdown("#### Additional Filter")
                filter_options = [
                    'Reserve (ton)', 'Production (ton)', 'HHI (USGS)',
                    'ESG Score', 'CO2 footprint max (kg/kg)', 
                    'Embodied energy max (MJ/kg)', 'Water usage max (l/kg)', 
                    'Toxicity', 'Companionality'
                ]
                
                selected_filter = st.selectbox("Choose a filter", filter_options, key="selected_filter")
                
                # Show input controls only after a filter is selected
                if selected_filter:
                    # Get min and max values for the selected filter
                    temp_filtered = filter_by_excluded_elements(df1, selected_elements) if selected_elements else df1
                    filter_min = float(temp_filtered[selected_filter].min())
                    filter_max = float(temp_filtered[selected_filter].max())
                    
                    # Number input for Production and Reserve (minimum requirement in tonnes)
                    if selected_filter in ['Production (ton)', 'Reserve (ton)']:
                        filter_min_input = st.number_input(
                            f"Minimum Requirement (tonnes)",
                            min_value=filter_min,
                            max_value=filter_max,
                            value=filter_min,
                            step=1000.0,  # Step by 1000 tonnes for easier input
                            format="%.2f",
                            key="filter_min_input"
                        )
                        
                        filter_range = (filter_min_input, filter_max)
                        
                        # Display formatted value for readability
                        st.caption(f"**Minimum Required:** {format_tons(filter_min_input)}")
                    
                    # Integer slider for Toxicity
                    elif selected_filter == 'Toxicity':
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            int(filter_min),
                            int(filter_max),
                            (int(filter_min), int(filter_max)),
                            step=1,
                            key="initial_filter_slider"
                        )
                    
                    # Standard slider for other filters
                    else:
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            filter_min,
                            filter_max,
                            (filter_min, filter_max),
                            key="initial_filter_slider"
                        )
                else:
                    filter_range = None
            
            # DYNAMIC ADDITIONAL FILTERS SECTION
            st.markdown("### 3. Additional Filters (Optional)")
            
            # Get all available filter options
            all_filter_options = [
                'Reserve (ton)', 'Production (ton)', 'HHI (USGS)',
                'ESG Score', 'CO2 footprint max (kg/kg)', 
                'Embodied energy max (MJ/kg)', 'Water usage max (l/kg)', 
                'Toxicity', 'Companionality'
            ]
            
            # Exclude the initial filter that's already selected
            available_for_dynamic = [f for f in all_filter_options if f != selected_filter]
            
            # Button to add new filter
            col_add_btn, col_info = st.columns([1, 3])
            with col_add_btn:
                if st.button("➕ Add Filter", key="add_dynamic_filter"):
                    if len(st.session_state.additional_dynamic_filters) < len(available_for_dynamic):
                        st.session_state.additional_dynamic_filters.append({
                            'filter_name': None,
                            'filter_range': None
                        })
                        st.rerun()
            with col_info:
                st.caption(f"You can add up to {len(available_for_dynamic)} additional filters")
            
            # Display dynamic filters
            dynamic_filter_values = {}
            filters_to_remove = []
            
            for idx, filter_config in enumerate(st.session_state.additional_dynamic_filters):
                st.markdown(f"#### Filter #{idx + 2}")
                
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    # Get filters that are not already used
                    used_filters = [selected_filter] + [f['filter_name'] for f in st.session_state.additional_dynamic_filters if f['filter_name']]
                    available_options = [f for f in available_for_dynamic if f not in used_filters or f == filter_config.get('filter_name')]
                    
                    if available_options:
                        dynamic_filter_name = st.selectbox(
                            "Select filter",
                            options=available_options,
                            index=available_options.index(filter_config['filter_name']) if filter_config.get('filter_name') in available_options else 0,
                            key=f"dynamic_filter_name_{idx}"
                        )
                        filter_config['filter_name'] = dynamic_filter_name
                    else:
                        st.warning("No more filters available")
                        dynamic_filter_name = None
                
                with col2:
                    if dynamic_filter_name:
                        # Get min/max for the filter
                        temp_filtered = filter_by_excluded_elements(df1, selected_elements) if selected_elements else df1
                        dyn_filter_min = float(temp_filtered[dynamic_filter_name].min())
                        dyn_filter_max = float(temp_filtered[dynamic_filter_name].max())
                        
                        # Integer slider for Toxicity
                        if dynamic_filter_name == 'Toxicity':
                            dyn_filter_range = st.slider(
                                f"{dynamic_filter_name} Range",
                                int(dyn_filter_min),
                                int(dyn_filter_max),
                                (int(dyn_filter_min), int(dyn_filter_max)),
                                step=1,
                                key=f"dynamic_filter_range_{idx}"
                            )
                        # Formatted slider for Production and Reserve
                        elif dynamic_filter_name in ['Production (ton)', 'Reserve (ton)']:
                            dyn_filter_range = st.slider(
                                f"{dynamic_filter_name} Range",
                                dyn_filter_min,
                                dyn_filter_max,
                                (dyn_filter_min, dyn_filter_max),
                                format="",
                                key=f"dynamic_filter_range_{idx}"
                            )
                            st.caption(f"**Range:** {format_tons(dyn_filter_range[0])} to {format_tons(dyn_filter_range[1])}")
                        else:
                            dyn_filter_range = st.slider(
                                f"{dynamic_filter_name} Range",
                                dyn_filter_min,
                                dyn_filter_max,
                                (dyn_filter_min, dyn_filter_max),
                                key=f"dynamic_filter_range_{idx}"
                            )
                        
                        dynamic_filter_values[dynamic_filter_name] = dyn_filter_range
                
                with col3:
                    if st.button("🗑️", key=f"remove_filter_{idx}", help="Remove this filter"):
                        filters_to_remove.append(idx)
            
            # Remove filters marked for deletion
            if filters_to_remove:
                for idx in sorted(filters_to_remove, reverse=True):
                    st.session_state.additional_dynamic_filters.pop(idx)
                st.rerun()
            
            # SINGLE APPLY BUTTON FOR ELEMENT EXCLUSION, INITIAL FILTERS, AND DYNAMIC FILTERS
            if st.button("Apply Filters", key="apply_all_filters", type="primary"):
                if filter_range is not None:
                    # Apply element exclusion
                    st.session_state.excluded_elements = selected_elements
                    
                    # Combine initial filter with dynamic filters
                    all_filters = {
                        "Bandgap": bandgap_range,
                        selected_filter: filter_range
                    }
                    
                    # Add dynamic filters
                    all_filters.update(dynamic_filter_values)
                    
                    # Apply all filters
                    st.session_state.filters = all_filters
                    st.session_state.initial_filters_only = all_filters.copy()
                    st.session_state.initial_filter_name = selected_filter
                    st.session_state.plot_x_col = 'Bandgap'
                    st.session_state.plot_y_col = selected_filter
                    st.session_state.filters_applied = True
                    
                    filter_count = len(all_filters)
                    st.success(f"✅ {filter_count} filter(s) applied successfully!")
                    st.rerun()
                else:
                    st.warning("Please select an additional filter and set its range.")
            
            # GRAPH DISPLAY
            st.subheader("Filtered Results")
            
            # Determine which data to show
            if st.session_state.initial_filters_only:
                df_filtered = filter_dataframe(df_after_element_exclusion, st.session_state.initial_filters_only)
                
                # Show applied filters summary
                filter_summary = ", ".join([f"{k}" for k in st.session_state.initial_filters_only.keys()])
                st.info(f"🔄 Showing {len(df_filtered)} materials | Filters applied: {filter_summary} | Available: {len(df_after_element_exclusion)} (after element exclusion)")
            else:
                df_filtered = df_after_element_exclusion.copy()
                st.info(f"📈 Showing all {len(df_after_element_exclusion)} available materials (after element exclusion)")
            
            # Get current axes from session state
            x_col = st.session_state.plot_x_col
            y_col = st.session_state.plot_y_col
            
            # Plot configuration
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**X-axis:** {x_col}")
            with col2:
                st.write(f"**Y-axis:** {y_col}")
            with col3:
                log_y = st.checkbox(f"Log Y-axis", key="log_y_main")
            
            # Automatic plot title
            plot_title = f"{x_col} vs {y_col}"
            
            # Create and display graph
            if not df_filtered.empty:
                p = create_professional_plot(
                    df_filtered, x_col, y_col, plot_title, x_col, y_col, False, log_y
                )
                streamlit_bokeh(p, key="professional_plot")
            else:
                st.warning("⚠️ No materials match the current filters")

            # MCDM ANALYSIS SECTION

            if st.session_state.filters_applied and not df_filtered.empty:
                st.markdown("---")
                st.subheader("4. Multi-Criteria Decision Making")
                st.info(f"Analyze the {len(df_filtered)} filtered materials using TOPSIS or PROMETHEE methods")
                
                cols_mcdm = st.columns(2)
                with cols_mcdm[0]:
                    mcdm_method = st.selectbox(
                        "Method",
                        ["TOPSIS", "PROMETHEE"],
                        help="TOPSIS: Technique for Order Preference by Similarity to Ideal Solution\nPROMETHEE: Preference Ranking Organization Method for Enrichment Evaluation",
                        key="mcdm_method_custom"
                    )
                with cols_mcdm[1]:
                    weighting_method = st.radio(
                        "Weighting",
                        ["Entropy Weighting", "Manual Weights"],
                        horizontal=True,
                        key="mcdm_weighting_custom"
                    )
                
                # Criteria selection
                criteria_options = {
                    'Reserve (ton)': 1, 'Production (ton)': 1, 'HHI (USGS)': -1,
                    'ESG Score': -1, 'CO2 footprint max (kg/kg)': -1,
                    'Embodied energy max (MJ/kg)': -1, 'Water usage max (l/kg)': -1,
                    'Toxicity': -1, 'Companionality': -1
                }
                available_criteria = {k: v for k, v in criteria_options.items() if k in df_filtered.columns}
                
                # Check if we have any criteria available
                if not available_criteria:
                    st.error("❌ No criteria columns found in filtered data. Please ensure your data contains the required columns.")
                    st.stop()
                
                # Weight assignment
                if weighting_method == "Entropy Weighting":
                    # Check if we have enough samples for entropy weighting
                    if len(df_filtered) < 20:
                        st.warning(f"⚠️ Warning: Only {len(df_filtered)} materials available. Entropy weighting works best with 30+ samples. Consider using Manual Weights instead.")
                    
                    try:
                        # Get the matrix for entropy calculation
                        matrix_for_entropy = df_filtered[list(available_criteria.keys())].values
                        
                        # Check for NaN in the matrix
                        if np.isnan(matrix_for_entropy).any():
                            nan_count = np.isnan(matrix_for_entropy).sum()
                            st.error(f"❌ Cannot calculate entropy weights: {nan_count} missing values found in criteria columns.")
                            st.info("💡 Tip: Use Manual Weights instead, or filter out materials with missing values.")
                            weights = None
                        
                        # Check for negative values (entropy requires non-negative data)
                        elif np.any(matrix_for_entropy < 0):
                            st.error("❌ Cannot calculate entropy weights: Negative values found in criteria columns.")
                            st.info("💡 Entropy weighting requires non-negative values. Consider data transformation or use Manual Weights.")
                            weights = None
                        
                        else:
                            # SUM NORMALIZATION: Create probability distribution for entropy calculation
                            # This is the correct method according to Entropy Weight Method theory
                            n = matrix_for_entropy.shape[0]  # number of alternatives
                            m = matrix_for_entropy.shape[1]  # number of criteria
                            
                            # Initialize probability matrix
                            probability_matrix = np.zeros_like(matrix_for_entropy, dtype=float)
                            
                            # Sum normalization for each criterion
                            for j in range(m):
                                col_data = matrix_for_entropy[:, j]
                                col_sum = np.sum(col_data)
                                
                                # Check if column sum is zero or near-zero
                                if col_sum < 1e-10:
                                    st.warning(f"⚠️ All values in '{list(available_criteria.keys())[j]}' are zero or near-zero. Using equal distribution.")
                                    probability_matrix[:, j] = 1.0 / n  # Equal probability for all alternatives
                                else:
                                    # Sum normalization: creates probability distribution directly
                                    probability_matrix[:, j] = col_data / col_sum
                            
                            # Calculate entropy for each criterion
                            entropies = []
                            diversities = []
                            
                            for j in range(m):
                                p = probability_matrix[:, j]
                                
                                # Calculate entropy using information theory formula
                                # Add small epsilon to avoid log(0)
                                p_safe = np.where(p > 1e-10, p, 1e-10)
                                e_j = -np.sum(p_safe * np.log(p_safe)) / np.log(n)
                                
                                # Calculate diversity (degree of differentiation)
                                d_j = 1 - e_j
                                
                                entropies.append(e_j)
                                diversities.append(d_j)
                            
                            entropies = np.array(entropies)
                            diversities = np.array(diversities)
                            
                            # Calculate weights from diversities
                            diversity_sum = np.sum(diversities)
                            if diversity_sum > 1e-10:
                                weights = diversities / diversity_sum
                            else:
                                # All criteria have zero diversity (all alternatives are identical)
                                st.warning("⚠️ All criteria have identical values across alternatives. Using equal weights.")
                                weights = np.ones(m) / m
                            
                            # Validate the calculated weights
                            if weights is None or np.isnan(weights).any() or np.isinf(weights).any():
                                st.error("❌ Entropy weighting failed: Invalid weight values calculated.")
                                st.info("💡 This often happens with small datasets (<30 samples) or when criteria have very similar values.")
                                st.info("🔄 Falling back to equal weights for all criteria.")
                                weights = np.ones(len(available_criteria)) / len(available_criteria)
                                st.success(f"✅ Using equal weights: {1/len(available_criteria):.2%} for each criterion")
                            else:
                                # Check if weights are all very similar (within 5% of equal weight)
                                equal_weight = 1 / len(available_criteria)
                                max_deviation = np.max(np.abs(weights - equal_weight))
                                
                                if max_deviation < 0.05:
                                    st.warning(f"⚠️ Entropy weights are very similar (max deviation: {max_deviation*100:.2f}%)")
                                    st.info("This indicates that all criteria have similar information content in your filtered dataset.")
                                    st.info("💡 Consider using Manual Weights to emphasize specific criteria based on your domain knowledge.")
                                else:
                                    st.success("✅ Entropy weights calculated successfully")
                                    
                    except Exception as e:
                        st.error(f"❌ Error calculating entropy weights: {str(e)}")
                        st.info("🔄 Falling back to equal weights for all criteria.")
                        weights = np.ones(len(available_criteria)) / len(available_criteria)
                        st.success(f"✅ Using equal weights: {1/len(available_criteria):.2%} for each criterion")

                else:
                    st.markdown("**📊 Criteria Weights** - Assign importance (0–5 scale):")
                    
                    # Initialize preset weights storage
                    if 'preset_weights' not in st.session_state:
                        st.session_state.preset_weights = {col: 3 for col in available_criteria.keys()}
                    
                    # PRESET WEIGHT TEMPLATES
                    st.markdown("##### Quick Presets")
                    preset_cols = st.columns(3)
                    
                    with preset_cols[0]:
                        if st.button("Balanced", key="preset_balanced", help="Equal importance for all criteria"):
                            st.session_state.preset_weights = {col: 3 for col in available_criteria.keys()}
                            st.rerun()
                    
                    with preset_cols[1]:
                        if st.button("Long-term goal", key="preset_long_term", help="Focus on sustainability (ESG, reserves,toxicity, companionality)"):
                            st.session_state.preset_weights = {}
                            for col in available_criteria.keys():
                                if col in ['ESG Score', 'Toxicity', 'Companionality', 'Reserve (ton)']:
                                    st.session_state.preset_weights[col] = 5
                                else:
                                    st.session_state.preset_weights[col] = 1
                            st.rerun()
                    
                    with preset_cols[2]:
                        if st.button("Short-term goal", key="preset_short_term", help="Focus on availability (production, HHI, CO2 footprint, energy, water)"):
                            st.session_state.preset_weights = {}
                            for col in available_criteria.keys():
                                if col in ['Production (ton)', 'HHI (USGS)', 'CO2 footprint max (kg/kg)', 'Water usage max (l/kg)', 'Embodied energy max (MJ/kg)']:
                                    st.session_state.preset_weights[col] = 5
                                else:
                                    st.session_state.preset_weights[col] = 1
                            st.rerun()
                    
                    # MANUAL WEIGHT SLIDERS - Arranged in 2 rows
                    st.markdown("##### Adjust Individual Weights")
                    
                    weights = []
                    criteria_list = list(available_criteria.items())
                    
                    # Split criteria into two rows
                    mid_point = (len(criteria_list) + 1) // 2  # Round up for first row
                    
                    # First row
                    cols_row1 = st.columns(mid_point)
                    for i, (col, direction) in enumerate(criteria_list[:mid_point]):
                        with cols_row1[i]:
                            default_value = st.session_state.preset_weights.get(col, 3)
                            weight = st.slider(
                                f"{col}",
                                0, 5, 
                                value=default_value,
                                key=f"weight_custom_{col}",
                                help=f"{'Maximize' if direction == 1 else 'Minimize'} this criterion"
                            )
                            weights.append(weight)
                    
                    # Second row
                    if len(criteria_list) > mid_point:
                        cols_row2 = st.columns(len(criteria_list) - mid_point)
                        for i, (col, direction) in enumerate(criteria_list[mid_point:]):
                            with cols_row2[i]:
                                default_value = st.session_state.preset_weights.get(col, 3)
                                weight = st.slider(
                                    f"{col}",
                                    0, 5, 
                                    value=default_value,
                                    key=f"weight_custom_{col}",
                                    help=f"{'Maximize' if direction == 1 else 'Minimize'} this criterion"
                                )
                                weights.append(weight)
                    
                    # Normalize weights
                    if sum(weights) == 0:
                        st.warning("All weights set to 0 - using equal weights instead")
                        weights = np.ones(len(weights)) / len(weights)
                    else:
                        weights = np.array(weights) / sum(weights)
                
                # Display weights
                weights_df = pd.DataFrame({
                    'Criterion': list(available_criteria.keys()),
                    'Weight': weights,
                    'Direction': ['Maximize' if d == 1 else 'Minimize' for d in available_criteria.values()]
                }).sort_values('Weight', ascending=False).reset_index(drop=True)
                
                # Index will serve as rank (0-based, but display will show 1-based)
                weights_df.index = weights_df.index + 1
                weights_df.index.name = 'Rank'
                
                st.subheader("Criteria Weights")
                
                # Validate weights before displaying
                if weights is None:
                    st.error("❌ Error: Weights are None. Please check the weighting calculation above.")
                elif len(weights) == 0:
                    st.error("❌ Error: No weights calculated.")
                elif np.isnan(weights).any():
                    st.error("❌ Error: Some weights are NaN (Not a Number).")
                    st.dataframe(weights_df, use_container_width=True)
                else:
                    st.dataframe(
                        weights_df.style.format({'Weight': '{:.2%}'}),
                        use_container_width=True
                    )
                
                # Run analysis
                if st.button("🚀 Run MCDM Analysis", type="primary", key="run_mcdm_custom"):
                    with st.spinner("Performing analysis..."):
                        # Prepare data
                        matrix = df_filtered[list(available_criteria.keys())].values
                        types = np.array([available_criteria[k] for k in available_criteria])
                        
                        # Validate data - check for NaN values
                        if np.isnan(matrix).any():
                            nan_count = np.isnan(matrix).sum()
                            st.error(f"❌ Error: Found {nan_count} missing values (NaN) in the data. Please filter out materials with missing values or fill them.")
                            
                            # Show which columns have NaN
                            nan_cols = []
                            for col in available_criteria.keys():
                                if df_filtered[col].isna().any():
                                    nan_cols.append(f"{col} ({df_filtered[col].isna().sum()} missing)")
                            st.warning(f"Columns with missing values: {', '.join(nan_cols)}")
                            st.stop()
                        
                        # Validate weights
                        if weights is None or len(weights) == 0:
                            st.error("❌ Error: Weights are not defined. Please check weight calculation.")
                            st.stop()
                        
                        if np.isnan(weights).any():
                            st.error("❌ Error: Weights contain NaN values. Please check your data.")
                            st.stop()
                        
                        # Check if weights sum to 1 (approximately)
                        if not np.isclose(np.sum(weights), 1.0):
                            st.warning(f"⚠️ Weights sum to {np.sum(weights):.4f}, normalizing to 1.0")
                            weights = weights / np.sum(weights)
                        
                        # Run MCDM method
                        try:
                            if mcdm_method == "TOPSIS":
                                scores = run_topsis(matrix, weights, types)
                                
                                # Check if scores contain NaN
                                if np.isnan(scores).any():
                                    st.error("❌ TOPSIS returned NaN scores. This may be due to data issues.")
                                    st.write("Debug info:")
                                    st.write(f"- Matrix shape: {matrix.shape}")
                                    st.write(f"- Weights: {weights}")
                                    st.write(f"- Types: {types}")
                                    st.stop()
                                
                                # Create results dataframe and sort by score
                                results = pd.DataFrame({
                                    'Material': df_filtered['Name'].values,
                                    'Bandgap (eV)': df_filtered['Bandgap'].values,
                                    'Score': scores
                                }).sort_values('Score', ascending=False).reset_index(drop=True)
                                
                                # Use index as rank (1-based)
                                results.index = results.index + 1
                                results.index.name = 'Rank'
                                
                            else:
                                flows = run_promethee(matrix, weights, types)
                                
                                # Check if flows contain NaN
                                if np.isnan(flows).any():
                                    st.error("❌ PROMETHEE returned NaN flows. This may be due to data issues.")
                                    st.stop()
                                
                                # Create results dataframe and sort by net flow
                                results = pd.DataFrame({
                                    'Material': df_filtered['Name'].values,
                                    'Bandgap (eV)': df_filtered['Bandgap'].values,
                                    'Net Flow': flows
                                }).sort_values('Net Flow', ascending=False).reset_index(drop=True)
                                
                                # Use index as rank (1-based)
                                results.index = results.index + 1
                                results.index.name = 'Rank'
                                
                        except Exception as e:
                            st.error(f"❌ Error running {mcdm_method}: {str(e)}")
                            st.write("Debug information:")
                            st.write(f"- Number of materials: {len(df_filtered)}")
                            st.write(f"- Number of criteria: {len(available_criteria)}")
                            st.write(f"- Weights: {weights}")
                            st.stop()
                    
                    # Display results
                    st.subheader("MCDM Results")
                    st.dataframe(
                        results.style.format({
                            'Bandgap (eV)': '{:.2f}',
                            'Score': '{:.4f}',
                            'Net Flow': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Visualize top materials
                    st.subheader("🏆 Top Materials")
                    
                    # Get unique top materials (skip duplicates)
                    unique_top_materials = results.drop_duplicates(subset=['Material'], keep='first').head(3)
                    top_n = len(unique_top_materials)
                    
                    if top_n > 0:
                        cols_top = st.columns(top_n)
                        for i in range(top_n):
                            with cols_top[i]:
                                material = unique_top_materials.iloc[i]['Material']
                                # Get the actual rank from the index
                                rank_num = unique_top_materials.index[i]
                                bandgap = unique_top_materials.iloc[i]['Bandgap (eV)']
                                score_val = unique_top_materials.iloc[i]['Score'] if 'Score' in unique_top_materials.columns else unique_top_materials.iloc[i]['Net Flow']
                                st.metric(
                                    label=f"Rank #{rank_num}",
                                    value=material
                                )
                    else:
                        st.info("No materials to display")
                    
                    # Download results
                    excel_data = create_full_output(df_filtered, results, weights_df)
                    st.download_button(
                        label="📥 Download Full MCDM Report",
                        data=excel_data,
                        file_name=f"mcdm_analysis_{mcdm_method}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_mcdm_custom"
                    )

if __name__ == "__main__":
    main()