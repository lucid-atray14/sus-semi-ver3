import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
import streamlit as st
import numpy as np
from pymcdm.methods import PROMETHEE_II
from pymcdm.methods import TOPSIS
from io import BytesIO
from bokeh.transform import jitter
from streamlit_bokeh import streamlit_bokeh
import os

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
def load_bandgap_database():
    df = pd.read_excel("materials_with_composition.xlsx")
    return df.iloc[:, 1:]


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
    df1 = load_bandgap_database()
    
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
        
            # Filters section at the top in expandable containers
            specified_names = [
                        "TiO2","ZnO","CdS","MoS2","SnO2","ZnS","WO3","CuO","Cu2O","Si"
                    ]
        
            # Process data with distinct colors
            filtered_df = df1[df1['Name'].isin(specified_names)] if specified_names else df1
            
            # Plot section
            st.markdown(f"**Analysis Results ({len(filtered_df)} materials)**")
        
            source = ColumnDataSource(filtered_df)
            
            # --- Plot setup ---
            p = figure(
                x_range=filtered_df["Name"].unique().tolist(),
                width=600, height=500,
                toolbar_location=None,
                title=None
            )
            
            # --- Scatter points with jitter and color mapping ---
            p.circle(
                x=jitter("Name", width=0.3, range=p.x_range),
                y="Bandgap",
                source=source,
                size=10, 
                alpha=0.9,
                color="#66c2a5"
            )
            
            # --- Hover tool ---
            hover = HoverTool(tooltips=[("Bandgap (eV)", "@Bandgap")])
            p.add_tools(hover)
            
            # --- Legend configuration ---
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
            
            # --- Aesthetics ---
            p.xaxis.axis_label = "Semiconductor"
            p.yaxis.axis_label = "Bandgap (eV)"
            p.xgrid.visible = False
            p.ygrid.visible = True
            p.outline_line_color = None
        
            streamlit_bokeh(
                p,
                key="bandgap_plot"
            )
            
            
            # Data download
            st.download_button(
                label="📥 Download Analysis Data",
                data=filtered_df.to_csv(index=False).encode('utf-8'),
                file_name="bandgap_analysis.csv",
                mime="text/csv",
                use_container_width=True
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
                
                # Show slider only after a filter is selected
                if selected_filter:
                    # Get min and max values for the selected filter (use current element-filtered data if elements are being changed)
                    temp_filtered = filter_by_excluded_elements(df1, selected_elements) if selected_elements else df1
                    filter_min = float(temp_filtered[selected_filter].min())
                    filter_max = float(temp_filtered[selected_filter].max())
                    
                    # Integer slider for Toxicity
                    if selected_filter == 'Toxicity':
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            int(filter_min),
                            int(filter_max),
                            (int(filter_min), int(filter_max)),
                            step=1,
                            key="initial_filter_slider"
                        )
                    # Formatted slider for Production and Reserve
                    elif selected_filter in ['Production (ton)', 'Reserve (ton)']:
                        filter_range = st.slider(
                            f"{selected_filter} Range",
                            filter_min,
                            filter_max,
                            (filter_min, filter_max),
                            format="",
                            key="initial_filter_slider"
                        )
                        # Display formatted values
                        st.caption(f"**Selected Range:** {format_tons(filter_range[0])} to {format_tons(filter_range[1])}")
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
                                if col in ['Production (ton)', 'HHI (USGS)', 'CO2 footprint max (kg/kg)', 'Water usage max (l/kg)', 'Energy usage max (MJ/kg)']:
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
