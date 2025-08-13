# This dashboard is built with Streamlit to visualize and model ticket transfers and donation propensities
# using network analysis (Louvain communities), predictive modeling, and descriptive statistics.

# -------------------------
# Imports and Dependencies
# -------------------------
# Standard libraries, graph processing, visualization, ML tools, statistical modeling, and file handling
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import community.community_louvain as community_louvain
from collections import Counter
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------
# Optional Password Protection (Disabled)
# -------------------------
# This code block provides basic user authentication for deployment (currently commented out)
# def check_password():
#     def password_entered():
#         if st.session_state["password"] == st.secrets["password"]:
#             st.session_state["authenticated"] = True
#             del st.session_state["password"]  # Don't store password in session
#         else:
#             st.session_state["authenticated"] = False

#     if "authenticated" not in st.session_state:
#         st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
#         return False
#     elif not st.session_state["authenticated"]:
#         st.error("❌ Incorrect password")
#         return False
#     else:
#         return True

# # Only show the app if password is correct
# if not check_password():
#     st.stop()

# -------------------------
# Page Configuration
# -------------------------
# Set up the Streamlit app layout, icon, and title for a polished UI
st.set_page_config(page_title="Louvain Network Dashboard", page_icon="📊", layout="wide")


# -------------------------
# Custom Styling for UI Components
# -------------------------
# Adds CSS for card-based KPIs, typography, responsiveness and thematic branding
st.markdown("""
<style>
/* Titles */
h1 {
    color: #C8102E;   /* Paciolan Red */
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

/* Subheaders */
h2 {
    color: #C8102E;
    font-size: 28px;
    font-weight: 700;
    margin-top: 0px;
}

/* Section Headers */
h3 {
       color: ##000000;
    font-size: 24px;
    font-weight: 700;
    border-bottom: 2px solid #C8102E;
    padding-bottom: 5px;
    display: inline-block;
    margin-top: 30px;
}

/* KPI Container */
.kpi-container {
    display: flex;
    flex-wrap: wrap;         /* Allow wrapping on smaller screens */
    justify-content: center; /* Center align */
    gap: 20px;               /* Space between cards */
    margin: 20px 0;
}

/* KPI Cards */
.kpi-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(200,16,46,0.2);
    flex: 1 1 200px;         /* Flexible width with min 200px */
    max-width: 220px;        /* Prevent too wide cards */
    box-sizing: border-box;
}

/* KPI Value */
.kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #C8102E;
}

/* KPI Label */
.kpi-label {
    font-size: 14px;
    color: #000000;
    margin-top: 5px;
}

/* Responsive Adjustments */
@media (max-width: 600px) {
    .kpi-card {
        flex: 1 1 100%;     /* On small screens, cards take full width */
        max-width: none;
    }
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Main Application Logic Begins After This
# -------------------------
# The rest of the script loads and prepares datasets, sets up sidebar navigation,
# and defines functionality for three primary modules:
# 1. Descriptive Analysis - Exploratory charts and KPIs
# 2. Social Network Analysis - Louvain graph clustering and node-level metrics
# 3. Predictive Modeling - ML models to predict ticket transfers and donation propensity
# Each module is user-selectable from the sidebar.

# --- Load Data ---

# Path to the parent directory where "data" is
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
image_path = os.path.join(base_dir, "images", "WBB_full_network.png")
model_dir = os.path.join(base_dir, "models")

# Read primary transfer records and supporting lookup tables
# Now load files from data/
df = pd.read_csv(os.path.join(data_dir, 'Data1_WIS.csv')) # Main transfer dataset
activity_map = pd.read_csv(os.path.join(data_dir, 'Activity_WIS.csv'))  # Maps activity codes to names
df_monthly = pd.read_csv(os.path.join(data_dir, 'wis_activity_level_transfer_counts.csv')) # Monthly-level transfer counts

# Create mapping between season codes and names using explode (used later for labeling charts)
season_map = dict(zip(df['SEASONS_SENT'].explode(), df['SEASONS_SENT_NAME'].explode()))

# Load web engagement datasets
web_df = pd.read_csv(os.path.join(data_dir, "wis_webactivity.csv"))
click_df = pd.read_csv(os.path.join(data_dir, "wis_clickthrough.csv"))

# Prepare activity code to name mapping
activity_dict = dict(zip(activity_map['ACTIVITY_ID'], activity_map['NAME']))

# Convert comma-separated strings into Python lists
for col in ['ACTIVITIES', 'SEASONS_SENT', 'SEASONS_SENT_NAME']:
    df[col] = df[col].apply(lambda x: [i.strip() for i in x.split(',')] if pd.notnull(x) else [])

# --- Sidebar Navigation ---
# Provides user interface to toggle between different views of the app
from streamlit_option_menu import option_menu
with st.sidebar:
    selected = option_menu(
        menu_title = "Navigation",
        options = ['Descriptive Analysis', 'Social Network Analysis', 'Predictive Modeling'],
        icons = ['bar-chart', 'share', 'graph-up'],
        default_index = 0
    )


# -------------------------
# Segment 1: Descriptive Analysis
# -------------------------

if selected == 'Descriptive Analysis':

    # Convert string columns to lists
    for col in ['ACTIVITIES', 'SEASONS_SENT', 'SEASONS_SENT_NAME']:
        df[col] = df[col].apply(
            lambda x: [i.strip() for i in x.split(',')] if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
    
    # Extract years from TRANSFER_20XX columns (Descending Order)
    years = sorted([int(col.split('_')[1]) for col in df.columns if col.startswith('TRANSFERS_')], reverse=True)
    selected_year = st.sidebar.selectbox("Select Year", options=years)
    
    # Filter rows where TRANSFER_selected_year > 0
    transfer_col = f'TRANSFERS_{selected_year}'
    filtered_df = df[df[transfer_col] > 0]
    
    # Title and Dynamic Subheader
    st.markdown("# Descriptive Analysis of Ticket Transfers")
    st.subheader(f"Exploratory Data Analysis (EDA) for the Year {selected_year}")
    
    # Prepare Data for Top 10 Activities
    activity_counts = filtered_df['ACTIVITIES'].explode().value_counts().head(10).reset_index()
    activity_counts.columns = ['Activity', 'Count']
    
    # Map activity codes to full names
    activity_counts['Activity_Full'] = activity_counts['Activity'].map(activity_dict).fillna(activity_counts['Activity'])
    
    # Prepare Data for Top 10 Seasons
    season_counts = filtered_df['SEASONS_SENT'].explode().value_counts().head(10).reset_index()
    season_counts.columns = ['Season', 'Count']

    # Add full season names
    season_counts['Season_Full'] = season_counts['Season'].map(season_map).fillna(season_counts['Season'])

    # Display Charts in 2 Columns
    col1, col2 = st.columns(2)
    
    with col1:
        if not activity_counts.empty:
            fig_activity = px.bar(
                activity_counts,
                x='Activity',
                y='Count',
                title='Top Activities by Volume of Transfers',
                text='Count',
                color='Activity_Full'
            )
            fig_activity.update_traces(width=0.6, textposition='outside')
            fig_activity.update_layout(showlegend=True, xaxis_title="Activity")
            # Adjust layout for better spacing
            fig_activity.update_layout(
                showlegend=True,
                height=500,                # Increase height as needed
                xaxis_title="Activity Code",
                legend_title="Activity Name"
            )

            st.plotly_chart(fig_activity, use_container_width=True)
        else:
            st.info("No activity data for selected year.")
    
    with col2:
        if not season_counts.empty:
            fig_seasons = px.bar(
                season_counts,
                x='Season',
                y='Count',
                title='Top Seasons by Volume of Transfers',
                text='Count',
                color='Season_Full'
            )
            
            fig_seasons.update_traces(width=0.6, textposition='outside')
            fig_seasons.update_layout(showlegend=True, height=500, xaxis_title="Season Code", legend_title="Season Name")
            st.plotly_chart(fig_seasons, use_container_width=True)
        else:
            st.info("No season data for selected year.")
    
    # KPI Summary Cards   
    # Calculate KPIs
    st.markdown("### Key KPIs")
    st.write("")   # Adds a small vertical space
    st.write("")   # Adds a small vertical space
    total_transfers = filtered_df[transfer_col].sum()
    unique_senders = filtered_df['SENDER_DIM_ACCOUNT_KEY'].nunique()
    unique_recipients = filtered_df['RECEPIENT_DIM_ACCOUNT_KEY'].nunique()
    
    activity_counts_total = filtered_df['ACTIVITIES'].explode().value_counts()
    top_activity = activity_counts_total.idxmax() if not activity_counts_total.empty else "No Data"
    top_activity_name = activity_dict.get(top_activity, top_activity)
    
    avg_transfers = round(total_transfers / unique_senders, 2) if unique_senders > 0 else 0
    
    # Layout for KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_transfers:,}</div>
                <div class="kpi-label">🔄 Total Transfers</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{unique_senders}</div>
                <div class="kpi-label">📨 Unique Senders</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{unique_recipients}</div>
                <div class="kpi-label">👥 Unique Recipients</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{top_activity_name}</div>
                <div class="kpi-label">🏆 Top Activity</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{avg_transfers}</div>
                <div class="kpi-label">📊 Avg Transfers per Sender</div>
            </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # Yearly & Monthly Transfer Trends
    # -------------------------    
    # Activity Trends Over Years
    st.write("")   # Adds a small vertical space
    st.write("")   # Adds a small vertical space
    st.markdown("### Activity Trends Over Years")
    
    #Identify Top 10 Activities Overall
    top_activities = df['ACTIVITIES'].explode().value_counts().head(10).index.tolist()
    
    #Prepare Transfer Columns
    transfer_cols = [col for col in df.columns if col.startswith('TRANSFERS_')]
    
    #Explode activities for proper mapping
    df_exploded = df.explode('ACTIVITIES')
    
    #Filter dataset for only Top 10 Activities
    df_top = df_exploded[df_exploded['ACTIVITIES'].isin(top_activities)]
    
    #Melt the dataframe to long format
    df_melted = df_top.melt(
        id_vars=['ACTIVITIES'],
        value_vars=transfer_cols,
        var_name='Year',
        value_name='Transfers'
    )
    
    # Clean Year column (convert 'TRANSFERS_2023' → 2023)
    df_melted['Year'] = df_melted['Year'].str.split('_').str[1].astype(int)
    
    #Group by Activity and Year
    activity_trends = df_melted.groupby(['ACTIVITIES', 'Year'])['Transfers'].sum().reset_index()
    activity_trends['Activity_Full'] = activity_trends['ACTIVITIES'].map(activity_dict)
    
    fig_trend = px.line(
        activity_trends,
        x='Year',
        y='Transfers',
        color='Activity_Full',
        title='Transfer Trends Over Years 2020 - 2024',
        markers=True
    )
    
    fig_trend.update_layout(
        title_font_size=22,
        title_font_color='#C8102E',
        xaxis_title='Year',
        yaxis_title='Total Transfers',
        height=500,
        legend_title="Activity",
        xaxis=dict(
            title='Year',
            type='category'
        ),
        title_font=dict(
            size=16,                # Same size as bar chart title
            color='black',          # Standard black color
            family='Arial'
        )

    )
    
    # Create two columns: 60% width for chart, 40% for summary
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div style='
            background-color:#ffffff;
            padding:20px;
            border-radius:10px;
            box-shadow:2px 2px 15px rgba(200,16,46,0.3);
            width: 70%;
            margin-left: auto;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        '>
            <h4 style='color:#C8102E; text-align:center;'>EDA Summary Insights</h4>
            <ul style='line-height:1.8; font-size:14px;'>
                <li><b style='color:#E75454;'>Football</b> Shows the steepest growth among all activities. Transfers rose from nearly zero in 2020 to over 40,000 by 2024. Indicates strong and rapidly increasing fan engagement or secondary market activity.</li>
                <li><b style='color:#E75454;'>Men’s Basketball</b> Demonstrates consistent year-on-year growth, crossing 15,000 transfers by 2024. Likely benefits from a large and engaged base, though growth has plateaued slightly compared to basketball.</li>
                <li><b style='color:#E75454;'>Women’s Sports (Basketball, Hockey)</b> Women’s Basketball and Women’s Hockey show modest but steady gains. Indicate niche but growing fan bases; potential for targeted engagement campaigns.</li>
                <!-- <li>Peak transfer volume in <b style='color:#E75454;'>2023</b>.</li> -->
                <li>Post-2021 boom across most activities suggests digital adoption or return of live events after initial pandemic disruption. Emerging growth in women’s sports presents an opportunity for inclusion-focused outreach.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    # Monthly Transfer Trends Section for selected activities
    st.markdown("### Monthly Trends for Top Activities")

    # Convert year-month columns to datetime
    df_monthly['YearMonth'] = pd.to_datetime(
        df_monthly['TRANSFER_YEAR'].astype(str) + '-' + df_monthly['TRANSFER_MONTH'].astype(str).str.zfill(2)
    )

    # Map full names to activity codes
    df_monthly['Activity_Full'] = df_monthly['ACTIVITY_CODE'].map(activity_dict).fillna(df_monthly['ACTIVITY_CODE'])
    
    # Auto-select top 3 most transferred activities for default plot
    top_3_activities = (df_monthly.groupby('Activity_Full')['TOTAL_TRANSFERS'].sum().sort_values(ascending=False).head(3).index.tolist())

    # Sidebar multiselect for user to pick activities
    selected_monthly_activities = st.multiselect(
        "Select Activities for Monthly Trend",
        options=sorted(df_monthly['Activity_Full'].unique()),
        default=top_3_activities
    )
    
    # Filter and plot monthly trends
    monthly_filtered = df_monthly[df_monthly['Activity_Full'].isin(selected_monthly_activities)]

    # Create the line chart
    fig_monthly = px.line(
        monthly_filtered,
        x='YearMonth',
        y='TOTAL_TRANSFERS',
        color='Activity_Full',
        title='Monthly Ticket Transfer Trends by Activity',
        markers=True
    )
    
    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Transfers",
        height=500,
        legend_title='Activity_Full',
        title_font_size=22,
        title_font_color='#C8102E'
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)

# -------------------------
# Segment 2: Social Network Analysis (Louvain Clustering)  on subsets of data based on selected Activity
# -------------------------

elif selected == 'Social Network Analysis':
    st.title("Social Network Analysis")

    # Expandable explainer section
    with st.expander("ℹ️ Understanding Network Graphs", expanded=False):
        col1, col2 = st.columns([2, 1])
    
        with col1:
            st.markdown("""
            - **Color (Viridis Scale):** Each color represents a **community** detected by the Louvain algorithm.  
            - **Nodes:** Each dot represents a **unique account** (Sender or Recipient).  
            - **Edges:** Lines indicate **ticket transfers** between accounts.  
            - **Graph Layout:** Uses *spring layout* to visually separate clusters.  
            - Larger clusters indicate tightly connected groups—often friends, family, or frequent trading partners.  
    
            **Example Insights:**
            - Densely packed areas = High interaction communities.  
            - Sparse nodes = Occasional or isolated transfers.   
            """)
    
        with col2:
            st.image(image_path, use_container_width=True, caption="Sample Louvain Community Graph")

    st.write("")   # Adds a small vertical space

    ##Preparing Data for Network Creation
    #------------------------------------
    # Merge all unique users from sender and recipient roles
    # Extract sender and recipient info
    senders = df[['SENDER_DIM_ACCOUNT_KEY', 'SENDER_NAME']].rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node', 'SENDER_NAME': 'NAME'})
    recipients = df[['RECEPIENT_DIM_ACCOUNT_KEY', 'RECEPIENT_NAME']].rename(columns={'RECEPIENT_DIM_ACCOUNT_KEY': 'node', 'RECEPIENT_NAME': 'NAME'})

    # Combine and drop duplicates
    user_info = pd.concat([senders, recipients]).drop_duplicates(subset='node')

    ##Creating the network graphs for selected activities
    # Load feature dataset (used later for node-level attributes)
    df_features = pd.read_csv(os.path.join(data_dir,"final_model_dataset_WIS.csv"))

    ##Build the Network Graph
    #------------------------
    @st.cache_resource(show_spinner=False)
    # Function to create directed graph from sender→recipient edges
    def build_graph(filtered_df):        
        G_directed = nx.from_pandas_edgelist(
            filtered_df,
            source='SENDER_DIM_ACCOUNT_KEY',
            target='RECEPIENT_DIM_ACCOUNT_KEY',
            edge_attr='TOTAL_TRANSFERS',
            create_using=nx.DiGraph()
        )

        # Run Louvain on undirected version to detect communities
        partition = community_louvain.best_partition(G_directed.to_undirected())
        nx.set_node_attributes(G_directed, partition, 'community')

        # Compute out-degree for each node
        out_degree = dict(G_directed.out_degree(weight='TOTAL_TRANSFERS'))
        nx.set_node_attributes(G_directed, out_degree, 'out_degree')

        # Assign community size to each node
        community_sizes = Counter(partition.values())
        nx.set_node_attributes(G_directed, {node: community_sizes[comm] for node, comm in partition.items()}, 'community_size')
        return G_directed, partition, out_degree, community_sizes
        
    ##Activity Filter + Community Selection
    #------------------------
    all_activities = sorted({activity for sublist in df['ACTIVITIES'] for activity in sublist})
    selected_activities = st.sidebar.multiselect("Select Activities", options=all_activities, default=all_activities[:2])
    filtered_df = df[df['ACTIVITIES'].apply(lambda acts: any(act in acts for act in selected_activities))]

    full_names = [activity_dict.get(code, code) for code in selected_activities]
    act_display = ", ".join(full_names)

    st.markdown(f"""
    <h3>
    Filtered Dataset for <span style='color:#d62728'>{act_display}</span>: 
    <span style='color:green; font-size:22px;'>{filtered_df.shape[0]:,}</span> rows
    </h3>
    """, unsafe_allow_html=True)

    
    if filtered_df.empty:
        st.warning("No data matches the selected activities.")
        
##Activity Filter + Community Selection
#------------------------
    else:
        # --- Compute Network ---
        G_directed, partition, out_degree, community_sizes = build_graph(filtered_df)

        total_communities = len(set(partition.values()))
        st.sidebar.markdown(f"**Total Communities Detected:** {total_communities}")

        # Determine community slider boundaries
        sorted_community_sizes = sorted(set(community_sizes.values()))
        max_comm_size = max(sorted_community_sizes)
        default_comm_size = min(5, max_comm_size)
    
        min_comm_size = st.sidebar.slider(
            "Minimum Community Size",
            min_value=min(sorted_community_sizes),
            max_value=max_comm_size,
            value=default_comm_size,
            step=1
        )

        # Select community
        valid_communities = sorted([comm for comm, size in community_sizes.items() if size >= min_comm_size])
    
        if not valid_communities:
            st.warning("No communities match the minimum size filter.")
        else:
            selected_community = st.sidebar.selectbox("Select Community ID", options=valid_communities)
            nodes_in_comm = [n for n in G_directed.nodes if partition[n] == selected_community]
            max_out_degree = max([out_degree[n] for n in nodes_in_comm]) if nodes_in_comm else 0
            out_degree_thresh = st.sidebar.slider("Minimum Out-Degree", 0, int(max_out_degree), 0)
    
            final_nodes = [n for n in nodes_in_comm if out_degree[n] >= out_degree_thresh]
            G_sub = G_directed.subgraph(final_nodes).copy()

            # --- Compute Centrality Metrics & Build Node Table ---
            # Existing node data
            node_data = [{'node': n, **d} for n, d in G_sub.nodes(data=True)]
            node_df = pd.DataFrame(node_data)
            
            # Merge with user names
            node_df = node_df.merge(user_info, on='node', how='left')
            
            # Add community and out-degree from global partition/out_degree
            node_df['community'] = node_df['node'].map(partition)
            node_df['out_degree'] = node_df['node'].map(out_degree)

            # Compute centrality measures
            degree_centrality = nx.degree_centrality(G_sub)
            betweenness_centrality = nx.betweenness_centrality(G_sub)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G_sub, max_iter=500)
            except nx.PowerIterationFailedConvergence:
                eigenvector_centrality = {node: None for node in G_sub.nodes()}
                        
            # Map them to dataframe
            node_df['Degree Centrality'] = node_df['node'].map(degree_centrality)
            node_df['Betweenness'] = node_df['node'].map(betweenness_centrality)
            node_df['Eigenvector'] = node_df['node'].map(eigenvector_centrality)
            # node_df = node_df.merge(df_features, left_on='node', right_on='SENDER_DIM_ACCOUNT_KEY', how='left')

            # --- Community Size Distribution with Plotly ---
            st.write("#### Community Size Distribution over all communities for selected activities")
            # Prepare data
            community_size_df = pd.DataFrame(list(community_sizes.items()), columns=["Community ID", "Size"])
            filtered_df_sizes = community_size_df[community_size_df['Size'] >= min_comm_size]
            
            def size_bucket(size):
                if size <= 5:
                    return 'Very Small (1–5)'
                elif size <= 20:
                    return 'Small (6–20)'
                elif size <= 50:
                    return 'Medium (21–50)'
                elif size <= 100:
                    return 'Large (51–100)'
                else:
                    return 'Very Large (100+)'
            
            community_size_df['Size Bucket'] = community_size_df['Size'].apply(size_bucket)
            bucket_counts = community_size_df['Size Bucket'].value_counts().reset_index()
            bucket_counts.columns = ['Community Size Bucket', 'Count']
            
            # Layout split
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_bar = px.bar(
                    bucket_counts,
                    x='Count',
                    y='Community Size Bucket',
                    orientation='h',
                    title="Community Size Bucket Distribution",
                    color='Community Size Bucket',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
            
                fig_bar.update_layout(
                    showlegend=False,
                    height=400,
                    width=500,  # compress width
                    xaxis_title="Number of Communities",
                    yaxis_title="Community Size Bucket"
                )
                fig_bar.update_traces(width=0.5)
                st.plotly_chart(fig_bar)
            
            with col2:
                st.markdown("""
                <div style="background-color:#f9f9f9; padding: 20px; border-radius: 10px; border-left: 6px solid #e45756;">
                    <h4 style="margin-top:0;">Community Size Categories</h4>
                    <ul>
                        <li><b>Very Small (1–5)</b>: Micro clusters, often isolated or new users.</li>
                        <li><b>Small (6–20)</b>: Tight-knit user groups with limited spread.</li>
                        <li><b>Medium (21–50)</b>: Balanced communities with moderate interaction.</li>
                        <li><b>Large (51–100)</b>: Well-connected groups showing strong participation.</li>
                        <li><b>Very Large (100+)</b>: Highly active communities, possibly containing influential nodes.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # --- Community Summary Table ---
            st.write("")
            st.subheader("Filtered Dataset Summary Metrics")
            st.markdown("This table summarizes key metrics for each community for the filtered activity")

            # Load supplemental data if not already done
            age_df = pd.read_csv(os.path.join(data_dir, "WIS_age.csv"))
            age_df.rename(columns={'ID': 'node'}, inplace=True)
            don_df = pd.read_csv(os.path.join(data_dir, "WIS_donations.csv"))
            don_df.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'}, inplace=True)
            click_df = pd.read_csv(os.path.join(data_dir, "WIS_clickthrough.csv"))
            click_df.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'}, inplace=True)

            # Merge demographic and donation data
            node_data = [{'node': n, **G_directed.nodes[n]} for n in G_directed.nodes()]
            merged = pd.DataFrame(node_data)
            merged = merged.merge(age_df[['node', 'AGE']], on='node', how='left')
            merged = merged.merge(don_df, on='node', how='left')
            merged = merged.merge(click_df[['node', 'TOTAL_CLICKTHROUGHS']], on='node', how='left')
            merged['AGE'] = pd.to_numeric(merged['AGE'], errors='coerce')
            merged.fillna(0, inplace=True)

            community_summary = merged.groupby('community').agg({
                'out_degree': lambda x: x[x != 0].mean(),
                'community_size': 'first',
                'TOTAL_CLICKTHROUGHS': lambda x: x[x != 0].mean(),
                'AGE': lambda x: x[x != 0].mean(),
                'MAX_LIFETIME_DONATION_AMOUNT': lambda x: x[x != 0].mean(),
                'AVG_LIFETIME_DONATION_AMOUNT': lambda x: x[x != 0].mean(),
                'YEARS_OF_DONATING': lambda x: x[x != 0].mean()
            }).reset_index()

            community_summary.columns = [
                'Community', 'Avg Out-Degree', 'Community Size', 'Avg Clickthroughs', 'Avg Age',
                'Avg Max Donation', 'Avg Donation', 'Avg Years Donating'
            ]
            community_summary.fillna(0, inplace=True)
            styled_summary = community_summary.style.background_gradient(cmap='Blues')
            st.dataframe(styled_summary)
            st.write("")
            st.write("")


            # Get the community size from any node in the subgraph
            if len(G_sub.nodes) > 0:
                first_node = list(G_sub.nodes)[0]
                community_size = G_sub.nodes[first_node]['community_size']
            else:
                community_size = 0
                
            st.write(f"### Filtered Dataset: Community - {selected_community} &nbsp;&nbsp; "
                         f"<span style='color:green; font-size:20px;'>({community_size} nodes)</span>",
                         unsafe_allow_html=True)
            
            st.write("#### Louvain Network Graphs and Observations")
            
            if len(G_sub.nodes) == 0:
                st.warning("No nodes match the selected filters.")
            else:
                # --- Layout: Network Graph + Geo Plot ---
                col1, col2 = st.columns(2)
    
                with col1:
                    st.write("#### Network Graph")
                    pos = nx.spring_layout(G_sub, seed=42, k=0.5)
                    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
                    for node in G_sub.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        comm = G_sub.nodes[node]['community']
                        od = G_sub.nodes[node]['out_degree']
                        node_color.append(comm)
                        node_size.append(5 + (od / max_out_degree * 20) if max_out_degree > 0 else 5)
                        node_text.append(f"Node: {node}<br>Out-Degree: {od}<br>Community Size: {G_sub.nodes[node]['community_size']}")

                    # Ensure node_df has all necessary info: node, NAME, community, out_degree
                    node_info = node_df.set_index('node')
                    
                    node_text = [
                            f"<b>Node ID:</b> {n}<br><b>Name:</b> {row['NAME']}<br><b>Community:</b> {row['community']}<br><b>Out-Degree:</b> {row['out_degree']}"
                            for n, row in node_info.loc[list(G_sub.nodes())].iterrows()
                        ]
    
                    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers',
                                            marker=dict(size=node_size, color=node_color, colorscale='Viridis', showscale=False),
                                            text=node_text, hoverinfo='text', name='Nodes')
    
                    edge_x, edge_y = [], []
                    for src, tgt in G_sub.edges():
                        x0, y0 = pos[src]
                        x1, y1 = pos[tgt]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]
    
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines',name='Edges')
                    fig = go.Figure(data=[edge_trace, node_trace])
                    fig.update_layout(
                        title='',
                        title_x=0.5,
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        plot_bgcolor='white',
                        height=500,   # Reduce height
                        width=700,    # Reduce width
                        margin=dict(l=20, r=20, t=40, b=20),
                        dragmode="zoom"
                    )
                    
                    fig.update_traces(marker=dict(line=dict(width=0)))

                    # Update colorbar title for nodes
                    for trace in fig.data:
                        if trace.mode == 'markers':
                            trace.marker.colorbar.title = 'Community ID'
                    
                    # Display
                    st.plotly_chart(fig, use_container_width=False)
    
                with col2:
                    st.write("#### Geographic Node Distribution")

                    # --- Load Geo Data ---
                    node_zip_df = pd.read_csv(os.path.join(data_dir, 'node_zipcodes_wis.csv'))   # Contains: node_id, ZIP_CODE
                    zip_latlon_df = pd.read_csv(os.path.join(data_dir, 'us_zip_latlon.csv')) # Contains: ZIP_CODE, LATITUDE, LONGITUDE
                    
                    # Ensure ZIP_CODE columns are strings
                    node_zip_df['ZIP_CODE'] = node_zip_df['ZIP_CODE'].astype(str)
                    zip_latlon_df['ZIP_CODE'] = zip_latlon_df['ZIP_CODE'].astype(str)
                    
                    # Merge ZIP codes with lat/lon
                    geo_nodes = pd.merge(node_zip_df, zip_latlon_df, on='ZIP_CODE', how='left')
                    geo_nodes = geo_nodes.dropna(subset=['LATITUDE', 'LONGITUDE'])
                    
                    # --- Filter nodes present in current graph ---
                    geo_nodes_filtered = geo_nodes[geo_nodes['node_id'].isin(final_nodes)]
                    
                    # --- Merge Graph Attributes ---
                    attributes = pd.DataFrame.from_dict(dict(G_sub.nodes(data=True)), orient='index').reset_index()
                    attributes.rename(columns={'index': 'node_id'}, inplace=True)
                    
                    geo_plot_df = pd.merge(geo_nodes_filtered, attributes, on='node_id', how='inner')
                    
                    # --- Prepare Marker Size & Color ---
                    geo_plot_df['SIZE'] = geo_plot_df['out_degree'].apply(lambda x: max(8, min(x * 2, 20)))  # Size between 8 and 20
                    geo_plot_df['COLOR'] = 'red'  # Fixed color; can be dynamic if needed
                    
                    # --- Prepare Hover Text ---
                    geo_plot_df = geo_plot_df.merge(user_info, left_on='node_id', right_on='node', how='left')  # Get NAME, EMAIL
                    
                    geo_plot_df['HOVER_TEXT'] = geo_plot_df.apply(lambda row: 
                        f"<b>Node ID:</b> {row['node_id']}<br>"
                        f"<b>Name:</b> {row.get('NAME', 'N/A')}<br>"
                        f"<b>Community:</b> {row['community']}<br>"
                        f"<b>Out-Degree:</b> {row['out_degree']}", axis=1)
                    
                    # --- Build Plotly ScatterGeo ---
                    fig = go.Figure(go.Scattergeo(
                        lon = geo_plot_df['LONGITUDE'],
                        lat = geo_plot_df['LATITUDE'],
                        mode = 'markers',
                        marker = dict(
                            size = geo_plot_df['SIZE'],
                            color = geo_plot_df['COLOR'],
                            opacity = 0.7,
                            line = dict(width=1, color='black')
                        ),
                        text = geo_plot_df['HOVER_TEXT'],
                        hoverinfo = 'text'
                    ))
                    
                    fig.update_layout(
                        title = '',
                        geo = dict(
                            scope = 'usa',
                            showland = True,
                            landcolor = 'rgb(240, 240, 240)',
                            showcountries = False,
                            showlakes = False
                        ),
                        margin={"r":0,"t":0,"l":0,"b":0},
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    
                # --- Data Outputs ---                
                # --- Node Summary Table with Explanation ---
                st.write("")
                st.write("#### Node Summary Table")
                # st.write(f"#### Node Summary Table – Community {selected_community} &nbsp;&nbsp; "
                #          f"<span style='color:green; font-size:20px;'>({community_size} nodes)</span>",
                #          unsafe_allow_html=True)
                
                # Rename for clarity
                node_df = node_df.rename(columns={
                    "node": "DIM_ACCOUNT_KEY",
                    "NAME": "Name",
                    "out_degree": "Out-Degree",
                    "Degree Centrality": "Degree Centrality",
                    "Betweenness": "Betweenness Centrality"
                })
                
                # Select only relevant columns
                display_cols = ['DIM_ACCOUNT_KEY', 'Name', 'Out-Degree', 'Degree Centrality', 'Betweenness Centrality']
                
                # Layout: Table on left (2/3) and Info Box on right (1/3)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(node_df[display_cols].sort_values(by='Out-Degree', ascending=False).reset_index(drop=True))
                with col2:
                    st.markdown("""
                            <div style="background-color: #f9f9f9; border-left: 5px solid #f63366; padding: 16px; border-radius: 8px;">
                            <h4> Centrality Metrics Explained</h4>
                            <ul>
                            <li><strong>Out-Degree:</strong> — Number of total ticket transfers a user/node has made.</li>
                            <li><strong>Degree Centrality:</strong> — Influence based on total direct connections in the network.</li>
                            <li><strong>Betweenness Centrality:</strong> — Frequency of being on the shortest paths between others (acts as a bridge).</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                # Download option
                st.download_button("Download Node Summary CSV", node_df[display_cols].to_csv(index=False), "node_summary.csv")
                st.write("")
                
                # Miniature Network Graph of Nodes with Non-Zero Betweenness
                nonzero_bt_nodes = [n for n, val in betweenness_centrality.items() if val > 0]
                if nonzero_bt_nodes:
                    G_bt = G_sub.subgraph(nonzero_bt_nodes).copy()
                    # Keep only the largest connected component
                    pos_bt = nx.spring_layout(G_bt, seed=42, k=0.3, scale=1.5)
                
                    node_x, node_y, node_size, node_hover = [], [], [], []
                    for node in G_bt.nodes():
                        x, y = pos_bt[node]
                        node_x.append(x)
                        node_y.append(y)
                        od = G_bt.nodes[node].get('out_degree', 1)
                        bt = betweenness_centrality[node]
                        name = node_df[node_df['DIM_ACCOUNT_KEY'] == node]['Name'].values[0] if node in node_df['DIM_ACCOUNT_KEY'].values else node
                        node_size.append(8 + od)  # scale out-degree
                        node_hover.append(f"<b>Name:</b> {name}<br><b>Out-Degree:</b> {od}<br><b>Betweenness:</b> {bt:.5f}")
                
                    node_trace_bt = go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode='markers',
                        marker=dict(size=node_size, color='orange', line=dict(color='black', width=1)),
                        text=node_hover,
                        hoverinfo='text',
                        name='Betweenness Nodes'
                    )
                
                    edge_x, edge_y = [], []
                    for src, tgt in G_bt.edges():
                        x0, y0 = pos_bt[src]
                        x1, y1 = pos_bt[tgt]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]
                
                    edge_trace_bt = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#999'),
                        hoverinfo='none',
                        mode='lines',
                        name='Edges'
                    )
                
                    fig_bt = go.Figure(data=[edge_trace_bt, node_trace_bt])
                    fig_bt.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        plot_bgcolor='white'
                    )
                    col1, col2 = st.columns([2, 1])

                    
                    with col1:
                        st.write("#### Mini Network: Nodes with Non-Zero Betweenness")
                        st.plotly_chart(fig_bt, use_container_width=True)
                    
                    with col2:
                        st.markdown("""
                            <div style="background-color: #f9f9f9; border-left: 5px solid #f63366; padding: 16px; border-radius: 8px;">
                            <h4> Betweenness Insights</h4>
                            <ul>
                            <li>Nodes shown have <strong>non-zero betweenness</strong> — they lie on shortest paths between others.</li>
                            <li>These users may act as <strong>bridges</strong> between communities.</li>
                            <li>Node size reflects <strong>out-degree</strong>.</li>
                            <li>Hover to see name, betweenness score, and out-degree.</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No nodes in this community have non-zero betweenness centrality.")

             # --- Load supplemental data for engagement ---
            click_df = pd.read_csv(os.path.join(data_dir, "WIS_clickthrough.csv"))
            web_df = pd.read_csv(os.path.join(data_dir, "WIS_webactivity.csv"))

            # Rename keys to match join expectations
            click_df.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'}, inplace=True)
            web_df.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'}, inplace=True)

            st.write("")
            st.write("#### Community-specific Engagement Visualizations")

            # Merge in clickthrough and web visit data
            engagement_df = pd.merge(click_df, web_df, on='node', how='outer')
            node_data = [{'node': n, **G_directed.nodes[n]} for n in G_directed.nodes()]
            node_df = pd.DataFrame(node_data)
            node_df = pd.merge(node_df, engagement_df, on='node', how='left')

            node_df_comm = node_df[node_df['community'] == selected_community].copy()
            if 'TOTAL_CLICKTHROUGHS' in node_df_comm.columns and 'WEBVISIT_2020' in node_df_comm.columns:
                node_df_comm['Avg Web Visits'] = node_df_comm[[
                    'WEBVISIT_2020', 'WEBVISIT_2021', 'WEBVISIT_2022', 'WEBVISIT_2023', 'WEBVISIT_2024'
                ]].astype(float).mean(axis=1)

                nonzero_df = node_df_comm[node_df_comm['out_degree'] > 0].copy()
                top_engaged = node_df_comm.nlargest(10, 'out_degree').copy()

            
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Out-Degree vs Total Clickthroughs**")
                    fig3 = px.scatter(
                        nonzero_df,
                        x='out_degree',
                        y='TOTAL_CLICKTHROUGHS',
                        hover_data=['SENDER_NAME', 'SENDER_EMAIL'],
                        opacity=0.6,
                        labels={'out_degree': 'Out-Degree', 'TOTAL_CLICKTHROUGHS': 'Total Clickthroughs'}
                    )
                    fig3.add_trace(go.Scatter(
                        x=top_engaged['out_degree'],
                        y=top_engaged['TOTAL_CLICKTHROUGHS'],
                        mode='markers',
                        marker=dict(color='orange', size=10, line=dict(color='black', width=1)),
                        name='Top 10 Out-Degree'
                    ))
                    st.plotly_chart(fig3, use_container_width=True, key='click_chart')
            
                with col2:
                    st.markdown("**Out-Degree vs Avg Web Visits**")
                    fig4 = px.scatter(
                        nonzero_df,
                        x='out_degree',
                        y='Avg Web Visits',
                        hover_data=['SENDER_NAME', 'SENDER_EMAIL'],
                        opacity=0.6,
                        labels={'out_degree': 'Out-Degree', 'Avg Web Visits': 'Average Web Visits'}
                    )
                    fig4.add_trace(go.Scatter(
                        x=top_engaged['out_degree'],
                        y=top_engaged['Avg Web Visits'],
                        mode='markers',
                        marker=dict(color='orange', size=10, line=dict(color='black', width=1)),
                        name='Top 10 Out-Degree'
                    ))
                    st.plotly_chart(fig4, use_container_width=True, key='webvisits_chart')
            
                # Clickthrough Leaderboard
                st.write("")
                st.write("#### Engagement metrics for Top Users by Out-Degree")
                top_clicks_table = top_engaged[['SENDER_NAME', 'SENDER_EMAIL', 'TOTAL_CLICKTHROUGHS', 'Avg Web Visits', 'out_degree']]
                top_clicks_table.columns = ['Name', 'Email', 'Total Clickthroughs', 'Avg Web Visits', 'Out-Degree']
                st.dataframe(top_clicks_table.sort_values(by='Out-Degree', ascending=False).reset_index(drop=True))


               # --- Load Age and Donation Data ---
                age_df = pd.read_csv(os.path.join(data_dir, "WIS_age.csv"))  # Columns: ID, BIRTH_DATE, AGE
                age_df.rename(columns={'ID': 'node'}, inplace=True)
                
                don_df = pd.read_csv(os.path.join(data_dir, "WIS_donations.csv"))  # Contains SENDER_DIM_ACCOUNT_KEY and donation fields
                don_df.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'}, inplace=True)
                
                st.write("")
                st.write("#### Demographics and Donation Analysis")
                st.markdown("We are now combining node information with available demographic (age) and donation history data to enrich our understanding of community behavior.")
                
                # --- Merge All ---
                merged = node_df.merge(age_df[['node', 'AGE']], on='node', how='left')
                merged = merged.merge(don_df, on='node', how='left')
                merged['AGE'] = pd.to_numeric(merged['AGE'], errors='coerce')
                merged.fillna(0, inplace=True)
                
                # --- Filter for Selected Community ---
                comm_df = merged[merged['community'] == selected_community].copy()
                comm_df['AGE'] = pd.to_numeric(comm_df['AGE'], errors='coerce')
                age_valid = comm_df[comm_df['AGE'] > 0]['AGE']
                
                # --- Demographic Metrics ---
                mean_age = round(age_valid.mean(), 1)
                median_age = round(age_valid.median(), 1)
                above_60_pct = round((age_valid > 60).mean() * 100, 1)
                below_30_pct = round((age_valid < 30).mean() * 100, 1)
                
                # --- Donation Metrics ---
                # Filter to valid donors
                comm_df['YEARS_OF_DONATING'] = pd.to_numeric(comm_df['YEARS_OF_DONATING'], errors='coerce')
                donation_valid = comm_df[comm_df['YEARS_OF_DONATING'] > 0].copy()
            
                # Calculate donation metrics based on valid donors only
                total_donations = donation_valid['MAX_LIFETIME_DONATION_AMOUNT'].sum()
                avg_donation = donation_valid['AVG_LIFETIME_DONATION_AMOUNT'].mean()
                membership_rate = donation_valid['MEMBERSHIPS'].notnull().mean() * 100
                avg_years_donating = donation_valid['YEARS_OF_DONATING'].mean()
            
                # New proportions
                total_users = len(comm_df)
                nonzero_donors = len(donation_valid)
                zero_donors = total_users - nonzero_donors
                nonzero_pct = round((nonzero_donors / total_users) * 100, 1)
                zero_pct = round((zero_donors / total_users) * 100, 1)

                # --- Create side-by-side cards with equal height ---
                st.markdown(f"""
                <div style="display: flex; gap: 24px; align-items: stretch;">
                
                  <div style="flex: 1; background-color:#f9f9f9; border-left:5px solid #1f77b4; padding:16px 24px; border-radius:10px; min-height: 300px;">
                    <h4>Demographics – Community {selected_community}</h4>
                    <ul style="line-height:1.8;">
                      <li><strong>Mean Age:</strong> {mean_age}</li>
                      <li><strong>Median Age:</strong> {median_age}</li>
                      <li><strong>% Age 60+:</strong> {above_60_pct}%</li>
                      <li><strong>% Age &lt; 30:</strong> {below_30_pct}%</li>
                      <li><strong>Population (with age):</strong> {len(age_valid)} members</li>
                    </ul>
                  </div>
                
                  <div style="flex: 1; background-color:#f9f9f9; border-left:5px solid #C8102E; padding:16px 24px; border-radius:10px; min-height: 300px;">
                    <h4>Philanthropy – Community {selected_community}</h4>
                    <ul style="line-height:1.8;">
                      <li><strong>% of Users with more than 0 years of Donation:</strong> {nonzero_pct}%</li>
                      <li><strong>% of Users no Years of Donation:</strong> {zero_pct}%</li>
                      <br><strong>Metrics for users with Donation history</strong>
                      <li><strong>Total Lifetime Donations:</strong> ${total_donations:,.0f}</li>
                      <li><strong>Avg Donation per User:</strong> ${avg_donation:,.2f}</li>
                      <li><strong>Avg Years Donating:</strong> {avg_years_donating:.1f}</li>
                    </ul>
                  </div>
                
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.write("")

                ##Building 1-Hop network for donators in community
                #------------------------
                # Step 1: Identify upto Top 10 Donors in the Community
                top_donors = (
                    merged[(merged['community'] == selected_community) & 
                           (merged['MAX_LIFETIME_DONATION_AMOUNT'] > 0)]
                    .sort_values(by='MAX_LIFETIME_DONATION_AMOUNT', ascending=False)
                    .head(10)
                )
                
                top_nodes = top_donors['node'].tolist()
                
                # Step 2: Build 1-hop ego network
                one_hop_nodes = set(top_nodes)
                for donor in top_nodes:
                    neighbors = list(G_sub.successors(donor)) + list(G_sub.predecessors(donor))
                    one_hop_nodes.update(neighbors)
                
                G_ego = G_sub.subgraph(one_hop_nodes).copy()
                
                # Step 3: Prepare node positions
                # Adjust layout parameters for better spacing
                pos = nx.spring_layout(G_ego, seed=42, k=1.0, iterations=100)
                
                # Optional: scale positions to prevent compression at center
                for key in pos:
                    pos[key] *= 2  # scale up the layout to space out nodes
                
                node_x, node_y, node_size, node_hover = [], [], [], []
                for node in G_ego.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    out_deg = G_ego.nodes[node].get('out_degree', 0)
                    
                    donor_row = merged[merged['node'] == node]
                    max_donation = donor_row['MAX_LIFETIME_DONATION_AMOUNT'].values[0] if not donor_row.empty else 0
                    avg_donation = donor_row['AVG_LIFETIME_DONATION_AMOUNT'].values[0] if not donor_row.empty else 0
                    name = donor_row['SENDER_NAME_x'].values[0] if not donor_row.empty else node
                
                    hover_text = (
                        f"<b>Name:</b> {name}<br>"
                        f"<b>Node:</b> {node}<br>"
                        f"<b>Out-Degree:</b> {out_deg}<br>"
                        f"<b>Max Donation:</b> ${max_donation:.2f}<br>"
                        f"<b>Avg Donation:</b> ${avg_donation:.2f}"
                    )
                
                    node_hover.append(hover_text)
                    node_size.append(10 + out_deg)
                
                # Step 4: Edge coordinates
                edge_x, edge_y = [], []
                for src, tgt in G_ego.edges():
                    x0, y0 = pos[src]
                    x1, y1 = pos[tgt]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                
                # Step 5: Plotly network plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='#aaa'),
                    hoverinfo='none'
                ))
                
                fig.add_trace(go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    marker=dict(
                        size=node_size,
                        color=['darkred' if node in top_nodes else 'lightblue' for node in G_ego.nodes()],
                        line=dict(color='black', width=1)
                    ),
                    text=node_hover,
                    hoverinfo='text',
                    name='1-Hop Network'
                ))
                
                fig.update_layout(
                    title='1-Hop Network of Top 10 Donors in Selected Community',
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor='white',
                    height=550,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                col1, col2 = st.columns([2, 1])  # 2/3 and 1/3 layout

                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div style="background-color: #f9f9f9; border-left: 5px solid #C8102E; padding: 16px 20px; border-radius: 10px;">
                    <h4>How to Read This Network</h4>
                    <ul>
                        <li><b>Red nodes</b> = Top 10 donors</li>
                        <li><b>Blue nodes</b> = Their 1-hop neighbors</li>
                        <li><b>Node size</b> = Scaled by Out-Degree</li>
                        <li><b>Hover:</b> See Max Donation, Avg Donation, Out-Degree</li>
                        <li><b>Layout:</b> Spring layout separates communities</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)


# -------------------------
# Segment 3 Predictive Modeling
# -------------------------
elif selected == 'Predictive Modeling':
    st.header("Predictive Modeling")

    # --- Sidebar to select prediction type ---
    model_option = st.sidebar.radio(
        "Select Prediction Type:",
        ["Ticket Transfer Propensity (2025)", "Donation Propensity"]
    )
    
    # --- Load Feature Dataset ---
    @st.cache_data
    def load_features():
        return pd.read_csv("merged_dataset_WIS.csv")
    
    df_features = load_features()
    
    if model_option == "Ticket Transfer Propensity (2025)":
        st.subheader("Ticket Transfer Propensity Predictor for 2025")
        
        # --- Define Top Activities ---
        top_activities = ["F", "B", "WH", "H", "V"]
        activity_names = {
            "F": "Football",
            "B": "Men's Basketball",
            "WH": "Women's Hockey",
            "H": "Hockey",
            "V": "Volleyball"
        }
    
        # --- User Selection ---
        selected_activity = st.selectbox("Select an activity to run predictions for:", top_activities, format_func=lambda x: activity_names.get(x, x))
    
        # --- Load Model and Scaler ---
        model_path = f"models/rf_model_{selected_activity}.pkl"
        scaler_path = f"models/scaler_{selected_activity}.pkl"
    
        try:
            clf = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            st.error(f"Model or scaler for activity {selected_activity} not found.")
            st.stop()
    
        # --- Prepare Features ---
        drop_cols = ['SENDER_DIM_ACCOUNT_KEY', 'SENDER_NAME_x', 'PREFERRED_EMAIL', 'EMAILADDRESS', 'DIM_ACCOUNT_KEY', 'TICKET_BILLING_ZIP']
        X = df_features.drop(columns=drop_cols, errors='ignore').select_dtypes(include='number').fillna(0)
        X_scaled = scaler.transform(X)
    
        # --- Predict ---
        preds = clf.predict(X_scaled)
        probs = clf.predict_proba(X_scaled)[:, 1]
    
        # --- Append Predictions ---
        output_df = df_features.copy()
        output_df[f'{selected_activity}_Predicted_Label'] = preds
        output_df[f'{selected_activity}_Predicted_Prob'] = probs
        positive_preds = output_df[output_df[f'{selected_activity}_Predicted_Label'] == 1].copy()
    
        positive_percentage = (output_df[f'{selected_activity}_Predicted_Label'] == 1).mean() * 100
    
        # --- Layout: Prediction Table + Business Insight Box ---
        col1, col2 = st.columns([3, 1])
    
        with col1:
            st.subheader(f" Users Predicted to Transfer for {activity_names[selected_activity]} in 2025")
            st.write(f"Total predicted transfers: {len(positive_preds)}")
            st.write(f"** {positive_percentage:.2f}% of total users in dataset** are predicted to transfer for {activity_names[selected_activity]} in 2025.")
            st.dataframe(positive_preds[['SENDER_DIM_ACCOUNT_KEY', 'SENDER_NAME_x', f'{selected_activity}_Predicted_Label']].sort_values(by=f'{selected_activity}_Predicted_Label', ascending=False))
    
            csv = positive_preds.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f"predicted_transfers_2025_{selected_activity}.csv",
                mime='text/csv'
            )
    
        with col2:
            st.markdown(f"""
            <div style="background-color:#f9f9f9; border-left:5px solid #28a745; padding:16px 20px; border-radius:10px; min-height:300px;">
            <h4>Business Impact from Model Predictions</h4>
            <ul style="line-height:1.8;">
                <li><strong>{positive_percentage:.2f}% of total users in dataset</strong> are likely to transfer for <strong>{activity_names[selected_activity]}</strong>.</li>
                <li>This allows focused marketing on a high-intent segment.</li>
                <li>Can reduce spend on low-response segments.</li>
                <li>Improves ROI on ticket promotions and communications.</li>
                <li>Use predicted list to create early-access, loyalty, or upsell campaigns.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
        # --- Load and Display Precomputed Metrics ---
        try:
            with open(f"models/model_metrics_{selected_activity}.json", "r") as f:
                metrics = json.load(f)
    
            activity_metrics = metrics.get("activity", {})
            auc = activity_metrics.get("test_auc_2024", "N/A")
            cv_auc = activity_metrics.get("cv_auc_2023", "N/A")
            report = activity_metrics.get("classification_report_2024", {})
    
            st.subheader("Model Performance on 2024 Data")
            col_metrics, col_info = st.columns([3, 1])
    
            with col_metrics:
                st.markdown(f"""
                <div class="kpi-container" style="display:flex; gap:20px; margin-bottom:20px;">
                    <div class="kpi-card" style="flex:1; background:#ffffff; padding:16px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;">
                        <div class="kpi-value" style="font-size:24px; font-weight:bold; color:#C8102E;">{auc:.3f}</div>
                        <div class="kpi-label">ROC AUC Score on 2024 Test Data</div>
                    </div>
                    <div class="kpi-card" style="flex:1; background:#ffffff; padding:16px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;">
                        <div class="kpi-value" style="font-size:24px; font-weight:bold; color:#C8102E;">{cv_auc:.3f}</div>
                        <div class="kpi-label">Cross Validation AUC on 2023 Train data</div>
                    </div>
                    <div class="kpi-card" style="flex:1; background:#ffffff; padding:16px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;">
                        <div class="kpi-value" style="font-size:24px; font-weight:bold; color:#C8102E;">{report.get("accuracy", 0.0):.3f}</div>
                        <div class="kpi-label"> Model Accuracy</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
                if isinstance(report, dict) and report:
                    report_df = pd.DataFrame(report).transpose()
                    report_df = report_df.drop(columns=['support'], errors='ignore')
                    report_df = report_df.drop(index=['macro avg', 'weighted avg','accuracy'], errors='ignore')
                    st.dataframe(report_df, use_container_width=True)
    
            with col_info:
                st.markdown("""
                    <div style="background-color:#f9f9f9; border-left:5px solid #1f77b4; padding:16px 20px; border-radius:10px; min-height:200px;">
                    <h4>Model Performance Insights</h4>
                    <ul style="line-height:1.8;">
                        <li>Trained and validated on 2023 data using cross-validation.</li>
                        <li>Evaluated on 2024 data to assess generalization.</li>
                        <li>Displayed metrics reflect performance on 2024 labels.</li>
                    </ul>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning("Could not load saved model performance metrics.")
    
    elif model_option == "Donation Propensity":
        st.subheader("Donation Propensity Model")
        st.markdown("This section visualizes and lists prospects predicted to make a donation based on model scores.")
    
        df = pd.read_csv("final_model_dataset_WIS.csv")
        df['made_gift'] = (df['MAX_LIFETIME_DONATION_AMOUNT'] > 0).astype(int)
    
        drop_gift = [c for c in df.columns if 'greatest' in c.lower() or 'lifetime' in c.lower()]
        drop_meta = [c for c in df.columns if any(k in c.lower() for k in ['key','id','name','email','date','status'])]
        X = df.drop(columns=drop_gift + drop_meta + ['made_gift','TOTAL_CLICKTHROUGHS_final','TOTAL_TRANSFERS','community'])
        y = df['made_gift']
    
        X = X.select_dtypes(include='number').fillna(0)
    
        leakers = []
        for col in X.columns:
            non_don_max = X.loc[y==0, col].max()
            don_min     = X.loc[y==1, col].min()
            don_max     = X.loc[y==1, col].max()
            non_don_min = X.loc[y==0, col].min()
            if non_don_max < don_min or don_max < non_don_min:
                leakers.append(col)
    
        if leakers:
            X = X.drop(columns=leakers)
    
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X)
        X = pd.DataFrame(vt.transform(X), columns=X.columns[vt.get_support()])
    
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        perfect_dups = [col for col in upper.columns if any(upper[col] == 1.0)]
        if perfect_dups:
            X = X.drop(columns=perfect_dups)
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
    
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
    
        # Score and select top prospects
        prospects = X.loc[df['made_gift']==0].copy()
        prospects['Donation_score']   = rf.predict_proba(prospects)[:,1]
        prospects['SENDER_DIM_ACCOUNT_KEY'] = df.loc[prospects.index,'SENDER_DIM_ACCOUNT_KEY']
        prospects['Name']   = df.loc[prospects.index,'SENDER_NAME_x']
        prospects = prospects[['SENDER_DIM_ACCOUNT_KEY', 'Name', 'Donation_score']]

        top_prospects = prospects.sort_values(by='Donation_score', ascending=False).head(50)

        percent_positive = (prospects['Donation_score'] > 0.5).mean() * 100
    
        # --- Layout ---
        col1, col2 = st.columns([3, 1])
    
        with col1:
            st.subheader("Top Donation Prospects")
            st.write(f"Users predicted with > 0.5 score: {percent_positive:.2f}%")
            st.dataframe(top_prospects)
    
            csv = top_prospects.to_csv(index=False)
            st.download_button(" Download Prospects", data=csv, file_name="donation_prospects.csv", mime="text/csv")
    
        with col2:
            st.markdown(f"""
            <div style="background-color:#f9f9f9; border-left:5px solid #28a745; padding:16px 20px; border-radius:10px; min-height:300px;">
            <h4>Business Insight</h4>
            <ul style="line-height:1.8;">
                <li><strong>{percent_positive:.2f}%</strong> users are viable donation leads.</li>
                <li>Helps identify hidden value among non-donors.</li>
                <li>Use to design fundraising and outreach campaigns.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
        # --- Model Performance ---
        st.subheader("Model Performance")
        col1, col2 = st.columns([3, 1])
    
        with col1:
            st.markdown(f"""
            <div class="kpi-container" style="display:flex; gap:20px; margin-bottom:20px;">
                <div class="kpi-card" style="flex:1; background:#fff; padding:16px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;">
                    <div class="kpi-value" style="font-size:24px; font-weight:bold; color:#C8102E;">{auc:.3f}</div>
                    <div class="kpi-label">ROC AUC</div>
                </div>
                <div class="kpi-card" style="flex:1; background:#fff; padding:16px; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.1); text-align:center;">
                    <div class="kpi-value" style="font-size:24px; font-weight:bold; color:#C8102E;">{acc:.3f}</div>
                    <div class="kpi-label">Accuracy</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose().drop(columns='support', errors='ignore')
            df_report = df_report.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            st.dataframe(df_report)
    
        with col2:
            st.markdown("""
            <div style="background-color:#f9f9f9; border-left:5px solid #1f77b4; padding:16px 20px; border-radius:10px; min-height:200px;">
            <h4>Model Info</h4>
            <ul style="line-height:1.8;">
                <li>Trained on historic donor activity.</li>
                <li>Filtered for data leakage, variance, and collinearity.</li>
                <li>Evaluated on stratified 20% test data.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- Feature Importance ---
        # --- Feature Importance ---
        st.subheader("Feature Importance")
        
        # Build feature importance dataframe
        importance_df = (
            pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            })
            .sort_values(by='Importance', ascending=False)
        )
        
        # Define readable names for display
        feature_name_mapping = {
            'eigenvector': 'Network Influence (Eigenvector)',
            'out_degree': 'Connections Made (Out-Degree)',
            'betweenness': 'Betweenness Centrality',
            'degree_centrality': 'Direct Influence (Degree Centrality)',
            'unique_recipients': 'Unique Transfer Recipients',
            'F_TRANSFER': 'Football Transfers',
            'B_TRANSFER': 'Men’s Basketball Transfers',
            'H_TRANSFER': 'Hockey Transfers',
            'WEBVISIT_2022_final': 'Web Visits in 2022',
            'WEBVISIT_2023_final': 'Web Visits in 2023',
            'WEBVISIT_2024_final': 'Web Visits in 2024',
            'TRANSFERS_2021_final': 'Total Transfers in 2021',
            'TRANSFERS_2022_final': 'Total Transfers in 2022',
            'TRANSFERS_2023_final': 'Total Transfers in 2023',
            'TRANSFERS_2024_final': 'Total Transfers in 2024',
        }
        
        # Apply renaming for readability
        importance_df['Features'] = importance_df['Feature'].map(feature_name_mapping).fillna(importance_df['Feature'])
        
        # Plot using Plotly
        fig_imp = px.bar(
            importance_df.head(5),
            x='Importance',
            y='Features',
            orientation='h',
            title="Top 5 Features Influencing Donation Propensity",
            color='Feature',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_imp.update_layout(
            yaxis={
                'categoryorder': 'total ascending',
                'title': 'Features',
                'titlefont': dict(color='black', size=14, family='Arial'),
                'tickfont': dict(color='black', size=12)
            },
            xaxis={
                'title': 'Importance',
                'titlefont': dict(color='black', size=14, family='Arial'),
                'tickfont': dict(color='black', size=12)
            },
            height=500
        )

        fig_imp.update_traces(width=0.5)
        
        # Display chart
        st.plotly_chart(fig_imp, use_container_width=True)
        
        ##1-Hop network of preducted donors
        #------------------------------------

        # Get the Top 20 Predicted Donors
        top_prospects = prospects.sort_values(by='Donation_score', ascending=False).head(20)
        top_nodes = top_prospects['SENDER_DIM_ACCOUNT_KEY'].tolist()
        prospects = prospects.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'})
        df = pd.read_csv('Data1_WIS.csv')
        # Load the full dataset if not already loaded

        # Create sender-recipient mapping for names
        senders = df[['SENDER_DIM_ACCOUNT_KEY', 'SENDER_NAME']].rename(columns={
            'SENDER_DIM_ACCOUNT_KEY': 'node',
            'SENDER_NAME': 'Name'
        })
        
        recipients = df[['RECEPIENT_DIM_ACCOUNT_KEY', 'RECEPIENT_NAME']].rename(columns={
            'RECEPIENT_DIM_ACCOUNT_KEY': 'node',
            'RECEPIENT_NAME': 'Name'
        })

        user_info = pd.concat([senders, recipients]).drop_duplicates(subset='node')
        # Load donation data
        don_df = pd.read_csv("WIS_donations.csv")
        don_df.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'}, inplace=True)
        don_df = don_df[['node', 'MAX_LIFETIME_DONATION_AMOUNT', 'YEARS_OF_DONATING']]

        # Extract relevant edges where either sender or recipient is a top prospect
        edges_df = df[
            df['SENDER_DIM_ACCOUNT_KEY'].isin(top_nodes) |
            df['RECEPIENT_DIM_ACCOUNT_KEY'].isin(top_nodes)
        ]
        
        # Optional: Filter down to essential columns
        edges_df = edges_df[['SENDER_DIM_ACCOUNT_KEY', 'RECEPIENT_DIM_ACCOUNT_KEY', 'TOTAL_TRANSFERS']]
        edges_df = edges_df.rename(columns={
            'SENDER_DIM_ACCOUNT_KEY': 'source',
            'RECEPIENT_DIM_ACCOUNT_KEY': 'target'
        })

        #Build the Network Graph
        G = nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr='TOTAL_TRANSFERS', create_using=nx.DiGraph())
        
        # Compute out-degree
        out_degree = dict(G.out_degree(weight='TOTAL_TRANSFERS'))
        nx.set_node_attributes(G, out_degree, 'out_degree')

        # Build node DataFrame and enrich
        node_df = pd.DataFrame({'node': list(G.nodes())})
        node_df = node_df.merge(don_df, on='node', how='left')
        node_df = node_df.merge(user_info, on='node', how='left')   # Name
        node_df = node_df.merge(prospects[['node', 'Donation_score']], on='node', how='left')   # Donation_score
        node_df.fillna(0, inplace=True)

        #Position and Visualize with Plotly
        # --- Layout ---
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        node_x, node_y, node_size, node_color, node_text = [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            od = out_degree.get(node, 0)
            node_size.append(10 + od)

            donation_row = node_df[node_df['node'] == node]
            max_donation = donation_row['MAX_LIFETIME_DONATION_AMOUNT'].values[0] if not donation_row.empty else 0
            years = donation_row['YEARS_OF_DONATING'].values[0] if not donation_row.empty else 0
            score = donation_row['Donation_score'].values[0] if not donation_row.empty else 0
            name = donation_row['Name'].values[0] if 'Name' in donation_row and not donation_row.empty else 'N/A'
            
            # Color by whether the node is a top prospect
            # Color logic
            if node in top_nodes:
                node_color.append("crimson")  # Top prospect
            elif max_donation > 0:
                node_color.append("orange")   # Any donor
            else:
                node_color.append("steelblue")  # Other nodes
            
            node_text.append(
                f"<b>Node:</b> {node}<br>"
                f"<b>Name:</b> {name}<br>"
                f"<b>Out-Degree:</b> {od}<br>"
                f"<b>Predicted Donation Score:</b> {score:.3f}<br>"
                f"<b>Max Donation:</b> ${max_donation:,.0f}<br>"
                f"<b>Years Donating:</b> {years}"
            )
        
        # --- Edges ---
        edge_x, edge_y = [], []
        for src, tgt in G.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        # --- Plotly Figure ---
        # --- Plotly Node Traces with Legend ---
        node_traces = []
        
        # 1. Top Prospects (Red)
        top_nodes_trace = go.Scatter(
            x=[pos[n][0] for n in G.nodes() if n in top_nodes],
            y=[pos[n][1] for n in G.nodes() if n in top_nodes],
            mode='markers',
            name='Top 20 Predicted Donors',
            marker=dict(size=[10 + out_degree.get(n, 0) for n in G.nodes() if n in top_nodes],
                        color='crimson',
                        line=dict(color='black', width=1)),
            text=[node_text[i] for i, n in enumerate(G.nodes()) if n in top_nodes],
            hoverinfo='text',
            legendgroup='prospect'
        )
        node_traces.append(top_nodes_trace)
        
        # 2. Donors (Orange)
        donor_nodes = [n for n in G.nodes() if n not in top_nodes and node_df.loc[node_df['node'] == n, 'MAX_LIFETIME_DONATION_AMOUNT'].values[0] > 0]
        donor_trace = go.Scatter(
            x=[pos[n][0] for n in donor_nodes],
            y=[pos[n][1] for n in donor_nodes],
            mode='markers',
            name='Existing Donor',
            marker=dict(size=[10 + out_degree.get(n, 0) for n in donor_nodes],
                        color='orange',
                        line=dict(color='black', width=1)),
            text=[node_text[i] for i, n in enumerate(G.nodes()) if n in donor_nodes],
            hoverinfo='text',
            legendgroup='donor'
        )
        node_traces.append(donor_trace)
        
        # 3. Other Nodes (Steel Blue)
        other_nodes = [n for n in G.nodes() if n not in top_nodes and n not in donor_nodes]
        other_trace = go.Scatter(
            x=[pos[n][0] for n in other_nodes],
            y=[pos[n][1] for n in other_nodes],
            mode='markers',
            name='Other Non-Donor',
            marker=dict(size=[10 + out_degree.get(n, 0) for n in other_nodes],
                        color='steelblue',
                        line=dict(color='black', width=1)),
            text=[node_text[i] for i, n in enumerate(G.nodes()) if n in other_nodes],
            hoverinfo='text',
            legendgroup='other'
        )
        node_traces.append(other_trace)
        
        # --- Edge Trace ---
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#aaa'),
            hoverinfo='none',
            showlegend=True
        )
        
        # --- Combine and Plot ---
        fig = go.Figure(data=[edge_trace] + node_traces)
        
        fig.update_layout(
            title="Top 20 Predicted Donors – 1-Hop Network",
            margin=dict(l=10, r=10, t=40, b=10),
            height=500,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Extract Top Prospects ---
        # Keep only necessary columns and rename sender key to match node IDs in the graph
        top_prospects = top_prospects[['SENDER_DIM_ACCOUNT_KEY', 'Donation_score']]
        top_prospects = top_prospects.rename(columns={'SENDER_DIM_ACCOUNT_KEY': 'node'})
        # Sort the full prospect list by predicted donation score and take the top 20
        top_prospects = prospects.sort_values(by='Donation_score', ascending=False).head(20)

        
        # --- Extract All 1-Hop Connections for Top 20 Nodes ---
        all_connections = []
        
        for donor_node in top_prospects['node']:
            # Add outgoing edges (donor → others)
            for tgt in G.successors(donor_node):
                all_connections.append({
                    'Donor Node ID': donor_node,
                    'Connected Node ID': tgt,
                    'Direction': 'Sent To'
                })
            # Add incoming edges (others → donor)
            for src in G.predecessors(donor_node):
                all_connections.append({
                    'Donor Node ID': donor_node,
                    'Connected Node ID': src,
                    'Direction': 'Received From'
                })

        # Convert connections list to DataFrame
        connections_df = pd.DataFrame(all_connections)
        
        # --- Merge Metadata to Enrich Connections ---
        # Add donor names
        connections_df = connections_df.merge(user_info.rename(columns={'node': 'Donor Node ID', 'Name': 'Donor Name'}), on='Donor Node ID', how='left')
        # Add connected node names
        connections_df = connections_df.merge(user_info.rename(columns={'node': 'Connected Node ID', 'Name': 'Connected Name'}), on='Connected Node ID', how='left')
        # Add predicted donation score for connected nodes (if applicable)
        connections_df = connections_df.merge(prospects[['node', 'Donation_score']].rename(columns={'node': 'Connected Node ID'}), on='Connected Node ID', how='left')
        # Add actual donation history: lifetime amount and years of donating
        connections_df = connections_df.merge(don_df.rename(columns={'node': 'Connected Node ID'}), on='Connected Node ID', how='left')
        
        # Add out-degree centrality for connected nodes
        out_deg_df = pd.DataFrame.from_dict(out_degree, orient='index', columns=['Out-Degree']).reset_index().rename(columns={'index': 'Connected Node ID'})
        connections_df = connections_df.merge(out_deg_df, on='Connected Node ID', how='left')
        
        # --- Step 4: Clean, Format, and Present ---
        # Select and reorder relevant columns
        connections_df = connections_df[[
            'Donor Node ID', 'Donor Name', 'Connected Node ID', 'Connected Name',
            'Direction', 'Donation_score', 'MAX_LIFETIME_DONATION_AMOUNT',
            'YEARS_OF_DONATING', 'Out-Degree'
        ]]
        # Rename columns for clarity
        connections_df.columns = [
            'Donor Node ID', 'Donor Name', 'Connected Node ID', 'Connected Name',
            'Direction', 'Predicted Score', 'Max Donation ($)', 'Years Donating', 'Out-Degree'
        ]

        # Replace NaNs with 0
        connections_df.fillna(0, inplace=True)
        
        # Format donation score and donation amount
        connections_df['Predicted Score'] = connections_df['Predicted Score'].round(3)
        connections_df['Max Donation ($)'] = connections_df['Max Donation ($)'].map('${:,.0f}'.format)
        
        
        # ---Display in Dashboard ---        
        # Show the table of 1-hop connections for the top 20 predicted donors
        st.subheader("1-Hop Connections of Top 20 Predicted Donors")
        st.dataframe(connections_df, use_container_width=True)
        
        # Option to download the data as CSV
        csv = connections_df.to_csv(index=False)
        st.download_button("Download 1-Hop Connection Table", data=csv, file_name="top20_donor_connections.csv", mime="text/csv")

