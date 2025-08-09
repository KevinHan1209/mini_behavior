"""
Interactive Streamlit Dashboard for APT_Experiments_v2 Results
Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="APT Experiments v2 Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    h2 {
        color: #34495e;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Note: Removed @st.cache_data to ensure fresh data loading
def load_experiment_config(exp_dir):
    """Load experiment configuration from JSON file."""
    config_path = exp_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

# Note: Removed @st.cache_data to ensure fresh data loading
def load_activity_data(exp_dir):
    """Load activity data from checkpoint CSV files."""
    activity_logs_dir = exp_dir / "checkpoints" / "activity_logs"
    if not activity_logs_dir.exists():
        return None
    
    checkpoints = [500000, 1000000, 1500000, 2000000, 2500000]
    activity_data = []
    
    for checkpoint in checkpoints:
        csv_path = activity_logs_dir / f"checkpoint_{checkpoint}_activity.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Calculate total activity count (sum of all non-zero activities)
                total_activity = df.iloc[0, 1:].sum()  # Skip checkpoint_id column
                unique_states = (df.iloc[0, 1:] > 0).sum()  # Count non-zero states
                
                # Get top active states
                state_values = df.iloc[0, 1:].to_dict()
                top_states = sorted(state_values.items(), key=lambda x: x[1], reverse=True)[:10]
                
                activity_data.append({
                    'checkpoint': checkpoint,
                    'total_activity': total_activity,
                    'unique_states': unique_states,
                    'state_coverage': unique_states / (len(df.columns) - 1),
                    'top_states': top_states
                })
    
    return pd.DataFrame(activity_data) if activity_data else None

# Note: Removed @st.cache_data to ensure fresh data loading
def load_intrinsic_rewards(exp_dir):
    """Load intrinsic rewards data from CSV file."""
    rewards_path = exp_dir / "intrinsic_rewards_log.csv"
    if rewards_path.exists():
        return pd.read_csv(rewards_path)
    return None

def get_experiment_label(exp_name, config):
    """Generate a descriptive label for the experiment."""
    if config:
        # Extract parameters - check if they're nested under 'hyperparameters'
        if 'hyperparameters' in config:
            k = config['hyperparameters'].get('k', 12)
            ent_coef = config['hyperparameters'].get('ent_coef', 0.01)
        else:
            k = config.get('k', 12)
            ent_coef = config.get('ent_coef', 0.01)
        
        if 'k_' in exp_name:
            return f"k={k}"
        elif 'ent_coef_' in exp_name:
            return f"ent_coef={ent_coef}"
        elif 'default' in exp_name:
            return "Default (k=12, ent=0.01)"
    
    if 'k_' in exp_name:
        k_value = exp_name.split('k_')[1].split('_')[0]
        return f"k={k_value}"
    elif 'ent_coef_' in exp_name:
        ent_value = exp_name.split('ent_coef_')[1]
        return f"ent_coef={ent_value}"
    elif 'default' in exp_name:
        return "Default"
    
    return exp_name

# Note: Removed @st.cache_data to ensure fresh data loading
def load_all_experiments(results_dir):
    """Load data from all experiments."""
    results_path = Path(results_dir)
    all_experiments = []
    experiment_configs = {}
    
    for exp_dir in sorted(results_path.iterdir()):
        if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
            config = load_experiment_config(exp_dir)
            label = get_experiment_label(exp_dir.name, config)
            experiment_configs[label] = config
            print(f"DEBUG: Loading {exp_dir.name} with label '{label}'")  # Debug line
            
            activity_df = load_activity_data(exp_dir)
            if activity_df is not None:
                activity_df['experiment'] = label
                activity_df['exp_name'] = exp_dir.name
                
                rewards_df = load_intrinsic_rewards(exp_dir)
                if rewards_df is not None:
                    checkpoint_rewards = []
                    for checkpoint in [500000, 1000000, 1500000, 2000000, 2500000]:
                        closest_idx = np.abs(rewards_df['global_step'] - checkpoint).argmin()
                        checkpoint_rewards.append({
                            'checkpoint': checkpoint,
                            'avg_intrinsic_reward': rewards_df.iloc[closest_idx]['avg_intrinsic_reward'],
                            'std_intrinsic_reward': rewards_df.iloc[closest_idx]['std_intrinsic_reward']
                        })
                    
                    rewards_checkpoint_df = pd.DataFrame(checkpoint_rewards)
                    activity_df = pd.merge(activity_df, rewards_checkpoint_df, on='checkpoint', how='left')
                
                all_experiments.append(activity_df)
    
    if all_experiments:
        combined_df = pd.concat(all_experiments, ignore_index=True)
        return combined_df, experiment_configs
    return None, {}

def main():
    # Title and description
    st.title("🔬 APT Experiments v2: Interactive Dashboard")
    st.markdown("### Explore and compare exploration metrics across different APT configurations")
    
    # Load data
    results_dir = "/Users/kevinhan/mini_behavior/APT_Experiments_v2/results"
    
    with st.spinner("Loading experiment data..."):
        combined_df, experiment_configs = load_all_experiments(results_dir)
    
    if combined_df is None:
        st.error("No experiment data found!")
        return
    
    # Sidebar for filtering
    st.sidebar.header("🎛️ Filters & Settings")
    
    # Experiment selection
    all_experiments = sorted(combined_df['experiment'].unique())
    selected_experiments = st.sidebar.multiselect(
        "Select Experiments",
        options=all_experiments,
        default=all_experiments,
        help="Choose which experiments to display"
    )
    
    # Filter data
    filtered_df = combined_df[combined_df['experiment'].isin(selected_experiments)]
    
    # Checkpoint selection for detailed view
    checkpoint_options = sorted(filtered_df['checkpoint'].unique())
    selected_checkpoint = st.sidebar.select_slider(
        "Select Checkpoint",
        options=checkpoint_options,
        value=checkpoint_options[-1],
        format_func=lambda x: f"{x/1e6:.1f}M steps"
    )
    
    # Color scheme selection
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ["Default", "Viridis", "Turbo", "Rainbow", "Blues"],
        help="Choose color palette for plots"
    )
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Evolution Metrics", 
        "🎯 Final Performance", 
        "📊 Comparative Analysis",
        "🔍 Detailed View",
        "📋 Summary Statistics"
    ])
    
    with tab1:
        st.header("Evolution Metrics Over Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Total Activity Count Evolution
            fig_activity = px.line(
                filtered_df,
                x='checkpoint',
                y='total_activity',
                color='experiment',
                title='Total Activity Count Evolution',
                labels={'checkpoint': 'Training Steps', 'total_activity': 'Total Activity Count'},
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
            )
            fig_activity.update_layout(
                xaxis_tickformat=',.0f',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_activity, use_container_width=True)
            
            # State Coverage Evolution
            fig_coverage = px.line(
                filtered_df,
                x='checkpoint',
                y='state_coverage',
                color='experiment',
                title='State Space Coverage Evolution',
                labels={'checkpoint': 'Training Steps', 'state_coverage': 'State Coverage'},
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
            )
            fig_coverage.update_layout(
                yaxis_tickformat='.1%',
                xaxis_tickformat=',.0f',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col2:
            # Unique States Evolution
            fig_unique = px.line(
                filtered_df,
                x='checkpoint',
                y='unique_states',
                color='experiment',
                title='Unique States Visited',
                labels={'checkpoint': 'Training Steps', 'unique_states': 'Unique States'},
                markers=True,
                color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
            )
            fig_unique.update_layout(
                xaxis_tickformat=',.0f',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_unique, use_container_width=True)
            
            # Intrinsic Reward Evolution
            if 'avg_intrinsic_reward' in filtered_df.columns:
                fig_reward = px.line(
                    filtered_df,
                    x='checkpoint',
                    y='avg_intrinsic_reward',
                    color='experiment',
                    title='Average Intrinsic Reward Evolution',
                    labels={'checkpoint': 'Training Steps', 'avg_intrinsic_reward': 'Avg Intrinsic Reward'},
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
                )
                fig_reward.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_reward.update_layout(
                    xaxis_tickformat=',.0f',
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_reward, use_container_width=True)
    
    with tab2:
        st.header("Final Performance Comparison")
        
        final_data = filtered_df[filtered_df['checkpoint'] == checkpoint_options[-1]]
        
        if not final_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart for total activity
                fig_bar_activity = px.bar(
                    final_data.sort_values('total_activity', ascending=True),
                    x='total_activity',
                    y='experiment',
                    orientation='h',
                    title=f'Total Activity Count at {checkpoint_options[-1]/1e6:.1f}M Steps',
                    labels={'total_activity': 'Total Activity Count', 'experiment': 'Experiment'},
                    color='total_activity',
                    color_continuous_scale="blues" if color_scheme == "Default" else color_scheme.lower(),
                    text='total_activity'
                )
                fig_bar_activity.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_bar_activity.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar_activity, use_container_width=True)
                
                # Bar chart for state coverage
                fig_bar_coverage = px.bar(
                    final_data.sort_values('state_coverage', ascending=True),
                    x='state_coverage',
                    y='experiment',
                    orientation='h',
                    title=f'State Coverage at {checkpoint_options[-1]/1e6:.1f}M Steps',
                    labels={'state_coverage': 'State Coverage (%)', 'experiment': 'Experiment'},
                    color='state_coverage',
                    color_continuous_scale="greens" if color_scheme == "Default" else color_scheme.lower(),
                    text='state_coverage'
                )
                fig_bar_coverage.update_traces(
                    texttemplate='%{text:.1%}', 
                    textposition='outside'
                )
                fig_bar_coverage.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar_coverage, use_container_width=True)
            
            with col2:
                # Scatter plot: Activity vs Coverage
                fig_scatter = px.scatter(
                    final_data,
                    x='unique_states',
                    y='total_activity',
                    color='experiment',
                    size='state_coverage',
                    title='Exploration Efficiency',
                    labels={
                        'unique_states': 'Unique States Visited',
                        'total_activity': 'Total Activity Count',
                        'state_coverage': 'Coverage'
                    },
                    hover_data=['state_coverage'],
                    color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Radar chart for multi-metric comparison
                metrics = ['total_activity', 'unique_states', 'state_coverage']
                
                # Normalize metrics for radar chart
                normalized_data = final_data.copy()
                for metric in metrics:
                    max_val = normalized_data[metric].max()
                    if max_val > 0:
                        normalized_data[f'{metric}_norm'] = normalized_data[metric] / max_val
                
                fig_radar = go.Figure()
                
                for _, row in normalized_data.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['total_activity_norm'], 
                           row['unique_states_norm'], 
                           row['state_coverage_norm']],
                        theta=['Total Activity', 'Unique States', 'State Coverage'],
                        fill='toself',
                        name=row['experiment']
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Multi-Metric Comparison (Normalized)",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.header("Comparative Analysis")
        
        # Growth rate analysis
        growth_data = []
        for exp in selected_experiments:
            exp_data = filtered_df[filtered_df['experiment'] == exp].sort_values('checkpoint')
            if len(exp_data) > 1:
                for i in range(1, len(exp_data)):
                    growth_rate = (exp_data.iloc[i]['total_activity'] - exp_data.iloc[i-1]['total_activity']) / \
                                 ((exp_data.iloc[i]['checkpoint'] - exp_data.iloc[i-1]['checkpoint']) / 1e6)
                    growth_data.append({
                        'experiment': exp,
                        'checkpoint': (exp_data.iloc[i]['checkpoint'] + exp_data.iloc[i-1]['checkpoint']) / 2,
                        'growth_rate': growth_rate
                    })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Growth rate over time
                fig_growth = px.line(
                    growth_df,
                    x='checkpoint',
                    y='growth_rate',
                    color='experiment',
                    title='Activity Growth Rate (Exploration Velocity)',
                    labels={'checkpoint': 'Training Steps', 'growth_rate': 'Growth Rate (activities/million steps)'},
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
                )
                fig_growth.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_growth.update_layout(
                    xaxis_tickformat=',.0f',
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_growth, use_container_width=True)
            
            with col2:
                # Efficiency metric: Activity per unique state
                efficiency_df = filtered_df.copy()
                efficiency_df['efficiency'] = efficiency_df['total_activity'] / efficiency_df['unique_states'].replace(0, np.nan)
                
                fig_efficiency = px.line(
                    efficiency_df,
                    x='checkpoint',
                    y='efficiency',
                    color='experiment',
                    title='Exploration Efficiency (Activity per Unique State)',
                    labels={'checkpoint': 'Training Steps', 'efficiency': 'Activities per State'},
                    markers=True,
                    color_discrete_sequence=px.colors.qualitative.Plotly if color_scheme == "Default" else getattr(px.colors.sequential, color_scheme, px.colors.qualitative.Plotly)
                )
                fig_efficiency.update_layout(
                    xaxis_tickformat=',.0f',
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Heatmap comparison
        st.subheader("Checkpoint Comparison Heatmap")
        
        # Create pivot table for heatmap
        heatmap_metric = st.selectbox(
            "Select metric for heatmap",
            ["total_activity", "unique_states", "state_coverage"],
            format_func=lambda x: {
                "total_activity": "Total Activity",
                "unique_states": "Unique States",
                "state_coverage": "State Coverage"
            }[x]
        )
        
        pivot_data = filtered_df.pivot_table(
            values=heatmap_metric,
            index='experiment',
            columns='checkpoint',
            aggfunc='mean'
        )
        
        fig_heatmap = px.imshow(
            pivot_data,
            labels=dict(x="Checkpoint", y="Experiment", color=heatmap_metric),
            x=[f"{c/1e6:.1f}M" for c in pivot_data.columns],
            y=pivot_data.index,
            color_continuous_scale="blues" if color_scheme == "Default" else color_scheme.lower(),
            title=f"{heatmap_metric.replace('_', ' ').title()} Across Checkpoints"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.header("Detailed View")
        
        # Select specific checkpoint for detailed analysis
        checkpoint_data = filtered_df[filtered_df['checkpoint'] == selected_checkpoint]
        
        if not checkpoint_data.empty:
            st.subheader(f"Analysis at {selected_checkpoint/1e6:.1f}M Steps")
            
            # Create detailed metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            for idx, row in checkpoint_data.iterrows():
                with col1:
                    st.metric(
                        label=f"{row['experiment']}",
                        value=f"{int(row['total_activity'])}",
                        delta="Total Activities"
                    )
                with col2:
                    st.metric(
                        label="",
                        value=f"{int(row['unique_states'])}",
                        delta="Unique States"
                    )
                with col3:
                    st.metric(
                        label="",
                        value=f"{row['state_coverage']*100:.1f}%",
                        delta="Coverage"
                    )
                with col4:
                    if 'avg_intrinsic_reward' in row:
                        st.metric(
                            label="",
                            value=f"{row['avg_intrinsic_reward']:.4f}",
                            delta="Intrinsic Reward"
                        )
            
            # Top active states analysis
            st.subheader("Top Active States Analysis")
            
            selected_exp_detail = st.selectbox(
                "Select experiment for detailed state analysis",
                options=checkpoint_data['experiment'].unique()
            )
            
            exp_detail_data = checkpoint_data[checkpoint_data['experiment'] == selected_exp_detail].iloc[0]
            
            if 'top_states' in exp_detail_data and exp_detail_data['top_states']:
                top_states_df = pd.DataFrame(
                    exp_detail_data['top_states'],
                    columns=['State', 'Count']
                )
                
                fig_top_states = px.bar(
                    top_states_df,
                    x='Count',
                    y='State',
                    orientation='h',
                    title=f"Top 10 Active States - {selected_exp_detail}",
                    labels={'Count': 'Activity Count', 'State': 'State Name'},
                    color='Count',
                    color_continuous_scale="viridis" if color_scheme == "Default" else color_scheme.lower()
                )
                fig_top_states.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_top_states, use_container_width=True)
    
    with tab5:
        st.header("Summary Statistics")
        
        # Final checkpoint statistics
        final_stats = filtered_df[filtered_df['checkpoint'] == checkpoint_options[-1]]
        
        if not final_stats.empty:
            # Create summary table
            summary_data = []
            for _, row in final_stats.iterrows():
                summary_data.append({
                    'Experiment': row['experiment'],
                    'Total Activity': int(row['total_activity']),
                    'Unique States': int(row['unique_states']),
                    'State Coverage (%)': f"{row['state_coverage']*100:.1f}",
                    'Avg Intrinsic Reward': f"{row.get('avg_intrinsic_reward', 0):.4f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Style the dataframe
            st.dataframe(
                summary_df.style.highlight_max(
                    subset=['Total Activity', 'Unique States'],
                    color='lightgreen'
                ).highlight_min(
                    subset=['Total Activity', 'Unique States'],
                    color='lightcoral'
                ),
                use_container_width=True
            )
            
            # Key insights
            st.subheader("Key Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_activity = final_stats.loc[final_stats['total_activity'].idxmax()]
                st.info(f"""
                **Highest Total Activity:**  
                {best_activity['experiment']}  
                ({int(best_activity['total_activity'])} activities)
                """)
            
            with col2:
                best_coverage = final_stats.loc[final_stats['state_coverage'].idxmax()]
                st.info(f"""
                **Best State Coverage:**  
                {best_coverage['experiment']}  
                ({best_coverage['state_coverage']*100:.1f}%)
                """)
            
            with col3:
                best_unique = final_stats.loc[final_stats['unique_states'].idxmax()]
                st.info(f"""
                **Most Unique States:**  
                {best_unique['experiment']}  
                ({int(best_unique['unique_states'])} states)
                """)
            
            # Correlation analysis
            st.subheader("Correlation Analysis")
            
            corr_cols = ['total_activity', 'unique_states', 'state_coverage']
            if 'avg_intrinsic_reward' in final_stats.columns:
                corr_cols.append('avg_intrinsic_reward')
            
            corr_matrix = final_stats[corr_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title="Metric Correlation Matrix"
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        APT Experiments v2 Dashboard | Data from mini_behavior environment
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()