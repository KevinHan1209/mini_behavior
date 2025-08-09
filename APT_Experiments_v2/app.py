"""
Streamlit Dashboard for APT Experiments v2
Deployable version with embedded data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="APT Experiments v2 Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Embedded data extracted from actual checkpoint CSV files
# This is the real experimental data from APT_Experiments_v2
EXPERIMENT_DATA = {
    "exp_000_default": {
        "label": "Default (k=12, ent=0.01)",
        "k": 12,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 143, "unique_states": 34, "state_coverage": 0.179},
            1000000: {"total_activity": 73, "unique_states": 30, "state_coverage": 0.158},
            1500000: {"total_activity": 128, "unique_states": 40, "state_coverage": 0.211},
            2000000: {"total_activity": 145, "unique_states": 35, "state_coverage": 0.184},
            2500000: {"total_activity": 143, "unique_states": 60, "state_coverage": 0.316}
        }
    },
    "exp_001_k_5": {
        "label": "k=5",
        "k": 5,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 130, "unique_states": 41, "state_coverage": 0.216},
            1000000: {"total_activity": 103, "unique_states": 35, "state_coverage": 0.184},
            1500000: {"total_activity": 131, "unique_states": 23, "state_coverage": 0.121},
            2000000: {"total_activity": 117, "unique_states": 42, "state_coverage": 0.221},
            2500000: {"total_activity": 126, "unique_states": 41, "state_coverage": 0.216}
        }
    },
    "exp_002_k_25": {
        "label": "k=25",
        "k": 25,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 129, "unique_states": 56, "state_coverage": 0.295},
            1000000: {"total_activity": 192, "unique_states": 44, "state_coverage": 0.232},
            1500000: {"total_activity": 131, "unique_states": 34, "state_coverage": 0.179},
            2000000: {"total_activity": 110, "unique_states": 31, "state_coverage": 0.163},
            2500000: {"total_activity": 109, "unique_states": 40, "state_coverage": 0.211}
        }
    },
    "exp_003_k_50": {
        "label": "k=50",
        "k": 50,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 150, "unique_states": 43, "state_coverage": 0.226},
            1000000: {"total_activity": 149, "unique_states": 31, "state_coverage": 0.163},
            1500000: {"total_activity": 186, "unique_states": 26, "state_coverage": 0.137},
            2000000: {"total_activity": 120, "unique_states": 26, "state_coverage": 0.137},
            2500000: {"total_activity": 5, "unique_states": 5, "state_coverage": 0.026}
        }
    },
    "exp_004_k_100": {
        "label": "k=100",
        "k": 100,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 39, "unique_states": 25, "state_coverage": 0.132},
            1000000: {"total_activity": 107, "unique_states": 36, "state_coverage": 0.189},
            1500000: {"total_activity": 129, "unique_states": 38, "state_coverage": 0.200},
            2000000: {"total_activity": 132, "unique_states": 29, "state_coverage": 0.153},
            2500000: {"total_activity": 101, "unique_states": 41, "state_coverage": 0.216}
        }
    },
    "exp_005_ent_coef_0": {
        "label": "ent_coef=0.0",
        "k": 12,
        "ent_coef": 0.0,
        "checkpoints": {
            500000: {"total_activity": 231, "unique_states": 51, "state_coverage": 0.268},
            1000000: {"total_activity": 191, "unique_states": 58, "state_coverage": 0.305},
            1500000: {"total_activity": 106, "unique_states": 28, "state_coverage": 0.147},
            2000000: {"total_activity": 79, "unique_states": 25, "state_coverage": 0.132},
            2500000: {"total_activity": 75, "unique_states": 31, "state_coverage": 0.163}
        }
    },
    "exp_006_ent_coef_0.005": {
        "label": "ent_coef=0.005",
        "k": 12,
        "ent_coef": 0.005,
        "checkpoints": {
            500000: {"total_activity": 122, "unique_states": 38, "state_coverage": 0.200},
            1000000: {"total_activity": 139, "unique_states": 43, "state_coverage": 0.226},
            1500000: {"total_activity": 132, "unique_states": 41, "state_coverage": 0.216},
            2000000: {"total_activity": 118, "unique_states": 29, "state_coverage": 0.153},
            2500000: {"total_activity": 119, "unique_states": 33, "state_coverage": 0.174}
        }
    },
    "exp_007_ent_coef_0.02": {
        "label": "ent_coef=0.02",
        "k": 12,
        "ent_coef": 0.02,
        "checkpoints": {
            500000: {"total_activity": 166, "unique_states": 35, "state_coverage": 0.184},
            1000000: {"total_activity": 120, "unique_states": 27, "state_coverage": 0.142},
            1500000: {"total_activity": 169, "unique_states": 38, "state_coverage": 0.200},
            2000000: {"total_activity": 152, "unique_states": 34, "state_coverage": 0.179},
            2500000: {"total_activity": 91, "unique_states": 31, "state_coverage": 0.163}
        }
    },
    "exp_008_ent_coef_0.05": {
        "label": "ent_coef=0.05",
        "k": 12,
        "ent_coef": 0.05,
        "checkpoints": {
            500000: {"total_activity": 115, "unique_states": 44, "state_coverage": 0.232},
            1000000: {"total_activity": 107, "unique_states": 43, "state_coverage": 0.226},
            1500000: {"total_activity": 130, "unique_states": 35, "state_coverage": 0.184},
            2000000: {"total_activity": 122, "unique_states": 28, "state_coverage": 0.147},
            2500000: {"total_activity": 100, "unique_states": 43, "state_coverage": 0.226}
        }
    },
    "exp_009_ent_coef_0.1": {
        "label": "ent_coef=0.1",
        "k": 12,
        "ent_coef": 0.1,
        "checkpoints": {
            500000: {"total_activity": 132, "unique_states": 38, "state_coverage": 0.200},
            1000000: {"total_activity": 128, "unique_states": 29, "state_coverage": 0.153},
            1500000: {"total_activity": 150, "unique_states": 31, "state_coverage": 0.163},
            2000000: {"total_activity": 154, "unique_states": 30, "state_coverage": 0.158},
            2500000: {"total_activity": 94, "unique_states": 34, "state_coverage": 0.179}
        }
    }
}

def load_embedded_data():
    """Convert embedded data to DataFrame format."""
    all_data = []
    
    for exp_name, exp_info in EXPERIMENT_DATA.items():
        for checkpoint, metrics in exp_info["checkpoints"].items():
            all_data.append({
                "experiment": exp_info["label"],
                "exp_name": exp_name,
                "checkpoint": checkpoint,
                "total_activity": metrics["total_activity"],
                "unique_states": metrics["unique_states"],
                "state_coverage": metrics["state_coverage"],
                "k": exp_info["k"],
                "ent_coef": exp_info["ent_coef"]
            })
    
    return pd.DataFrame(all_data)

def main():
    # Title and description
    st.title("🔬 APT Experiments v2: Interactive Dashboard")
    st.markdown("### Explore and compare exploration metrics across different APT configurations")
    
    # Load embedded data
    combined_df = load_embedded_data()
    
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
        value=checkpoint_options[-1] if checkpoint_options else 2500000,
        format_func=lambda x: f"{x/1e6:.1f}M steps"
    )
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Evolution Metrics", 
        "🎯 Final Performance", 
        "📊 Comparative Analysis",
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
                markers=True
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
                markers=True
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
                markers=True
            )
            fig_unique.update_layout(
                xaxis_tickformat=',.0f',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_unique, use_container_width=True)
            
            # Efficiency: Activity per unique state
            filtered_df['efficiency'] = filtered_df['total_activity'] / filtered_df['unique_states'].replace(0, np.nan)
            fig_efficiency = px.line(
                filtered_df,
                x='checkpoint',
                y='efficiency',
                color='experiment',
                title='Exploration Efficiency (Activity per State)',
                labels={'checkpoint': 'Training Steps', 'efficiency': 'Activities per State'},
                markers=True
            )
            fig_efficiency.update_layout(
                xaxis_tickformat=',.0f',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_efficiency, use_container_width=True)
    
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
                    color_continuous_scale="blues",
                    text='total_activity'
                )
                fig_bar_activity.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig_bar_activity.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar_activity, use_container_width=True)
            
            with col2:
                # Bar chart for state coverage
                fig_bar_coverage = px.bar(
                    final_data.sort_values('state_coverage', ascending=True),
                    x='state_coverage',
                    y='experiment',
                    orientation='h',
                    title=f'State Coverage at {checkpoint_options[-1]/1e6:.1f}M Steps',
                    labels={'state_coverage': 'State Coverage (%)', 'experiment': 'Experiment'},
                    color='state_coverage',
                    color_continuous_scale="greens",
                    text='state_coverage'
                )
                fig_bar_coverage.update_traces(
                    texttemplate='%{text:.1%}', 
                    textposition='outside'
                )
                fig_bar_coverage.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar_coverage, use_container_width=True)
            
            # Scatter plot
            fig_scatter = px.scatter(
                final_data,
                x='unique_states',
                y='total_activity',
                color='experiment',
                size='state_coverage',
                title='Exploration Trade-offs',
                labels={
                    'unique_states': 'Unique States Visited',
                    'total_activity': 'Total Activity Count'
                },
                hover_data=['state_coverage', 'k', 'ent_coef']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        st.header("Comparative Analysis")
        
        # Heatmap
        pivot_data = filtered_df.pivot_table(
            values='total_activity',
            index='experiment',
            columns='checkpoint',
            aggfunc='mean'
        )
        
        fig_heatmap = px.imshow(
            pivot_data,
            labels=dict(x="Checkpoint", y="Experiment", color="Total Activity"),
            x=[f"{c/1e6:.1f}M" for c in pivot_data.columns],
            y=pivot_data.index,
            color_continuous_scale="viridis",
            title="Total Activity Across Checkpoints"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Parameter correlation
        if len(final_data) > 3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_k = px.scatter(
                    final_data,
                    x='k',
                    y='total_activity',
                    title='Effect of k on Total Activity',
                    labels={'k': 'k parameter', 'total_activity': 'Total Activity'},
                    trendline="ols"
                )
                st.plotly_chart(fig_k, use_container_width=True)
            
            with col2:
                fig_ent = px.scatter(
                    final_data,
                    x='ent_coef',
                    y='total_activity',
                    title='Effect of Entropy Coefficient on Total Activity',
                    labels={'ent_coef': 'Entropy Coefficient', 'total_activity': 'Total Activity'},
                    trendline="ols"
                )
                st.plotly_chart(fig_ent, use_container_width=True)
    
    with tab4:
        st.header("Summary Statistics")
        
        final_stats = filtered_df[filtered_df['checkpoint'] == checkpoint_options[-1]]
        
        if not final_stats.empty:
            # Create summary table
            summary_df = final_stats[['experiment', 'total_activity', 'unique_states', 'state_coverage', 'k', 'ent_coef']].copy()
            summary_df['state_coverage'] = (summary_df['state_coverage'] * 100).round(1)
            summary_df = summary_df.sort_values('total_activity', ascending=False)
            
            st.dataframe(
                summary_df.style.highlight_max(
                    subset=['total_activity', 'unique_states', 'state_coverage'],
                    color='lightgreen'
                ).highlight_min(
                    subset=['total_activity', 'unique_states', 'state_coverage'],
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

if __name__ == "__main__":
    main()