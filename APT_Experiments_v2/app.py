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

# Embedded data - this would normally come from CSV files
# For deployment, we're embedding the key data directly
EXPERIMENT_DATA = {
    "exp_000_default": {
        "label": "Default (k=12, ent=0.01)",
        "k": 12,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 143, "unique_states": 60, "state_coverage": 0.316},
            1000000: {"total_activity": 165, "unique_states": 62, "state_coverage": 0.326},
            1500000: {"total_activity": 180, "unique_states": 64, "state_coverage": 0.337},
            2000000: {"total_activity": 195, "unique_states": 65, "state_coverage": 0.342},
            2500000: {"total_activity": 210, "unique_states": 66, "state_coverage": 0.347}
        }
    },
    "exp_001_k_5": {
        "label": "k=5",
        "k": 5,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 130, "unique_states": 55, "state_coverage": 0.289},
            1000000: {"total_activity": 148, "unique_states": 57, "state_coverage": 0.300},
            1500000: {"total_activity": 162, "unique_states": 58, "state_coverage": 0.305},
            2000000: {"total_activity": 175, "unique_states": 59, "state_coverage": 0.311},
            2500000: {"total_activity": 188, "unique_states": 60, "state_coverage": 0.316}
        }
    },
    "exp_002_k_25": {
        "label": "k=25",
        "k": 25,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 129, "unique_states": 54, "state_coverage": 0.284},
            1000000: {"total_activity": 145, "unique_states": 56, "state_coverage": 0.295},
            1500000: {"total_activity": 158, "unique_states": 57, "state_coverage": 0.300},
            2000000: {"total_activity": 170, "unique_states": 58, "state_coverage": 0.305},
            2500000: {"total_activity": 182, "unique_states": 59, "state_coverage": 0.311}
        }
    },
    "exp_003_k_50": {
        "label": "k=50",
        "k": 50,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 150, "unique_states": 58, "state_coverage": 0.305},
            1000000: {"total_activity": 168, "unique_states": 60, "state_coverage": 0.316},
            1500000: {"total_activity": 183, "unique_states": 61, "state_coverage": 0.321},
            2000000: {"total_activity": 196, "unique_states": 62, "state_coverage": 0.326},
            2500000: {"total_activity": 208, "unique_states": 63, "state_coverage": 0.332}
        }
    },
    "exp_004_k_100": {
        "label": "k=100",
        "k": 100,
        "ent_coef": 0.01,
        "checkpoints": {
            500000: {"total_activity": 39, "unique_states": 28, "state_coverage": 0.147},
            1000000: {"total_activity": 52, "unique_states": 32, "state_coverage": 0.168},
            1500000: {"total_activity": 63, "unique_states": 35, "state_coverage": 0.184},
            2000000: {"total_activity": 72, "unique_states": 37, "state_coverage": 0.195},
            2500000: {"total_activity": 80, "unique_states": 39, "state_coverage": 0.205}
        }
    },
    "exp_005_ent_coef_0": {
        "label": "ent_coef=0.0",
        "k": 12,
        "ent_coef": 0.0,
        "checkpoints": {
            500000: {"total_activity": 231, "unique_states": 68, "state_coverage": 0.358},
            1000000: {"total_activity": 252, "unique_states": 70, "state_coverage": 0.368},
            1500000: {"total_activity": 268, "unique_states": 71, "state_coverage": 0.374},
            2000000: {"total_activity": 282, "unique_states": 72, "state_coverage": 0.379},
            2500000: {"total_activity": 294, "unique_states": 73, "state_coverage": 0.384}
        }
    },
    "exp_006_ent_coef_0.005": {
        "label": "ent_coef=0.005",
        "k": 12,
        "ent_coef": 0.005,
        "checkpoints": {
            500000: {"total_activity": 122, "unique_states": 52, "state_coverage": 0.274},
            1000000: {"total_activity": 138, "unique_states": 54, "state_coverage": 0.284},
            1500000: {"total_activity": 151, "unique_states": 55, "state_coverage": 0.289},
            2000000: {"total_activity": 163, "unique_states": 56, "state_coverage": 0.295},
            2500000: {"total_activity": 174, "unique_states": 57, "state_coverage": 0.300}
        }
    },
    "exp_007_ent_coef_0.02": {
        "label": "ent_coef=0.02",
        "k": 12,
        "ent_coef": 0.02,
        "checkpoints": {
            500000: {"total_activity": 166, "unique_states": 61, "state_coverage": 0.321},
            1000000: {"total_activity": 184, "unique_states": 63, "state_coverage": 0.332},
            1500000: {"total_activity": 199, "unique_states": 64, "state_coverage": 0.337},
            2000000: {"total_activity": 212, "unique_states": 65, "state_coverage": 0.342},
            2500000: {"total_activity": 224, "unique_states": 66, "state_coverage": 0.347}
        }
    },
    "exp_008_ent_coef_0.05": {
        "label": "ent_coef=0.05",
        "k": 12,
        "ent_coef": 0.05,
        "checkpoints": {
            500000: {"total_activity": 115, "unique_states": 50, "state_coverage": 0.263},
            1000000: {"total_activity": 130, "unique_states": 52, "state_coverage": 0.274},
            1500000: {"total_activity": 142, "unique_states": 53, "state_coverage": 0.279},
            2000000: {"total_activity": 153, "unique_states": 54, "state_coverage": 0.284},
            2500000: {"total_activity": 163, "unique_states": 55, "state_coverage": 0.289}
        }
    },
    "exp_009_ent_coef_0.1": {
        "label": "ent_coef=0.1",
        "k": 12,
        "ent_coef": 0.1,
        "checkpoints": {
            500000: {"total_activity": 132, "unique_states": 56, "state_coverage": 0.295},
            1000000: {"total_activity": 148, "unique_states": 58, "state_coverage": 0.305},
            1500000: {"total_activity": 161, "unique_states": 59, "state_coverage": 0.311},
            2000000: {"total_activity": 173, "unique_states": 60, "state_coverage": 0.316},
            2500000: {"total_activity": 184, "unique_states": 61, "state_coverage": 0.321}
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