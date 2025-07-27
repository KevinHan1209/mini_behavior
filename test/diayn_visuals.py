import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re

def load_and_merge_skill_tables(folder_path):
    all_dfs = []

    for fname in os.listdir(folder_path):
        if fname.startswith("skill_") and "total_activity" in fname and fname.endswith(".table.json"):
            match = re.search(r"skill_(\d+)_total_activity", fname)
            if not match:
                continue
            skill = int(match.group(1))
            fpath = os.path.join(folder_path, fname)

            with open(fpath, 'r') as f:
                content = json.load(f)
                df = pd.DataFrame(content['data'], columns=content['columns'])
                df['skill'] = skill
                all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df

# Load CSVs
df1M = load_and_merge_skill_tables('wandb/run-20250506_132945-zd0guduu/files/media/table')
df500K = load_and_merge_skill_tables('wandb/run-20250506_124328-si4iyco0/files/media/table')
df100K = load_and_merge_skill_tables('wandb/run-20250506_111212-hq38mupc/files/media/table')
df0 = load_and_merge_skill_tables('wandb/run-20250506_115809-zsg4vz5t/files/media/table')

exclude_states = {'infovofrobot','inreachofrobot', 'inleftreachofrobot', 'inrightreachofrobot'}

def find_max_in_dataset(df):
    df = df[~df['state_name'].isin(exclude_states)]
    max_val = 0
    for skill in df['skill'].unique():
        subset = df[df['skill'] == skill]
        pivot = subset.pivot_table(
            index='object_type',
            columns='state_name',
            values='activity_count',
            aggfunc='sum',
            fill_value=0
        )
        pivot['stack_total'] = pivot.sum(axis=1)
        local_max = pivot['stack_total'].max()
        if local_max > max_val:
            max_val = local_max

    df_all = df.groupby(['object_type','state_name'], as_index=False)['activity_count'].sum()
    pivot_all = df_all.pivot_table(
        index='object_type',
        columns='state_name',
        values='activity_count',
        aggfunc='sum',
        fill_value=0
    )
    pivot_all['stack_total'] = pivot_all.sum(axis=1)
    overall_max = pivot_all['stack_total'].max()
    return max(max_val, overall_max)

def find_max_per_skill_across_datasets(datasets):
    skill_max = {}
    for df in datasets:
        df = df[~df['state_name'].isin(exclude_states)]
        for skill in df['skill'].unique():
            subset = df[df['skill'] == skill]
            pivot = subset.pivot_table(
                index='object_type',
                columns='state_name',
                values='activity_count',
                aggfunc='sum',
                fill_value=0
            )
            pivot['stack_total'] = pivot.sum(axis=1)
            local_max = pivot['stack_total'].max()
            if skill not in skill_max or local_max > skill_max[skill]:
                skill_max[skill] = local_max
    return skill_max

def plot_stacked(df, out_dir, prefix, global_max, skill_max_dict):
    os.makedirs(out_dir, exist_ok=True)

    df = df[~df['state_name'].isin(exclude_states)].copy()
    

    all_states = sorted(df['state_name'].unique())
    cmap = cm.get_cmap('tab20', len(all_states))  # Use a colormap with at least 20 colors
    color_map = {state: mcolors.to_hex(cmap(i)) for i, state in enumerate(all_states)}

    def apply_colors(ax, pivot_cols):
        handles, labels = ax.get_legend_handles_labels()
        for bar, label in zip(ax.patches, labels * (len(ax.patches) // len(labels))):
            bar.set_color(color_map.get(label, '#000000'))
        return ax

    df_all = df.groupby(['object_type','state_name'], as_index=False)['activity_count'].sum()
    pivot_all = df_all.pivot_table(
        index='object_type',
        columns='state_name',
        values='activity_count',
        aggfunc='sum',
        fill_value=0
    )
    ax = pivot_all.plot(kind='bar', stacked=True, figsize=(8,6), color=[color_map[c] for c in pivot_all.columns])
    ax.set_ylim([0, global_max])
    ax.set_ylabel("Activity Count")
    ax.set_xlabel("Object Type")
    ax.set_title(f"{prefix} - All Skills Combined")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="State Name", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_all_skills.png"))
    plt.close()

    for skill in sorted(df['skill'].unique()):
        df_skill = df[df['skill'] == skill]
        pivot = df_skill.pivot_table(
            index='object_type',
            columns='state_name',
            values='activity_count',
            aggfunc='sum',
            fill_value=0
        )
        ax = pivot.plot(kind='bar', stacked=True, figsize=(8,6), color=[color_map[c] for c in pivot.columns])
        ax.set_ylim([0, skill_max_dict.get(skill, 1)])
        ax.set_ylabel("Activity Count")
        ax.set_xlabel("Object Type")
        ax.set_title(f"{prefix} - Skill {skill}")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="State Name", bbox_to_anchor=(1.0, 1.0))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_skill_{skill}.png"))
        plt.close()


all_dfs = [df1M, df500K, df100K, df0]

# Global max for combined plot
global_max = max(find_max_in_dataset(df) for df in all_dfs)
print("Global max y-value across all:", global_max)

# Skill-specific max y-values
skill_max_dict = find_max_per_skill_across_datasets(all_dfs)

plot_stacked(df1M,   'img/diayn_8x8_new_env/1000000',   '1M',   global_max, skill_max_dict)
plot_stacked(df500K, 'img/diayn_8x8_new_env/500000', '500K', global_max, skill_max_dict)
plot_stacked(df100K,   'img/diayn_8x8_new_env/100000',   '100K',   global_max, skill_max_dict)
plot_stacked(df0, 'img/diayn_8x8_new_env/0', '0', global_max, skill_max_dict)
