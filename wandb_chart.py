import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

def find_global_max_y(dir_step_map):
    global_max_y = 0

    for tables_dir in dir_step_map.keys():
        pattern = re.compile(r"^episode_\d+_activity_.*\.table\.json$")
        table_files = [f for f in os.listdir(tables_dir) if pattern.match(f)]

        if not table_files:
            continue

        for filename in table_files:
            full_path = os.path.join(tables_dir, filename)

            with open(full_path, "r") as f:
                table_json = json.load(f)

            df = pd.DataFrame(table_json["data"], columns=table_json["columns"])

            excluded_states = ["infovofrobot", "inreachofrobot"]
            filtered_df = df[~df["state_name"].isin(excluded_states)]
            total_activity = filtered_df["activity_count"].sum()

            global_max_y = max(global_max_y, total_activity)

    return global_max_y

def process_directory(tables_dir, steps_label, global_max_y, output_root="img/rnd32x32"):

    output_dir = os.path.join(output_root, steps_label)
    os.makedirs(output_dir, exist_ok=True)

    pattern = re.compile(r"^episode_\d+_activity_.*\.table\.json$")
    table_files = [
        f for f in os.listdir(tables_dir)
        if pattern.match(f)
    ]

    if not table_files:
        print(f"No matching .table.json files found in: {tables_dir}")
        return

    print(f"Found {len(table_files)} files in {tables_dir}")
    
    dfs = []
    for filename in table_files:
        full_path = os.path.join(tables_dir, filename)
        
        with open(full_path, "r") as f:
            table_json = json.load(f)
        
        columns = table_json["columns"]  # e.g. ["flag_id", "object_type", "state_name", "activity_count"]
        data = table_json["data"]        # e.g. [[val1, val2, ...], ...]
        df = pd.DataFrame(data, columns=columns)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    excluded_states = ["infovofrobot", "inreachofrobot"]
    filtered_df = combined_df[~combined_df["state_name"].isin(excluded_states)]

    total_activities = filtered_df["activity_count"].sum()

    grouped_df = filtered_df.groupby(["object_type", "state_name"], as_index=False)["activity_count"].sum()
    pivot_df = grouped_df.pivot(
        index="object_type",
        columns="state_name",
        values="activity_count"
    ).fillna(0)

    plt.figure(figsize=(10, 6))
    pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6)
    )

    plt.title(f"Combined Activity Counts {steps_label} Steps (Total = {total_activities})")
    plt.xlabel("Object Type")
    plt.ylabel("Activity Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="State Name")
    plt.ylim(0, global_max_y)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "combined_bar_chart_stacked.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

def main():

    dir_step_map8x8 = {
        "wandb/run-20250226_161228-wujjib4o/files/media/table": "1M",
        "wandb/run-20250226_153950-yjtx8jn3/files/media/table": "1.5M",
        "wandb/run-20250226_152956-r9bjaghg/files/media/table": "2M",
        "wandb/run-20250226_151804-35hk76zq/files/media/table": "2.5M",
        "wandb/run-20250226_145733-f67hu2ym/files/media/table": "3M",
        "wandb/run-20250308_141926-dplydfdj/files/media/table": "100K",
        "wandb/run-20250310_222440-oyqkbba0/files/media/table": "200K",
        "wandb/run-20250310_223334-wm6147cc/files/media/table": "300K",
        "wandb/run-20250310_224151-pkvejzom/files/media/table": "400K",
        "wandb/run-20250310_225022-oz3yyci9/files/media/table": "500K",
        "wandb/run-20250310_230512-33hg8inw/files/media/table": "600K",
        "wandb/run-20250310_231423-a0dd1m1d/files/media/table": "700K",
        "wandb/run-20250310_232235-yk2kz7kd/files/media/table": "800K",
        "wandb/run-20250310_233055-82z75wxd/files/media/table": "900K",
        "wandb/run-20250311_113947-snrpkwkt/files/media/table": "0"
    }

    dir_step_map = {
        "wandb/run-20250311_125815-7ghfto8y/files/media/table": "0",
        "wandb/run-20250311_125214-dkipt8dh/files/media/table": "100K",
        "wandb/run-20250311_124630-8uxl4u0z/files/media/table": "1M",
        "wandb/run-20250311_124052-7khks0cp/files/media/table": "2M",
        "wandb/run-20250311_123502-egepa1md/files/media/table": "3M"
    }

    global_max_y = find_global_max_y(dir_step_map)

    for tables_dir, steps_label in dir_step_map.items():
        process_directory(tables_dir, steps_label, global_max_y)

if __name__ == "__main__":
    main()
