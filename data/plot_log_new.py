import json
import os
from collections import defaultdict
import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.font_manager as fm
font_dirs = ['/cluster/home/snf52395/mmsegmentation']
font_files = fm.findSystemFonts(fontpaths=font_dirs, fontext='ttf')
for font_file in font_files:
    print(font_file) if 'TimesNewRoman' in font_file else None
    fm.fontManager.addfont(font_file)


def load_json_logs(json_logs):
    """Loads and processes JSON log files into a list of dictionaries.

    Args:
        json_logs (list): List of paths to the JSON log files.

    Returns:
        list: List of dictionaries, where each dictionary represents a log file.
    """
    log_dicts = [dict() for _ in json_logs]
    prev_step = 0
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log) as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # the final step in json file is 0.
                if 'step' in log and log['step'] != 0:
                    step = log['step']
                    prev_step = step
                else:
                    step = prev_step
                if step not in log_dict:
                    log_dict[step] = defaultdict(list)
                for k, v in log.items():
                    log_dict[step][k].append(v)
    return log_dicts

def logdicts2df(log_dicts):
    # Check if log_dicts has at least one log processed
    if not log_dicts:
        print("Error: No logs were processed.")
    else:
        # Target the first log dictionary
        target_log_dict = log_dicts[0]

        # Define the metrics we want to extract
        metrics_to_extract = [
            'loss', 'aAcc', 'mIoU', 'mAcc', 'mDice',
            'mFscore', 'mPrecision', 'mRecall'
        ]

        # Extract steps (keys) and sort them
        steps = sorted(target_log_dict.keys())

        # Prepare data for DataFrame construction
        data_for_df = []
        for step in steps:
            step_data = target_log_dict[step]
            row_data = {'step': step} # Start row with the step number

            # Extract each metric, using pd.NA if missing or empty
            for metric in metrics_to_extract:
                if metric in step_data and step_data[metric]:
                    # Take the first value from the list for this metric
                    row_data[metric] = step_data[metric][0]
                else:
                    # Use pandas' preferred NA value
                    row_data[metric] = pd.NA

            data_for_df.append(row_data)

        # Create the Pandas DataFrame from the list of dictionaries
        df_metrics = pd.DataFrame(data_for_df)
        return df_metrics


# Replace with your actual log file paths
jason_folder = '/cluster/projects/nn10004k/packages_install/SAM2.1_tiny_OASIs_3in1_40K/20251113_104942'
json_filename = f"{Path(jason_folder).name}.json"
expected_json_path = jason_folder + '/vis_data/' + json_filename
json_logs = [str(expected_json_path)]

for json_log in json_logs:
    assert json_log.endswith('.json')

log_dicts = load_json_logs(json_logs)

path_obj = Path(jason_folder)
project_name = path_obj.parent.name
output_dir = Path("training_metrics")
output_dir.mkdir(parents=True, exist_ok=True)

df_metrics = logdicts2df(log_dicts)

file_name = os.path.join(output_dir, f"{project_name}.csv")
df_metrics.to_csv(file_name, index=False)


sns.set_style({'font.family': 'Times New Roman','font.size': 10})

#=============== plot loss============
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax = sns.lineplot(data=df_metrics, x='step', y='loss', linewidth=0.5,
                  color='black') # Adjust linewidth if needed
ax.set_xlabel("Step")
ax.set_ylabel("Loss")

ax.tick_params(top=True, right=True)
ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)
plt.tight_layout(pad=0.5) # Adjust padding if necessary

# plt.show()
output_filename =  os.path.join(output_dir, f"loss_{project_name}.png")
fig.savefig(output_filename, dpi=600, bbox_inches='tight')
plt.close()
#============= plot 'mIoU','mDice','mFscore', 'mPrecision', 'mRecall'

df_accuracy = pd.melt(df_metrics, id_vars=['step'], 
                      value_vars=['mIoU','mDice','mFscore'])

#drop nan
df_accuracy = df_accuracy.dropna()
fig, ax = plt.subplots(figsize=(3.5, 2.7))

sizes = {'mIoU': 0.8, 'mDice': 1.5, 'mFscore': 0.5}
ax = sns.lineplot(data=df_accuracy, x='step', y='value', hue='variable', #linewidth=0.8,
                  size='variable', sizes=sizes,
                  palette='tab10') # Adjust linewidth if needed



ax.set_xlabel("Step")
ax.set_ylabel("Validation metrics (%)")

ax.tick_params(top=True, right=True)
ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='',  loc='lower right', frameon=False)

plt.tight_layout(pad=0.5) # Adjust padding if necessary

# plt.show()
output_filename =  os.path.join(output_dir,f"vali_{project_name}.png")
fig.savefig(output_filename, dpi=600, bbox_inches='tight')
plt.close()

                     