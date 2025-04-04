import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def plot_curve(log_dicts, json_logs, keys, title, legend, backend, style, out):
    """Plots curves based on the provided log dictionaries and parameters.

    Args:
        log_dicts (list): List of dictionaries containing log data.
        json_logs (list): List of paths to the JSON log files.
        keys (list): List of keys (metrics) to plot.
        title (str): Title of the plot.
        legend (list): List of legend labels.
        backend (str): Backend for matplotlib (e.g., 'Agg', None).
        style (str): Style for seaborn (e.g., 'dark', 'whitegrid').
        out (str): Output path to save the plot, or None to display it.
    """
    if backend is not None:
        plt.switch_backend(backend)
    sns.set_style(style)

    # if legend is None, use {filename}_{key} as legend
    if legend is None:
        legend = []
        for json_log in json_logs:
            for metric in keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(json_logs) * len(keys))
    metrics = keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {json_logs[i]}, metric is {metric}')
            plot_epochs = []
            plot_iters = []
            plot_values = []
            # In some log files exist lines of validation,
            # `mode` list is used to only collect iter number
            # of training line.
            for epoch in epochs:
                epoch_logs = log_dict[epoch]
                if metric not in epoch_logs.keys():
                    continue
                if metric in ['mIoU', 'mAcc', 'aAcc']:
                    plot_epochs.append(epoch)
                    plot_values.append(epoch_logs[metric][0])
                else:
                    for idx in range(len(epoch_logs[metric])):
                        plot_iters.append(epoch_logs['step'][idx])
                        plot_values.append(epoch_logs[metric][idx])
            ax = plt.gca()
            label = legend[i * num_metrics + j]
            if metric in ['mIoU', 'mAcc', 'aAcc']:
                ax.set_xticks(plot_epochs)
                plt.xlabel('step')
                plt.plot(plot_epochs, plot_values, label=label, marker='o')
            else:
                plt.xlabel('iter')
                plt.plot(plot_iters, plot_values, label=label, linewidth=0.5)
        plt.legend()
        if title is not None:
            plt.title(title)
    if out is None:
        plt.show()
    else:
        print(f'save curve to: {out}')
        plt.savefig(out)
        plt.cla()


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


def main(json_logs, keys_to_plot=['mIoU'], plot_title=None, plot_legend=None, matplotlib_backend=None, seaborn_style='dark', output_path=None):
    """Main function to analyze and plot data from JSON log files.

    Args:
        json_logs (list): List of paths to the JSON log files.
        keys (list): List of keys (metrics) to plot (default: ['mIoU']).
        title (str): Title of the plot (default: None).
        legend (list): List of legend labels (default: None).
        backend (str): Backend for matplotlib (default: None).
        style (str): Style for seaborn (default: 'dark').
        out (str): Output path to save the plot (default: None).
    """
    
    for json_log in json_logs:
        assert json_log.endswith('.json')
    log_dicts = load_json_logs(json_logs)
    plot_curve(log_dicts, json_logs, keys_to_plot, plot_title, plot_legend, matplotlib_backend, seaborn_style, output_path)

# Example Usage (outside the script, or at the end of the file if you're running it directly):
if __name__ == "__main__":
    # Replace with your actual log file paths
    json_logs = ['/cluster/projects/nn10004k/packages_install/seaobject_ocrnet80000/20250325_143528/vis_data/20250325_143528.json'] 
    
    # you can change these variable to get other result
    keys_to_plot = ['loss']   #['mIoU','mAcc','aAcc']
    plot_title = "Training Metrics"
    plot_legend =['loss']  # ['mIoU','mAcc','aAcc'] ['loss'] #If not None, should match len(json_logs)*len(keys_to_plot)
    matplotlib_backend = None  
    seaborn_style = 'whitegrid'
    output_path = '/cluster/home/snf52395/mmsegmentation/data/loss_adamwFocal_ocrnet_80000_p100_b1.png'
    main(json_logs=json_logs,
        keys_to_plot=keys_to_plot,
        plot_title=plot_title,
        plot_legend=plot_legend,
        plot_legend=matplotlib_backend,
        seaborn_style=seaborn_style,
        output_path=output_path)

