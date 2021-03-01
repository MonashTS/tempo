import os
import argparse
import matplotlib.pyplot as plt
import pathlib as path
import numpy as np
import json
import datetime
import math

import pprint
pp = pprint.PrettyPrinter()

# Get the paths of all elements from a directory
def get_file_paths(directory):
    return [f for f in path.Path(directory).iterdir() if f.is_file()]

# Read all json file from a directory, return a list of dictionary
def read_all_in(directory):
    #
    results = []
    #
    all_file_paths = get_file_paths(directory)
    print(f"Found {len(all_file_paths)} files...")
    for fp in all_file_paths:
        with open(fp) as f:
            try:
                data = json.load(f)
                results.append(data)
            except json.decoder.JSONDecodeError as e:
                print("Could not read file " + str(fp) +": " + str(e))
    #
    return results


# From the doc https://matplotlib.org/examples/api/barchart_demo.html
def autolabel(ax, rects, fontsize):
    for rect in rects:
        height = rect.get_height()
        txt = f"{height:2.2f}"
        ax.text(rect.get_x() + rect.get_width() / 2., height / 2, txt, ha='center', va='bottom',
                fontsize=fontsize,
                color=(0, 0, 0),
                backgroundcolor=(0.9, 0.9, 0.9))


# Create bar plot


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Analyse 2021-03-LB-DTW results")
    parser.add_argument("folder", help="Folder containing the results")
    args = parser.parse_args()

    # Manage input/output folders
    RESULT_DIR = path.Path(args.folder).absolute()
    PARENT_DIR = RESULT_DIR.parent
    OUT_DIR = PARENT_DIR/"analysis"
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    # Read all the results
    json_results = read_all_in(RESULT_DIR)

    # Read CPU info
    with open(PARENT_DIR/"cpu.json") as f:
        cpu_json = json.load(f)
        pp.pprint(cpu_json)
        cpu_info = f"{cpu_json['brand_raw']} -- {cpu_json['hz_actual_friendly']}"


    # Iterate over the results, aggregate by distance, mode, lb
    results_cdtw = {}   # CDTW results
    results_dtw = {}    # DTW results
    lbs = set()         # Set of discovered lower bound
    for r in json_results:
        # Switch our target dictionary according to the distance
        target_dic = results_cdtw
        if r['distance']['name'] == "dtw":
            target_dic = results_dtw

        # aggregate by lower bound
        lb_name = r['distance']['lb']
        # Bounds with parameter
        if lb_name.startswith("lb-enhanced"):
            v = r['distance']['v']
            lb_name = lb_name + f"_{v}"
        lbs.add(lb_name)
        lb_dic = target_dic.setdefault(lb_name, {})

        # sum time
        lb_total_runtime = lb_dic.setdefault('total_time_ns', int(0))
        runtime = r['timing_ns']
        total = lb_total_runtime + runtime
        lb_dic['total_time_ns'] = total
        lb_dic['total_time'] = str(datetime.timedelta(microseconds=total / 1e3))

    pp.pprint(results_dtw)

    # For each distance, do bar plots of lower bounds
    def barplot(dname, ddic):
        # convert data in plottable tuple (lower bound name, time in minutes)
        data_to_plot = []
        for lb in lbs:
            in_ns = ddic[lb]['total_time_ns']
            in_minutes = float(in_ns)/(1e9*60)
            data_to_plot.append((lb, in_minutes))

        # Sort by decreasing time
        data_to_plot = sorted(data_to_plot, key=lambda k: k[1], reverse=True)
        (lbnames, lbtimes) = zip(*data_to_plot)

        # Do the plot
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(1, 1, 1)
        bars = ax.bar(lbnames, lbtimes)

        # Annotations
        fontsize = 24
        fontsize_label = 16
        ax.set_xlabel("lower bounds", fontsize=fontsize_label, labelpad=15)
        ax.set_ylabel("runtime in minutes", fontsize=fontsize_label, labelpad=10)
        ax.set_title(f"{dname}-NN1 on UCR Archive\n85 datasets - parameter from EE\n({cpu_info})", fontsize=fontsize)

        # Add timing in the bars
        autolabel(ax, bars, fontsize_label)
        ax.tick_params(axis='both', labelsize=fontsize_label)

        # Write the file
        ax.autoscale_view()
        fig.tight_layout()
        output_path = os.path.join(OUT_DIR, f"{dname}")
        fig.savefig(output_path + ".pdf", bbox_inches='tight', pad_inches=0)
        fig.savefig(output_path + ".png", bbox_inches='tight', pad_inches=0)

    barplot("CDTW", results_cdtw)
    barplot("DTW", results_dtw)


    #    all_bars = [item for sublist in group_bars for item in sublist]


    #    ylabel = "runtime in hours"

    #    ax.legend(tuple([bg[0] for bg in group_bars]), tuple(group_lb), fontsize=fontsize)




    ## --- ---

    ## CDTW
    #distname = "CDTW"
    #title = f"{distname} runtime in minutes per mode and lower bound\n85 UCR Datasets, window parameter from EE"
    #analysis(results_cdtw, distname, title)

    ## DTW
    #distname = "DTW"
    #title = f"{distname} runtime in minutes per mode and lower bound\n85 UCR Datasets"
    #analysis(results_dtw, distname, title)


