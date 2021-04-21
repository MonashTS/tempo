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

# Create an entry for a dataset name in an output dictionary.  Guarantees the same format between java and cpp results
def mk_entry(results_dic, name, accuracy, train_ms, test_ms):
    total_ms = train_ms + test_ms
    dic = results_dic.setdefault(name, {})
    dic["accuracy"] = accuracy
    dic["train_ms"] = train_ms
    dic["test_ms"] = test_ms
    dic["total_ms"] = total_ms
    dic['train'] = str(datetime.timedelta(milliseconds=train_ms))
    dic['test_time'] = str(datetime.timedelta(milliseconds=test_ms))
    dic['total_time'] = str(datetime.timedelta(milliseconds=total_ms))


# Read output from the JAVA code, return a dictionary
def java_read_all_in(directory):
    results = {}
    all_file_paths = get_file_paths(directory)
    print(f"JAVA output: found {len(all_file_paths)} files")
    for fp in all_file_paths:
        with open(fp) as f:
            for l in f:
                if l.startswith("REPEAT"):
                    cols = [x.strip() for x in l.split(",")]
                    name = cols[1][:-11]            # remove "_TRAIN.arff", ie 11 char
                    accuracy = float(cols[2])
                    train_ms = int(float(cols[3]))
                    test_ms = int(float(cols[4]))
                    mk_entry(results, name, accuracy, train_ms, test_ms)
    return results

# Read all json file from a directory, return a dictionary
def cpp_read_all_in(directory):
    results = {}
    all_file_paths = get_file_paths(directory)
    print(f"Found {len(all_file_paths)} files...")
    for fp in all_file_paths:
        with open(fp) as f:
            try:
                data = json.load(f)
                name = data["train_set"]["identifier"][:-6] # Remove "_TRAIN"
                accuracy = data["Accuracy"]
                train_ms = int(data["train_time_ns"]/1e6)
                test_ms = int(data["test_time_ns"]/1e6)
                total_ms = train_ms + test_ms
                mk_entry(results, name, accuracy, train_ms, test_ms)
            except json.decoder.JSONDecodeError as e:
                print("Could not read file " + str(fp) +": " + str(e))
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
    parser = argparse.ArgumentParser(description="Analyse 2021-04-PF2018-comparison")
    parser.add_argument("folder", help="Folder containing the cpp_results and java_results folders")
    args = parser.parse_args()

    # Manage input/output folders
    RESULT_DIR = path.Path(args.folder).absolute()
    CPP_RESULT_DIR = RESULT_DIR/"cpp_results"
    JAVA_RESULT_DIR = RESULT_DIR/"java_results"
    OUT_DIR = RESULT_DIR/"analysis"
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)

    # Read all the results
    java_results = java_read_all_in(JAVA_RESULT_DIR)
    cpp_results = cpp_read_all_in(CPP_RESULT_DIR)
    # Merge the results
    results = {}
    cpp_train_ms = 0
    cpp_test_ms = 0
    cpp_total_ms = 0
    java_train_ms = 0
    java_test_ms = 0
    java_total_ms = 0
    for (name, java_data) in java_results.items():
        if name in cpp_results:
            cpp_data = cpp_results[name]
            dic = results.setdefault(name, {})
            dic["java"] = java_data
            java_train_ms += java_data["train_ms"]
            java_test_ms += java_data["test_ms"]
            java_total_ms += java_data["total_ms"]
            dic["cpp"] = cpp_data
            cpp_train_ms += cpp_data["train_ms"]
            cpp_test_ms += cpp_data["test_ms"]
            cpp_total_ms += cpp_data["total_ms"]
    print(f"Reconciled results for {len(results)} datasets")
    print("JAVA:")
    print("  Train:",str(datetime.timedelta(milliseconds=java_train_ms)))
    print("  Test:", str(datetime.timedelta(milliseconds=java_test_ms)))
    print("  Total:", str(datetime.timedelta(milliseconds=java_total_ms)))
    print("CPP:")
    print("  Train:", str(datetime.timedelta(milliseconds=cpp_train_ms)))
    print("  Test:", str(datetime.timedelta(milliseconds=cpp_test_ms)))
    print("  Total:", str(datetime.timedelta(milliseconds=cpp_total_ms)))
    print("SPEEDUP:")
    print("  Train:", java_train_ms/cpp_train_ms)
    print("  Test:", java_test_ms/cpp_test_ms)
    print("  Total:", java_total_ms/cpp_total_ms)

    # Read CPU info
    with open(RESULT_DIR/"cpu.json") as f:
        cpu_json = json.load(f)
        #pp.pprint(cpu_json)
        cpu_info = f"{cpu_json['brand_raw']} -- {cpu_json['hz_actual_friendly']}"
        print("CPU Info:", cpu_info)