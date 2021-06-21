import csv
import pathlib
import cpuinfo

import argparse

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Manage command line
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
def get_cmd_args():
    parser = argparse.ArgumentParser(description="Generate commands for parallel launch")
    parser.add_argument("N", help="number of threads", type=int)
    args = parser.parse_args()
    return (parser, args)

# --- --- --- --- Load configuration
import importlib.util
spec = importlib.util.spec_from_file_location("CONFIGURE_ME", "../CONFIGURE_ME.py")
CONFIGURE_ME = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIGURE_ME)

# --- --- --- --- Generate command for one CSV record
def generate_cpp_cmd(EXEC_PATH, UCR_PATH, RESULT_DIR, record, sampling_kind, sampling_number, pargs, nbthreads, output):
    # Extract all components
    name, \
    cdtw_w, \
    wdtw_weight, \
    __ddtw_w, \
    __wddtw_weight, \
    lcss_w, lcss_epsilon, \
    msm_cost, \
    twe_nu, twe_lambda, \
    erp_g, erp_w = record

    def do_cmd(transform):
        fname = f"{name}-adtw-{transform}-{sampling_kind}-{sampling_number}-{pargs}.json"
        return str(EXEC_PATH)+f" {UCR_PATH} {name} {sampling_kind} {sampling_number} {transform} {pargs} {nbthreads} {RESULT_DIR}/{fname}"

    print(do_cmd("original"), file=output)
    print(do_cmd("derivative"), file=output)


# --- --- --- --- Main
if __name__ == '__main__':
    (parser, args) = get_cmd_args()
    nbthreads = args.N

    EE_PATH = pathlib.Path("../eeOutputFold0.csv").absolute()
    timestamp = CONFIGURE_ME.get_timestemp()
    OUT_DIR = pathlib.Path("generated-"+timestamp).absolute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_CMD = (OUT_DIR/"commands").absolute()
    OUT = open(OUT_CMD, "w")

    RES_DIR = OUT_DIR/"results"
    RES_DIR.mkdir(parents=True, exist_ok=True)

    UCR_TS_PATH = str(CONFIGURE_ME.get_ucr_folder())
    print("UCR FOLDER = ", UCR_TS_PATH)

    EXEC_PATH = pathlib.Path("./loocv_adtw/cmake-build-release/loocv_adtw").absolute()

    # --- --- --- --- Commands
    # Sampling configuration
    sampling_conf = [('points', 100000, "0:250:1:100"), ('dtw', 1000, "0:250:1:100"), ('sqed', 1000, "0:250:1:100")]
    with open(EE_PATH, newline='') as csvfile:
        records = csv.reader(csvfile)
        header = next(records)          # Skip header
        print(header)
        for r in records:
            for (sampling_kind, sampling_nb, pargs) in sampling_conf:               # Sampling kind
                generate_cpp_cmd(EXEC_PATH, UCR_TS_PATH, RES_DIR, r, sampling_kind, sampling_nb, pargs, nbthreads, OUT)

    # --- --- --- --- CPU INFO
    # CPU_INFO file
    OUT_CPU = (OUT_DIR/"cpu.json").absolute()
    OUT_CPU = open(OUT_CPU, "w")
    print(cpuinfo.get_cpu_info_json(), file=OUT_CPU)

