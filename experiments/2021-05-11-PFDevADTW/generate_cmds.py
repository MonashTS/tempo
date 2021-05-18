import csv
import pathlib
import cpuinfo

# --- --- --- --- Load configuration
import importlib.util
spec = importlib.util.spec_from_file_location("CONFIGURE_ME", "../CONFIGURE_ME.py")
CONFIGURE_ME = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIGURE_ME)

# --- --- --- --- Generate command for one CSV record
def generate_cpp_cmd(EXEC_PATH, UCR_PATH, RESULT_DIR, record, output):
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

    cmd = str(EXEC_PATH)+f" -p 6 -ucr {UCR_PATH} {name} -out {RESULT_DIR}/{name}_pfdevadtw.json"
    print(cmd, file=output)


# --- --- --- --- Main
if __name__ == '__main__':
    EE_PATH = pathlib.Path("../eeOutputFold0.csv").absolute()
    timestamp = CONFIGURE_ME.get_timestemp()
    OUT_DIR = pathlib.Path("generated-"+timestamp).absolute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_RESULTS = OUT_DIR/"results"
    OUT_DIR_RESULTS.mkdir(parents=True, exist_ok=True)

    OUT_CMD = (OUT_DIR/"commands").absolute()
    OUT = open(OUT_CMD, "w")

    UCR_TS_PATH = str(CONFIGURE_ME.get_ucr_folder())
    EXEC_PATH = pathlib.Path("./cmake-build-release/pfdevADTW").absolute()

    # --- --- --- --- Commands
    with open(EE_PATH, newline='') as csvfile:
        records = csv.reader(csvfile)
        header = next(records)          # Skip header
        print(header)
        for r in records:
            generate_cpp_cmd(EXEC_PATH, UCR_TS_PATH, OUT_DIR_RESULTS, r, OUT)

    # --- --- --- --- CPU INFO
    # CPU_INFO file
    OUT_CPU = (OUT_DIR/"cpu.json").absolute()
    OUT_CPU = open(OUT_CPU, "w")
    print(cpuinfo.get_cpu_info_json(), file=OUT_CPU)
