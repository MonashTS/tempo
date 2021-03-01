import csv
import pathlib
import cpuinfo

# --- --- --- --- Load configuration
import importlib.util
spec = importlib.util.spec_from_file_location("CONFGURE_ME", "../CONFIGURE_ME.py")
CONFIGURE_ME = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIGURE_ME)

# --- --- --- --- Generate command for one CSV record
def generate_cmd(HEAD_CMD, RESULT_DIR, record, lb, output):
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

    lb_name = lb.replace(" ", "_")

    # CDTW
    cmd = HEAD_CMD+f" {name} -dist cdtw {cdtw_w} " + lb + f" -out {RESULT_DIR}/{name}_cdtw_{lb_name}.json"
    print(cmd, file=output)

    # DTW
    cmd = HEAD_CMD+f" {name} -dist dtw " + lb + f" -out {RESULT_DIR}/{name}_dtw_{lb_name}.json"
    print(cmd, file=output)

# --- --- --- --- Main
if __name__ == '__main__':
    # Get config info
    UCR_PATH = CONFIGURE_ME.get_ucr_folder()
    EXEC_PATH = pathlib.Path("../../app/nn1dist/cmake-build-release/nn1dist").absolute()
    EE_PATH = pathlib.Path("../eeOutputFold0.csv").absolute()

    # Create output directory
    ts = CONFIGURE_ME.get_timestemp()
    OUT_DIR = pathlib.Path("generated-"+ts).absolute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Generate commands
    OUT_CMD = (OUT_DIR/"commands").absolute()
    OUT = open(OUT_CMD, "w")
    # Results of the run
    RESULT_DIR = (OUT_DIR/"results").absolute()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Head (common part) of commands
    HEAD = f"{EXEC_PATH} -ucr {UCR_PATH} "

    # --- --- --- --- Commands
    with open(EE_PATH, newline='') as csvfile:
        records = csv.reader(csvfile)
        header = next(records)          # Skip header
        print(header)
        for r in records:
            for lb in ["lb-none", "lb-keogh", "lb-keogh2", "lb-keogh2j", "lb-enhanced 5", "lb-enhanced2j 5", "lb-webb"]:
                generate_cmd(HEAD, RESULT_DIR, r, lb, OUT)

    # --- --- --- --- CPU INFO
    # CPU_INFO file
    OUT_CPU = (OUT_DIR/"cpu.json").absolute()
    OUT_CPU = open(OUT_CPU, "w")
    print(cpuinfo.get_cpu_info_json(), file=OUT_CPU)

