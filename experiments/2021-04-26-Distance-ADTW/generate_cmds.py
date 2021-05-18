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

    def do_cmd(transform, kind):
        return str(EXEC_PATH)+f" {UCR_PATH} {name} {transform} {kind} 6 {RESULT_DIR}/{name}_adtw-{transform}-{kind}.json"

    print(do_cmd("original", "fixed"), file=output)
    print(do_cmd("original", "weighted"), file=output)
    print(do_cmd("derivative", "fixed"), file=output)
    print(do_cmd("derivative", "weighted"), file=output)


# --- --- --- --- Main
if __name__ == '__main__':
    EE_PATH = pathlib.Path("../eeOutputFold0.csv").absolute()
    timestamp = CONFIGURE_ME.get_timestemp()
    OUT_DIR = pathlib.Path("generated-"+timestamp).absolute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_CMD = (OUT_DIR/"commands").absolute()
    OUT = open(OUT_CMD, "w")

    UCR_TS_PATH = str(CONFIGURE_ME.get_ucr_folder())
    EXEC_PATH = pathlib.Path("./loocv_adtw/cmake-build-release/loocv_adtw").absolute()

    # --- --- --- --- Commands
    with open(EE_PATH, newline='') as csvfile:
        records = csv.reader(csvfile)
        header = next(records)          # Skip header
        print(header)
        for r in records:
            generate_cpp_cmd(EXEC_PATH, UCR_TS_PATH, OUT_DIR, r, OUT)

    # --- --- --- --- CPU INFO
    # CPU_INFO file
    OUT_CPU = (OUT_DIR/"cpu.json").absolute()
    OUT_CPU = open(OUT_CPU, "w")
    print(cpuinfo.get_cpu_info_json(), file=OUT_CPU)

