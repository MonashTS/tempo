import csv
import pathlib
import cpuinfo

# --- --- --- --- Load configuration
import importlib.util
spec = importlib.util.spec_from_file_location("CONFIGURE_ME", "../CONFIGURE_ME.py")
CONFIGURE_ME = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CONFIGURE_ME)

# --- --- --- --- Generate command for one CSV record
def generate_java_cmd(EXEC_PATH, UCR_PATH, RESULT_DIR, record, output):
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

    dataset=f"{UCR_PATH}/{name}/{name}"

    cmd= str(EXEC_PATH)+f" -target_column=last -train={dataset}_TRAIN.arff -test={dataset}_TEST.arff -trees=100 -r=5 -export=1 -out={RESULT_DIR}"
    print(cmd, file=output)



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

    cmd= str(EXEC_PATH)+f" -ucr {UCR_PATH} {name} -t 100 -c 5 -out {RESULT_DIR}/{name}.json"
    print(cmd, file=output)




# --- --- --- --- Main
if __name__ == '__main__':
    EE_PATH = pathlib.Path("../eeOutputFold0.csv").absolute()
    timestamp = CONFIGURE_ME.get_timestemp()
    OUT_DIR = pathlib.Path("generated-"+timestamp).absolute()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ### ### All
    OUT_CMD = (OUT_DIR/"commands").absolute()
    OUT = open(OUT_CMD, "w")

    # ### ### CPP & TS
    UCR_TS_PATH = str(CONFIGURE_ME.get_ucr_folder())
    EXEC_CPP_PATH = pathlib.Path("../../app/pf2018/cmake-build-release/pf2018").absolute()
    RESULT_CPP_DIR = (OUT_DIR/"cpp_results").absolute()
    RESULT_CPP_DIR.mkdir(parents=True, exist_ok=True)

    # ### ### Java & ARFF
    UCR_ARFF_PATH = UCR_TS_PATH.replace("Univariate_ts", "Univariate_arff")
    EXEC_JAVA_PATH = pathlib.Path("PF2018_java/pf2018java.sh").absolute()
    RESULT_JAVA_DIR = (OUT_DIR/"java_results").absolute()
    RESULT_JAVA_DIR.mkdir(parents=True, exist_ok=True)

    # --- --- --- --- Commands
    with open(EE_PATH, newline='') as csvfile:
        records = csv.reader(csvfile)
        header = next(records)          # Skip header
        print(header)
        for r in records:
            generate_java_cmd(EXEC_JAVA_PATH, UCR_ARFF_PATH, RESULT_JAVA_DIR, r, OUT)
            generate_cpp_cmd(EXEC_CPP_PATH, UCR_TS_PATH, RESULT_CPP_DIR, r, OUT)

    # --- --- --- --- CPU INFO
    # CPU_INFO file
    OUT_CPU = (OUT_DIR/"cpu.json").absolute()
    OUT_CPU = open(OUT_CPU, "w")
    print(cpuinfo.get_cpu_info_json(), file=OUT_CPU)

