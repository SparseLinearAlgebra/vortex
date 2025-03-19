import subprocess
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import re
import sys
from scipy import stats
import numpy as np

# architecture parameters
@dataclass
class arch:
    warps: int
    cores: int
    threads: int
@dataclass
# fpga parameters (CHANGE HERE)
class fpga_data:
    platform = "xilinx_u50_gen3x16_xdma_5_202210_1"
    dirpref = "test1"
    sim = "hw"
# running parameters
@dataclass
class run:
    arch: arch
    fpga_data: fpga_data
    perf: int
    kernel: str
    driver: str
    msize: int

path_to_vortex = Path.cwd().parent
tile_size = 'TS'
work_per_thread = 'WPT'
width = 'WIDTH'
tests_num = 'TESTS_NUM'

def error_running (run_params: run, error_text: str) -> str:
    return f"error running in {run_params.kernel} : warps={run_params.arch.warps} cores={run_params.arch.cores} threads={run_params.arch.threads}" \
         f" driver={run_params.driver} args=-n{run_params.msize} error message - {error_text}/n"

def error_verification (run_params: run, number_of_errors: str) -> str:
    return f"error in verifing results {run_params.kernel} : warps={run_params.arch.warps} cores={run_params.arch.cores} threads={run_params.arch.threads}" \
        f" driver={run_params.driver} args=-n{run_params.msize} Number of errors : {number_of_errors}'\n'"

def create_common_h (params: dict, kernel_name: str):
    file_name = f"{path_to_vortex}/tests/opencl/{kernel_name}/common.h"
    with open(file_name, 'w') as file:
        file.write("#ifndef COMMON_H\n" + "#define COMMON_H\n" + "\n")
        file.write(f"#define TESTS_NUM {params[tests_num]}\n" + "\n")
        if tile_size in params:
            file.write(f"#define TS {params[tile_size]}\n")
        if work_per_thread in params:
            file.write(f"#define WPT {params[work_per_thread]}\n")
            file.write("#define RTS (TS/WPT)\n")
        if width in params:
            file.write(f"#define WIDTH {params[width]}\n")
        file.write('\n' +  "#endif // COMMON_H")
    # open main.cc file to recompile before run with new common.h
    Path(f"{path_to_vortex}/tests/opencl/{kernel_name}/main.cc").touch(exist_ok=True)

def runtest(run_params: run, path_to_output_file: str) -> int:
    perf = f"--perf={run_params.perf}"
    run_args = f"-n{run_params.msize}"
    vortex = f"--warps={run_params.arch.warps} --cores={run_params.arch.cores} --threads={run_params.arch.threads}"
    if run_params.driver == "simx" or run_params.driver == "rtlsim":
        command = f"cd {path_to_vortex}/build && ./ci/blackbox.sh {vortex} {perf} --driver={run_params.driver} --app={run_params.kernel} --args={run_args}"
    elif run_params.driver == "xrt":
        fpga_data = run_params.fpga_data
        fpga_pref = f"FPGA_BIN_DIR={path_to_vortex}/hw/syn/xilinx/xrt/{fpga_data.dirpref}_{fpga_data.platform}_{fpga_data.sim}/bin TARGET={fpga_data.sim} PLATFORM={fpga_data.platform}"
        command = f"cd {path_to_vortex}/build && {fpga_pref} ./ci/blackbox.sh {perf} --driver={run_params.driver} --app={run_params.kernel} --args={run_args}"
    print(command)
    result = subprocess.run(f"{command} >> {path_to_output_file}", shell=True)
    return result.returncode

def collect(run_params: run, path_to_output_file: str) -> pd.DataFrame:
    with open(path_to_output_file, 'r') as file:
        lines = file.readlines()
    error_message = ""
    perf_dict = {}

    # matches all string currently starting with "PERF", such as
    # "core0: lmem reads=2134241" and "instrs=123, cycles=123, IPC=1.0"
    pattern =  r"(?:core\d+: )?([a-zA-Z0-9\.\-_+]+(?: [a-zA-Z0-9\.\-_+]+)*)=(\d+\.?\d*)"

    for line in lines:
        if line.startswith("PERF:"):
            parts = line.split(',')
            for part in parts:
                matches = re.findall(pattern, part)
                for key, value in matches:
                    perf_dict[key] = float(value)
        elif line.startswith("Elapsed time:"):
            matches = re.findall(r'(\S+)\s*[:]\s*(\d+)', line) # matches "time: 1234 ms"
            for key, value in matches:
                perf_dict[key] = float(value)
        # check for errors
        if "FAILED" in line:
            error_message = error_verification(run_params, line[line.find("FAILED! - "):])
        if "Error" in line:
            error_message = error_running(run_params, line[line.find("Error:"):])

    # parse string with perf statistic of running kernel
    if perf_dict["cycles"] <= 0:
        error_message = error_running(run_params, "Invalid number of cycles")
    # (ADD MORE IF NEEDED)
    #  write result to data frame
    run_result = pd.DataFrame([{"kernel": run_params.kernel[-1], "driver": run_params.driver, "cores": run_params.arch.cores,
                                "warps": run_params.arch.warps, "threads": run_params.arch.threads, "n": run_params.msize,
                                "instrs": perf_dict["instrs"], "cycles": perf_dict["cycles"],
                                "IPC": perf_dict["IPC"], "lmem reads": perf_dict["lmem reads"], "lmem writes": perf_dict["lmem writes"],
                                "local memory requests": perf_dict["lmem reads"] + perf_dict["lmem writes"], "global memory requests": perf_dict["memory requests"],
                                "time": perf_dict["time"],"error": error_message}])
    return run_result

def draw(data_frame: pd.DataFrame, x_label: str, y: str, y_label: str, title: str, path: str):
    data_frame.plot(kind = "bar", x = x_label, y = y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(path)

def check_time_stats(data_frame: pd.DataFrame) -> int:
    t = data_frame["time"].tolist()
    plt.hist(t)
    _, pval1 = stats.normaltest(t)
    _, pval2 = stats.shapiro(t)
    mean = np.mean(t)
    instr_error = 1
    error = stats.sem(t) + instr_error
    std = np.std(t, ddof=1)

    print(f"Mean is {mean}")
    print(f"Error is {error}")
    print(f"Standard deviation is {std}")
    if pval1 <= 0.05:
        print(f"P-value in D'Agostino and Pearson's test is {pval1}, which is less than 0.05")
        return -1
    if pval2 <= 0.05:
        print(f"P-value in Shapiro-Wilk test is {pval2}, which is less than 0.05")
        return -1
    if std / mean > 0.05 * mean:
        print("Standard deviation is greater than 5 percents of mean")
        return -1

    return 0


if len(sys.argv) > 1 and (sys.argv[1] == "xrt" or sys.argv[1] == "fpga"):
    drivers = ["xrt"]
else:
    drivers = ["simx", "rtlsim"]

TILESIZE = 4
WORKPERTHREAD = 4
WIDTH = 4
TESTSNUM = 100

# create common.h files for each kernel
params1 = {
    tests_num: TESTSNUM,
    tile_size: TILESIZE
}
create_common_h(params1, "kernel1")
create_common_h(params1, "kernel2")

params3 = {
    tests_num: TESTSNUM,
    tile_size: TILESIZE,
    work_per_thread: WORKPERTHREAD
}
create_common_h(params3, "kernel3")

params4 = {
    tests_num: TESTSNUM,
    tile_size: TILESIZE,
    width: WIDTH 
}
create_common_h(params4, "kernel4")

kernels = ["kernel1", "kernel2", "kernel3"]

experiment_dir = f"{path_to_vortex}/for_experiments"
output_dir = f"{experiment_dir}/outputs"
graphics_dir = f"{experiment_dir}/graphics"
stats1 = ["local memory requests", "global memory requests"]
stats2 = "IPC"
stats3 = "time"

mat_sizes = [32, 128] # square matrix sizes
THREADS = 16
WARPS = 2
CORES = 2
# 0 for nothing, 1 for cores info (stalls, fetches etc), 2 for memory info (lmem reads/writes etc)
PERFTYPE = 2
fpga_d = fpga_data()

for n, W in zip(mat_sizes, WARPS):
    for driver in drivers:
        driver_dfs = []
        for kernel in kernels:
            if kernel == "kernel3":
                T = int(THREADS / WORKPERTHREAD)
            elif kernel == "kernel4":
                T = int(THREADS / WIDTH)
            else:
                T = THREADS

            C = CORES
            arch_p = arch(threads=T, cores=C, warps=W)
            run_p = run(arch_p, kernel=kernel, driver=driver, msize=n, perf=PERFTYPE, fpga_data=fpga_d)

            # run kernel
            output_file = f"{output_dir}/output_{driver}_n{n}_{kernel}_TS{TILESIZE}_WPT{WORKPERTHREAD}_WID{WIDTH}_t{THREADS}w{W}_c{CORES}.txt"
            open(output_file, 'w').close()
            ret = runtest(run_p, output_file)
            if ret:
                sys.exit("Error occured when running latest command")
            # collect kernel statistics
            kernel_df = collect(run_p, output_file)
            driver_dfs.append(kernel_df)

        # put different kernel statistics into one data frame
        df = pd.concat(driver_dfs, ignore_index=True)
        if driver == "simx":
            sim_type = "Cycle-approximate simulation"
        elif driver == "rtlsim":
            sim_type = "RTL simulation"
        elif driver == "xrt":
            sim_type == "FPGA"
            ret = check_time_stats(df)
            if ret:
                print("Normality tests haven't passed")

        # draw graphs based on the recived statistic
        draw(df, "kernel", stats1, "Memory requests", f"Number of memory requests, {sim_type}",
        f"{graphics_dir}/mem_graph_{driver}_n{n}_TS{TILESIZE}_WPT{WORKPERTHREAD}_WID{WIDTH}_t{THREADS}_w{W}_c{CORES}.png")
        draw(df, "kernel", stats2, "",  f"Instructions per cycle, {sim_type}",
        f"{graphics_dir}/ipc_graph_{driver}_n{n}_TS{TILESIZE}_WPT{WORKPERTHREAD}_WID{WIDTH}_t{THREADS}_w{W}_c{CORES}.png")
        if driver == "xrt":
            draw(df, "kernel", stats3, "",  f"Elapsed time, {sim_type}",
            f"{graphics_dir}/time_graph_{driver}_n{n}_TS{TILESIZE}_WPT{WORKPERTHREAD}_WID{WIDTH}_t{THREADS}_w{W}_c{CORES}.png")
