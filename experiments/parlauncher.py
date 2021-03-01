#!/bin/env python3
import shlex
import subprocess
import os
import argparse

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Manage command line
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
def get_cmd_args():
    parser = argparse.ArgumentParser(description="Launch N commands in parallel")
    parser.add_argument("N", help="max number of processes", type=int)
    parser.add_argument("commands", help="the file containing the commands to run")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    args = parser.parse_args()
    return (parser, args)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Manage the process
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
def run(c, processes):
    cmd = c.strip() #shlex.split(c)
    if cmd == "":
        pass
    pidobj = subprocess.Popen(cmd, shell=True)
    processes[pidobj.pid] = (c, pidobj) # Must keep the popen object alive to prevent collection behind our back

def wait(processes):
    (pid, exit_status) = os.wait()
    signum = exit_status & 0xFF
    exit_code = (exit_status << 8) & 0xFF
    (c, pidobj) = processes.pop(pid)

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Main
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
if __name__ == "__main__":
    (parser, args) = get_cmd_args()
    maxp = args.N
    commands = []
    with open(args.commands) as fc: commands = fc.readlines()
    ### Launchd all the commands up to a maxp, then wait
    processes = {}
    for c in commands:
        run(c, processes)
        if len(processes) >= maxp:
            wait(processes)
    ### Wait for last processes
    while len(processes)>0:
        wait(processes)
