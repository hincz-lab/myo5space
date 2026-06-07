#!/usr/bin/env python3
"""
File: run_hpc.py
Author: Joshua Holmes
Email: jbh92@case.edu

Script to evaluate how well various motors describe the experimental results for
myosin-XI reported in Tominaga2003. This script is meant to be run on an HPC
cluster.
"""

import numpy as np
import myo5space.hpceval as hpceval
import sys


workerdict = {
    "dwell 1um": hpceval.dwell1um_worker,
    "vrun 100um": hpceval.vrun100um_worker,
    "detach 1um": hpceval.detach1um_worker,
    "detach 100um": hpceval.detach100um_worker,
}

paramdict = {
    'kd1kd2': hpceval.make_kd1kd2_params,
    'kd1kd2nucthetac': hpceval.make_kd1kd2nutheta_params
}

def main(kind: str, _dataset: str):
    datsetdict = {
        "dwell1um": "dwell 1um",
        "vrun100um": "vrun 100um",
        "detach1um": "detach 1um",
        "detach100um": "detach 100um",
    }
    dataset = datsetdict.get(_dataset)
    params_func = paramdict.get(kind)
    params = params_func()
    print("Making Jobs")
    jobs = hpceval.make_jobs(params)
    print("Finished Making Jobs")
    outshape = tuple(p.size for p in params.values())
    worker = workerdict.get(dataset)
    print("Starting Work.")
    loglhs = hpceval.run_jobs(jobs, worker, outshape)
    print("Finishing Work.")
    filename = kind + '_' + _dataset + ".npy"
    np.save(filename, loglhs)


if __name__ == "__main__":
    args = sys.argv
    kind = sys.argv[1]
    dataset = sys.argv[2]
    main(kind, dataset)
