#!/usr/bin/env python3
"""
File: make_surmfiles.py
Author: Joshua Holmes
Email: jbh92@case.edu

This file is a script to create four slurm files used to run the run_hpc.py script.
"""

from pathlib import Path

def make_slurmfiles(prefix: str) -> None:
    datasets = ["dwell1um", "vrun100um", "detach1um", "detach100um"]
    for dataset in datasets:
        body = (
            "#!/bin/bash\n"
            + "#SBATCH -N 1\n"
            + "#SBATCH -c 40\n"
            + "#SBATCH --mem=191000\n"
            "#SBATCH --time=13-00:00:00 # Time To Run\n"
            + "#SBATCH --output=%j.out\n"
            + "#SBATCH --mail-user=jbh92@case.edu\n"
            + "#SBATCH --mail-type=END\n"
            + "# Put commands for executing job below this line\n\n"
            + "module purge\n"
            + "source load_python_3_8_6\n"
            + "source start_venv\n"
            + f"python run_hpc.py {prefix} {dataset}"
        )

        filename = prefix + "_" + dataset + ".slurm"
        path = Path.cwd().joinpath('scripts').joinpath(filename)
        with open(path, "w") as f:
            f.write(body)


def main():
    make_slurmfiles("kd1kd2")
    make_slurmfiles("kd1kd2nucthetac")


if __name__ == "__main__":
    main()
