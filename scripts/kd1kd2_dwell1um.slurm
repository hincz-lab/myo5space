#!/bin/bash
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem=191000
#SBATCH --time=13-00:00:00 # Time To Run
#SBATCH --output=%j.out
#SBATCH --mail-user=jbh92@case.edu
#SBATCH --mail-type=END
# Put commands for executing job below this line

module purge
source load_python_3_8_6
source start_venv
python run_hpc.py kd1kd2 dwell1um