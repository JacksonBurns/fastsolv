#!/bin/bash -l
#SBATCH -J hopt
#SBATCH -t 01-04:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH -N 1
#SBATCH -p xeon-g6-volta
#SBATCH --exclusive

module purge
module load anaconda
source activate fprop

mkdir -p /state/partition1/user/jburns

cd /home/gridsan/jburns/fastsolv/models/fastprop_custom
DISABLE_CUSTOM_LOSS=1 python hopt.py
mv hopt_results.csv hopt_results_fastprop.csv
DISABLE_CUSTOM_LOSS=0 python hopt.py
mv hopt_results.csv hopt_results_fastprop_sobolev.csv
