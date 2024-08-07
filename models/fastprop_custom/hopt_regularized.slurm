#!/bin/bash -l
#SBATCH -J hopt_regularized
#SBATCH -t 04-04:00:00
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
ENABLE_REGULARIZATION=1 DISABLE_CUSTOM_LOSS=1 python hopt.py
mv hopt_results.csv hopt_results_regularized.csv
