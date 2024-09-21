#!/bin/bash -l
#SBATCH -J test
#SBATCH -t 00-00:10:00
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
CUDA_VISIBLE_DEVICES=0 python test.py
