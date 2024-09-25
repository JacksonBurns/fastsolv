#!/bin/bash -l
#SBATCH -J features
#SBATCH -t 00-01:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH -N 1
#SBATCH -p xeon-p8
#SBATCH --exclusive

module purge
module load anaconda
source activate fprop

mkdir -p /state/partition1/user/jburns

cd /home/gridsan/jburns/fastsolv/data
export OMP_NUM_THREADS=1
python vermeire.py
python krasnov.py
python boobier.py
