#!/bin/bash
#SBATCH --job-name=h2_sto-3g
#SBATCH --time=00-12:00
#SBATCH --partition=gpuA100x4
#SBATCH --mem=64G
#SBATCH --signal=USR2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --account=bdgj-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbhat@ucmerced.edu
#SBATCH --output=h2_sto-3g.out
#SBATCH --error=h2_sto-3g.err
# ---------------------------------------------------

python compareQOC.py --mol h2 --basis sto-3g --outpath ./output/ --nsteps 100 --dt 1e-1 --numruns 100

