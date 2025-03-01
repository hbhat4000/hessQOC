#!/bin/bash
#SBATCH --job-name=heh+_sto-3g
#SBATCH --time=00-06:00
#SBATCH --partition=gpuA100x4
#SBATCH --mem=64G
#SBATCH --signal=USR2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --account=bdgj-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbhat@ucmerced.edu
#SBATCH --output=heh+_sto-3g.out
#SBATCH --error=heh+_sto-3g.err
# ---------------------------------------------------

python compareQOC.py --mol heh+ --basis sto-3g --outpath ./output/compare_heh+_sto-3g/ --nsteps 200 --dt 1e-1 --numruns 100 --postfix 4

