#!/bin/bash
#SBATCH --job-name=heh+_6-31g
#SBATCH --time=00-01:00
#SBATCH --partition=gpuA100x4
#SBATCH --mem=64G
#SBATCH --signal=USR2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --account=bdgj-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hbhat@ucmerced.edu
#SBATCH --output=heh+_6-31g.out
#SBATCH --error=heh+_6-31g.err

# ---------------------------------------------------

python compareQOC.py --mol heh+ --basis 6-31g --outpath ./output/ --nsteps 200 --dt 1e-1 --numruns 1 --postfix 10
python compareQOC.py --mol heh+ --basis 6-31g --outpath ./output/ --nsteps 200 --dt 1e-1 --numruns 1 --postfix 48
python compareQOC.py --mol heh+ --basis 6-31g --outpath ./output/ --nsteps 200 --dt 1e-1 --numruns 1 --postfix 93

# python hessQOC.py --mol heh+ --basis 6-31g --outpath ./output/ --postfix "" --nsteps 200 --dt 1e-1