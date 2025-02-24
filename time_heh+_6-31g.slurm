#!/bin/bash
#SBATCH --job-name=time_heh+_6-31g
#SBATCH --time=00-23:59
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

for i in $(seq 2 11);
do
    nsteps=$((2**i))
    echo $nsteps
    python timeQOC.py --mol heh+ --basis 6-31g --outpath ./output/ --postfix "" --nsteps $nsteps --dt 1e-1 --numruns 1000
done

