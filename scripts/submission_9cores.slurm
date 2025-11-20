#!/bin/bash
#SBATCH --job-name=SpMM-test_9cores
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1850M
#SBATCH -A f202409396cpcaa2x
#SBATCH --partition=normal-x86


echo "==============================="
echo "Starting SpMM test with 9 cores"
echo "Total Tasks: $SLURM_NTASKS"
echo "Cores per Task: $SLURM_CPUS_PER_TASK"
echo "==============================="

julia --project=../. ../src/distributed_run.jl

