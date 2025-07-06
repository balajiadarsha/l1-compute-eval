#!/bin/bash -l
#PBS -N AFFINITY
#PBS -l select=4:ncpus=256
#PBS -l walltime=2:00:00
#PBS -q debug-scaling
#PBS -A ReForMer  # Replace with your project

# Change the directory to work directory, which is the directory you submit the job.
cd $PBS_O_WORKDIR
