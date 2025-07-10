#!/bin/bash

#SBATCH --job-name="surf2025"                   # Job name
#SBATCH -p gpu-ampere                           # GPU Parition requested
#SBATCH --output=GPU_run.out             # GPU output file
#SBATCH --error=GPU_run.err              # GPU error file



$LAMMPS_GPU/build/lmp_gpu_serial -sf gpu -pk gpu 1 -in load_nw.in -log none
