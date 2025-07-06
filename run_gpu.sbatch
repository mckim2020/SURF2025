#!/bin/bash

#SBATCH --job-name="surf2025"                   # Job name
#SBATCH -p gpu-ampere                           # GPU Parition requested
#SBATCH --output=./outs/GPU_run.out             # GPU output file
#SBATCH --error=./outs/GPU_run.err              # GPU error file



python 