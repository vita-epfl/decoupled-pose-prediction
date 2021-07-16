#!/bin/bash
#SBATCH --chdir ./
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G 
#SBATCH --time 24:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output slurm



echo STARTING AT `date`
srun python train.py 
#srun python cropp.py 
echo FINISHED at `date`
