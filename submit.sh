#!/bin/bash
#SBATCH --chdir ./
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 10G 
#SBATCH --time 03:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output slurmout
##SBATCH --output slurm_lstm_bodyposes_test

#source ../venv/bin/activate
 
echo STARTING AT `date`

CUDA_VISIBLE_DEVICES=0 
srun python train.py
#srun python3 test.py


echo FINISHED at `date`
