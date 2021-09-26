#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 10G 
#SBATCH --time 03:00:00
#SBATCH --account vita
#SBATCH --gres gpu:1
#SBATCH --output jta_long

module load gcc python py-torchvision py-torch
source ../../venv*/bin/activate
 
echo STARTING AT `date`

#python val.py
python test.py

echo FINISHED at `date`
