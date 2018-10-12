#!/bin/bash

#Script to run a program on Teton
#Submit the job to the scheduler using the command : sbatch name_of_your_sbatch_script.sh


#Assign Job Name
#SBATCH --job-name=fasttext

#Assign Account Name
#SBATCH --account=lsrtwitter


#Set Max Wall Time
#days-hours:minutes:seconds
#SBATCH --time=00:03:00

#Specify Resources Needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1


#Load required modules

module load swset/2018.05
module load gcc/7.3.0
module load python/3.6.3
module load py-numpy

#srun bias_vs_labelefficiency/parallel_test.py
srun bias_vs_labelefficiency/main_test.py
