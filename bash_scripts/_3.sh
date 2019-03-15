#!/bin/bash

#Script to run a program on Teton
#Submit the job to the scheduler using the command : sbatch name_of_your_sbatch_script.sh


#Assign Job Name
#SBATCH --job-name=kmm_3

#Assign Account Name
#SBATCH --account=lsrtwitter


#Set Max Wall Time
#days-hours:minutes:seconds
#SBATCH --time=50:00:00

#Specify Resources Needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4

#SBATCH --mem=128000mb
#SBATCH --partition=teton	

#Load required modules

#module load miniconda3
#module load swset/2018.05
#module load gcc/7.3.0
#module load python/3.6.3
#module load py-numpy

source activate test_project


srun  python /project/lsrtwitter/mcooley3/bias_vs_labelefficiency/MAIN_ALL.py --run 0 --learning_rate 0.3 > outfiles/0_3.txt
