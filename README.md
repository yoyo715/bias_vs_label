# Bias vs. Label Efficiency in Social Media Data


	- Python 3.5 


1. Pre-process text using script

	- python3 preprocess.py

2. Run main program

	- python3 main.py




Neural Kernel Network https://arxiv.org/pdf/1806.04326.pdf




# packages


1. to run optimal_beta.py and optimal_beta3.py need cvxopt package

        - to install 'conda --name envname cvxopt

        - if error 'need libgfortran'

                - run 'conda install -- envname libgfortran==1'



About each file:

- main.py: file to run original model with multiple trials. Writes each output to separate file.

- main_test.py: file for testing the original model. Graphs the losses for one trial.

- gradient_check.py: script to double check the update rules in the model. Numerically calculates the derivative and compares to calculated derivative.

- graph.py: graphing file. Takes the output from the files in the output and Kmmout folders, and creates graphs (loss, class error, etc.).

- dictionary3.py: reads in all of the data files and creates the instances. This file also creates the optimal beta which can then be accessed by the main program.

- KMM_test.py: test function. Graphs the loss and class error for one trial.

- KMM3.py: Updated model with KMM implemented. Runs multiple trials and writes output of each trial and epoch into separate folders in the KMMoutput directory.

 

# Running On Teton

	1. To install python packages

		- create a miniconda env "conda create -n test_project python=3.5.6"

		- "source activate test_project"
		
		- "pip install packages"

	2. Must make python script executable!!
		- 'chmod +x script.py'

	3. Run SLURM script
	
		- sbatch run.sh 	

	
## Helpful commands

	- $ squeue  - shows all jobs running
 
	- $ squeue -u username  - shows your jobs

	- $ module spider   - shows available modules

	- $ module list  - show all currently loaded modules



## Slurm Script example

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

	#module load miniconda3
	source activate test_project

	srun python test.py

	source deactivate test_project








