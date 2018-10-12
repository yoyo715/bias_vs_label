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

	1. Need to first load required modules (while in /home/usernam) though this can just be done in the slurm script.
	
	- $ module load gcc

  	- $ module load swset

  	- $ module load python/3.6.3

	2. Run SLURM script
	
	- sbatch run.sh 

	
## Helpful commands

	- $ squeue  - shows all jobs running
 
	- $ squeue -u username  - shows your jobs

	- $ module spider   - shows available modules

	- $ module list  - show all currently loaded modules

