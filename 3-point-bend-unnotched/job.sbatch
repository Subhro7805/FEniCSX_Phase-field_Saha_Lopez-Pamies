#!/bin/bash
#SBATCH --time=0:30:00                  
#SBATCH --job-name=3ptunnotch                 # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --partition=debug
#SBATCH --ntasks-per-node=100            # Number of tasks per node
#SBATCH --mem=150G                       # Memory per node
#SBATCH --account=uic442


# Change to the directory from which the batch job was submitted
cd $SLURM_SUBMIT_DIR
mkdir results

# Assigned the numeric portion of the pbs jobid to a varaible
export JOBID=`echo $SLURM_JOB_ID | cut -d"." -f1`
module reset
module load singularitypro/4.1.2
module unload slurm
singularity exec -B $SLURM_SUBMIT_DIR:/expanse/lustre/projects/uic442/ssaha6/testing_guide/Github-Code/3-point-bending-unnotched $HOME/dolfinx/dolfinx-stable/ mpirun -launcher fork -n 100 python3 -u $SLURM_SUBMIT_DIR/main.py >& $SLURM_SUBMIT_DIR/results/results.${JOBID}.oe
