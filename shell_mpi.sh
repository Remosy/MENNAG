#!/bin/bash
#PBS -l walltime=24:10:00
#PBS -l select=8:ncpus=24:mpiprocs=24:mem=120gb

#PBS -A UQ-EAIT-ITEE

module purge
module load OpenMPI/2.0.2


cd MENNAG

#mpiexec python3 -m mpi4py.futures src/train.py -n 48 -t CartPole-v1 -c settings/cartpole_default.json -g 10 -o out/cartout --mpi

mpiexec python3 -m mpi4py.futures src/train.py -n 48 -t BipedalWalker-v3 -c settings/bipedal_default.json -g 2000 -o out/bipedout --mpi
