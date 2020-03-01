#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=24:mem=120gb

#PBS -A UQ-EAIT-ITEE

module purge


cd MENNAG

#mpiexec python3 -m mpi4py.futures src/train.py -n 48 -t CartPole-v1 -c settings/cartpole_default.json -g 10 -o out/cartout --mpi

python3 src/train.py -n 24 -t BipedalWalker-v3 -c settings/bipedal_default.json -g 10 -o out/bipedout
