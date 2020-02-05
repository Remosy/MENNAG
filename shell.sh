#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --job-name=biped1000
python3 src/train.py -t "BipedalWalker-v2" -c settings/bipedal_default.json -g 100 -n 8 -o out
