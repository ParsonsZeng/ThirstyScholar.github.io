#!/bin/bash


# Iteratively running the algo and save results
for i in {0..9}; 
  do python3 ddpg_pendulum_parser.py --file_index $i;
done
