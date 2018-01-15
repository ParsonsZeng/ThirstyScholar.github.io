#!/bin/bash

# Iterative running the algorithm and save results
for i in {0..9}; 
  do python3 r_a2c.py --file_index $i;
done
