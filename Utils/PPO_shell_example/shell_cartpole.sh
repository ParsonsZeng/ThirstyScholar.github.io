#!/bin/bash

file_name=ppo_cartpole

for i in {0..9};
  do python3 ppo_cartpole_parser.py $file_name --file_index $i;
done

python3 create_result.py $file_name
