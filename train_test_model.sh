#!/bin/bash

for i in {1..51..5}
do
  DSsize=$(($i*16))
  echo "training with DSsize = $DSsize"
  python3 train.py --dataset_size=$DSsize --epochs=30 --milestone=20
done

python3 test.py
