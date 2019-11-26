#!/bin/bash

for i in {1..10}
do
  DSsize = i * 16
  echo DSsize
  #python3 train.py --dataset-size=16 --epochs=30 --milestone=20
done
