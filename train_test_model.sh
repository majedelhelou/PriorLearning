#!/bin/bash

seed=1234
optimizer=Adam
lr=1e-3
batch_size=16
kernel=3
depth=10

while (( "$#" )); do
  case "$1" in
    -c|--create_dataset)
      createdataset=true
      shift
      ;;
    -s|--seed)
      if [ ! -z "$2" ]; then
        seed=$2
        shift
      fi
      shift
      ;;
    -o|--optimizer)
      if [ ! -z "$2" ]; then
        optimizer=$2
        shift
      fi
      shift
      ;;
    -lr|--learning_rate)
      if [ ! -z "$2" ]; then
        lr=$2
        shift
      fi
      shift
      ;;
    -b|--batch_size)
      if [ ! -z "$2" ]; then
        batch_size=$2
        shift
      fi
      shift
      ;;
    -k|--kernel)
      if [ ! -z "$2" ]; then
        kernel=$2
        shift
      fi
      shift
      ;;
    -d|--depth)
      if [ ! -z "$2" ]; then
        depth=$2
        shift
      fi
      shift
      ;;
    *)
      shift
      ;;
  esac
done

echo "training with DSsize = 16"
if [ "$createdataset" == "true" ]; then
  python3 train.py --preprocess=True --dataset_size=16 --dataset_seed=$seed --optimizer=$optimizer --lr=$lr --batch_size=$batch_size --num_of_layers=$depth --gsigma=$kernel
else
  python3 train.py --dataset_size=16 --dataset_seed=$seed --optimizer=$optimizer --lr=$lr --batch_size=$batch_size --num_of_layers=$depth --gsigma=$kernel
fi

for i in 100 200 300 400
do
  DSsize=$(($i*16))
  echo "training with DSsize = $DSsize"
  python3 train.py --dataset_size=$DSsize --dataset_seed=$seed --optimizer=$optimizer --lr=$lr --batch_size=$batch_size --num_of_layers=$depth --gsigma=$kernel
done
