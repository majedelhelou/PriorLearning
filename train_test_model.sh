#!/bin/bash

seed=1234

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
    *)
      shift
      ;;
  esac
done

echo "training with DSsize = 16"
if [ "$createdataset" == "true" ]; then
  python3 train.py --preprocess=True --dataset_size=16 --epochs=40 --milestone=20 --dataset_seed=$seed
else
  python3 train.py --dataset_size=16 --epochs=40 --milestone=20 --dataset_seed=$seed
fi

for i in {2..400}
do
  DSsize=$(($i*16))
  echo "training with DSsize = $DSsize"
  python3 train.py --dataset_size=$DSsize --epochs=40 --milestone=20 --dataset_seed=$seed
done

python3 test.py
