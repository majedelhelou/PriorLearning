#!/bin/bash

echo "training with DSsize = 16"
if [ "$1" == "create_dataset" ]; then
  python3 train.py --preprocess=True --dataset_size=16 --epochs=40 --milestone=20
else
  python3 train.py --dataset_size=16 --epochs=40 --milestone=20
fi

for i in {2..400}
do
  DSsize=$(($i*16))
  echo "training with DSsize = $DSsize"
  python3 train.py --dataset_size=$DSsize --epochs=40 --milestone=20
done

python3 test.py
