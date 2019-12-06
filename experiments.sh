#!/bin/bash

mode = $1

if [ "$1" == "1" ]; then
  ./train_test_model.sh -o SGD -lr 1e-3
  ./train_test_model.sh -o SGD -lr 1e-2 -b 32
  ./train_test_model.sh -o SGD -lr 1e-4 -b 32
  ./train_test_model.sh -o SGD -lr 1e-2 -b 64
  ./train_test_model.sh -o SGD -lr 1e-4 -b 64
  ./train_test_model.sh -o SGD -lr 1e-2 -b 128
  ./train_test_model.sh -o SGD -lr 1e-4 -b 128
  ./train_test_model.sh -o SGD -k 1
  ./train_test_model.sh -o SGD -k 2
  ./train_test_model.sh -o SGD -k 4
  ./train_test_model.sh -o SGD -k 5
fi

if [ "$1" == "2" ]; then
  ./train_test_model.sh -o Adam -lr 1e-2 -b 32
  ./train_test_model.sh -o Adam -lr 1e-4 -b 32
  ./train_test_model.sh -o Adam -lr 1e-2 -b 64
  ./train_test_model.sh -o Adam -lr 1e-4 -b 64
  ./train_test_model.sh -o Adam -lr 1e-2 -b 128
  ./train_test_model.sh -o Adam -lr 1e-4 -b 128
  ./train_test_model.sh -o Adam -k 1
  ./train_test_model.sh -o Adam -k 2
  ./train_test_model.sh -o Adam -k 4
  ./train_test_model.sh -o Adam -k 5
fi

if [ "$1" == "3" ]; then
  ./train_test_model.sh -o SGD -d 5 -lr 1e-2
  ./train_test_model.sh -o SGD -d 10 -lr 1e-2
  ./train_test_model.sh -o SGD -d 15 -lr 1e-2
  ./train_test_model.sh -o SGD -d 5 -lr 1e-3
  ./train_test_model.sh -o SGD -d 10 -lr 1e-3
  ./train_test_model.sh -o SGD -d 15 -lr 1e-3
  ./train_test_model.sh -o SGD -d 5 -lr 1e-4
  ./train_test_model.sh -o SGD -d 10 -lr 1e-4
  ./train_test_model.sh -o SGD -d 15 -lr 1e-4
fi

if [ "$1" == "4" ]; then
  ./train_test_model.sh -o Adam -d 5 -lr 1e-2
  ./train_test_model.sh -o Adam -d 10 -lr 1e-2
  ./train_test_model.sh -o Adam -d 15 -lr 1e-2
  ./train_test_model.sh -o Adam -d 5 -lr 1e-3
  ./train_test_model.sh -o Adam -d 10 -lr 1e-3
  ./train_test_model.sh -o Adam -d 15 -lr 1e-3
  ./train_test_model.sh -o Adam -d 5 -lr 1e-4
  ./train_test_model.sh -o Adam -d 10 -lr 1e-4
  ./train_test_model.sh -o Adam -d 15 -lr 1e-4
fi
