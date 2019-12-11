#!/bin/bash

if [ "$1" == "1" ]; then
  ./train_test_model.sh -o SGD -lr 1e-3
  ./train_test_model.sh -o SGD -k 1
  ./train_test_model.sh -o SGD -k 2
  ./train_test_model.sh -o SGD -k 4
  ./train_test_model.sh -o SGD -k 5
fi

if [ "$1" == "2" ]; then
  ./train_test_model.sh -o Adam -lr 1e-3
  ./train_test_model.sh -o Adam -k 1
  ./train_test_model.sh -o Adam -k 2
  ./train_test_model.sh -o Adam -k 4
  ./train_test_model.sh -o Adam -k 5
fi
