#!/bin/bash

PYTHON_VIRTUAL_ENV_PATH=$(which python)
LD_PATHS=LD_LIBRARY_PATH:

for library in $(ls -R "${PYTHON_VIRTUAL_ENV_PATH%bin/*}lib/python3.8/site-packages/nvidia/" | grep lib:)
  do
  LD_PATHS=$LD_PATHS$library
  done

export LD_LIBRARY_PATH=$LD_PATHS

cd planner_learning

start_time=$(date +"%H:%M:%S")
screen -d -m -S "${start_time}" -L -Logfile "./train_${start_time}.log" env LD_LIBRARY_PATH=$LD_PATHS nice -n0 python "train.py" --settings_file="config/train_settings.yaml" & pids+=($!)
wait "${pids[@]}"
cd ../