#!/usr/bin/env bash

cd {{ cwd }}
source ../../env/bin/activate
python experiment_driver.py -c "add new trees fixed seed" --force with {{ job_parameters }} &> {{ job_name }}.log

if [ $? -eq 0 ]; then rm {{ job_file }}; fi
