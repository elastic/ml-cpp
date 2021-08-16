# Running ML Experiments <!-- omit in toc -->

- [Submit a series of jobs to the task spooler manually](#submit-a-series-of-jobs-to-the-task-spooler-manually)
- [Submit ML experiments using job_submitter.py](#submit-ml-experiments-using-job_submitterpy)

To submit multiple jobs in a queue, we are using [task-spooler](http://vicerveza.homeunix.net/~viric/soft/ts/) utility.
In the Docker container, it is built and installed from the source. On your host machine, you can install it using
[homebrew](https://formulae.brew.sh/formula/task-spooler) or a
[Ubuntu package](https://howtoinstall.co/en/task-spooler).

Note that it can be called `ts` or `tsp` depending on your distribution. In the Docker container, both aliases are
valid. In the following, we will use `tsp`.

## Submit a series of jobs to the task spooler manually

1. Set the number of "slots" (threads) available simultaneously for the scheduler. To use all available CPUs, run:

   ```bash
   tsp -S `nproc`
   ```

2. Create a parametrizable bash script that activates the python environment and runs the python script. 
 
    Let's assume we want to run our experiment with the datasets `ccpp` and `electrical-grid-stability`. We create a bash script, that allows us to parametrize `dataset_name` and `threads`. 
- It creates a job file `<dataset_name>.job`, which activates the Python environment and calls the experiment driver with correct parameters. 
- It stores the output from the experiment driver in `<dataset_name>.log`.
- If the execution was successful, the job file gets deleted and the log file remains.

   ```bash
    cat > $1.job << END
    #!/bin/bash
    # submit.sh 
    cd $PWD
    source env/bin/activate
    python scripts/incremental_learning/experiment_driver.py with dataset_name="$1" threads=$2 &> $1.log
    if [ $? -eq 0 ]; then rm $1.job; fi
    END

    chmod 0775 $1.job
    tsp -N $2 $PWD/$1.job 
   ```

3. Submit the dataset jobs to run with 4 threads:

   ```bash
   for dataset in ccpp electrical-grid-stability; do submit.sh $dataset 4; done
   ```

4. Monitor the status of the execution queue with:

    ```bash
    watch tsp
    ```

*Note, if the task spooler has fewer slots (`tsp -S`) than required by the job (`tsp -N`), the job will hang in the queue forever!*

## Submit ML experiments using job_submitter.py

1. Make sure the task spooler is configured correctly for multithreaded execution:

   ```bash
   tsp -S `nproc`
   ```

2. Update the file `experiments.json` with the desired configurations.

3. Run the `job_submitter.py` script from this directory:

   ```bash
   ./job_submitter.py
   ```

    This will generate a number of uniquely named job scripts in this directory, e.g., `ccpp_42_1_resample_metric_features_0.5_0.1.job`. The script output is stored in the corresponding log file, e.g., `ccpp_42_1_resample_metric_features_0.5_0.1.log`.

4. Verify that jobs were submitted and run or finished:

    ```bash
    tsp

    ID   State      Output               E-Level  Times(r/u/s)   Command [run=0/12]
    1   finished   /tmp/ts-out.sT0GRu   0        136.56/3.14/2.63 /root/jupyter/scripts/incremental_learning/ccpp_42_4_resample_metric_features_0.5_0.5.job
    2   finished   /tmp/ts-out.zopCNu   0        156.50/3.32/2.69 /root/jupyter/scripts/incremental_learning/ccpp_42_4_resample_metric_features_0.5_0.1.job
    ```
