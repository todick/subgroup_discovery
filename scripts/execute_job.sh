#!/bin/bash

cluster=${SLURM_JOB_PARTITION}

filePath="$1"
lineToRead=${SLURM_ARRAY_TASK_ID}
# execute all arguments (script with parameters)
commandLine=$(sed "${lineToRead}q;d" ${filePath})
bash -c "${commandLine}"