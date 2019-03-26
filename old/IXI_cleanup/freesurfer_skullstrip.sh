#!/bin/bash
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=$PWD/out

time -o skullstrip_time.txt ls DATA/*.nii.gz | parallel --jobs 6 recon-all -s {.} -i {} -autorecon1 >> skulltrip_log.txt

