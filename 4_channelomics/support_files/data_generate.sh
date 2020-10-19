#!/bin/bash

N_SAMPLES=50000
VERSION=2

EXEC='data_generate.py'
PYTHON='/usr/bin/python'
BASEPATH='$HOME/ind/4_channelomics/'

export MKL_NUM_THREADS=1;
export NUMEXPR_NUM_THREADS=1;
export OMP_NUM_THREADS=1;


for rep in `seq 1 20`;
do
    NAME="V${VERSION}/${RANDOM}"

    $PYTHON ${BASEPATH}${EXEC}                          \
        --name $NAME                                    \
        --n_samples $N_SAMPLES &
done
