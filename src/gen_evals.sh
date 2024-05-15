#!/usr/bin/env bash

for rundir in "$1"/*
do
    echo "Evaluating run in $rundir..."
    python eval.py "$rundir"
done

