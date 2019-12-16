#!/bin/bash
for SEGMENTATION in 1 25 100 300 1000 ; do
    for TYPE in Fish Flower Gravel Sugar; do
        for FILTER in 0 1 2 3 4; do
        qsub -v SEGMENTATIONARG=$SEGMENTATION,TYPEARG=$TYPE,FILTERARG=$FILTER quejob_example.sub
        done
    done
done