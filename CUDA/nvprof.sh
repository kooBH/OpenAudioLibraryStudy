#!/bin/sh

###################################
#########     OPTIONS     #########
###################################
# (default)summary mode
#
# --print-gpu-trace : with timeline
#
# --cpu-profiling on : literally
#
# --log-file <filename>
#
# --normalized-time-unit [s, ms ,us, ns, col(fixed unit for each column), auto]
#

#nvprof --print-gpu-trace --cpu-profiling on --normalized-time-unit us ./$1 
nvprof --cpu-profiling on --normalized-time-unit ms ./$1 
