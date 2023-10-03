#!/bin/bash


# Start runner_k.py and runner_k2.py in the background
python r1.py > runner_k.log 2>&1 &
pid1=$!
python r2.py > runner_k2.log 2>&1 &
pid2=$!

# Wait for both processes to exit
wait $pid1
wait $pid2

# Now, execute runner_k3.py and runner_k4.py
python r3.py > runner_k3.log 2>&1 &
python r4.py > runner_k4.log 2>&1