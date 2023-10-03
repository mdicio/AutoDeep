#!/bin/bash
# Sleep for 6 hours (6 hours * 3600 seconds per hour)
sleep 21600

# Start runner_k.py and runner_k2.py in the background
python runner_k.py > runner_k.log 2>&1 &
pid1=$!
python runner_k2.py > runner_k2.log 2>&1 &
pid2=$!

# Wait for both processes to exit
wait $pid1
wait $pid2

# Now, execute runner_k3.py and runner_k4.py
python runner_k3.py > runner_k3.log 2>&1 &
python runner_k4.py > runner_k4.log 2>&1