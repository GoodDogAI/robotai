#!/bin/bash

# exit when any command fails
set -e

# Start the backend server
conda run -n robotai --live-stream uvicorn src.web.main:app --host 0.0.0.0 --timeout-keep-alive 60 --reload &

# Start tensorboard
tensorboard --logdir=./_sb3_logs --bind_all &

# Save the PID of tensorboard
# Send SIGTERM to tensorboard when this script exits
tpid=$!
trap "kill $tpid" EXIT

# Start the frontend server
cd frontend
npm update
npm start &

wait

