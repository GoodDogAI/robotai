#!/bin/bash

# exit when any command fails
set -e

if command -v conda &> /dev/null
then
    source activate robotai
fi

uvicorn src.web.main:app --host 0.0.0.0 --timeout-keep-alive 60 --reload &

cd frontend
npm start &

wait

