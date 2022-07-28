uvicorn src.web.logservice:app --host 0.0.0.0 --timeout-keep-alive 60 --reload &

cd frontend
npm start &

wait

