uvicorn src.web.logservice:app --host 0.0.0.0 --reload &

cd frontend
npm start &

wait

