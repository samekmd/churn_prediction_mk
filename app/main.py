from fastapi import FastAPI
from routes.route import predict_route

app = FastAPI()

app.include_router(predict_route)