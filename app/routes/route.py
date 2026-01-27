from fastapi import APIRouter, HTTPException, File, UploadFile
import pandas as pd
from utils.prediction import ModelService 
import io

predict_route = APIRouter(prefix='/predict', tags=['Predictions'])

service = ModelService()

@predict_route.post('/')
async def sample_prediction(file: UploadFile = File(...) ):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="The file isn't compatible")
        
        content = await file.read()
        data = pd.read_csv(io.StringIO(content.decode("utf-8")))
        
        if 'gender' in data.columns:
           data = data.drop(columns=['gender'], inplace=True)
       
        predicts = service.predict(data)
        
        return {
            "result": predicts.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))