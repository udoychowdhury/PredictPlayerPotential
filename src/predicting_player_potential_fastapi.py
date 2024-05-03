from fastapi import FastAPI, Depends, HTTPException, status, Query
from starlette.requests import Request
from fastapi import FastAPI, APIRouter, Query
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv(r'/data/Male_FIFA_24_Players.csv')

# Load the model
model = joblib.load('/models/fifa_model.pkl')

# Function to use model and give response
def predict_high_potential(short_name):
    player_data = data[data['short_name'] == short_name]
    
    # Check if the player exists or the name is correct
    if player_data.empty:
        return "Player not found."

    # Select features for that player name
    features = player_data[[
        # 'age', 
        # 'overall',
        # 'potential',
        'value_eur', 
        'wage_eur', 
        'height_cm', 
        'weight_kg', 
        'preferred_foot', 
        'weak_foot',
        'skill_moves', 
        'work_rate', 
        'body_type', 
        'pace', 
        'shooting', 
        'passing', 
        'dribbling', 
        'defending', 
        'physic', 
        'skill_dribbling', 
        'skill_curve',
        'skill_fk_accuracy', 
        'skill_long_passing', 
        'skill_ball_control', 
        'mentality_aggression',
        'mentality_interceptions', 
        'mentality_positioning', 
        'mentality_vision', 
        'mentality_penalties',
        'mentality_composure'
    ]]
    
    # Preprocess features and make a prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    # Get player information
    overall = player_data['overall'].values[0]
    potential = player_data['potential'].values[0]
    improvement = potential - overall
    improvement_percentage = (improvement / overall) * 100

    # Extract confidence for both classes
    confidence_high_potential = prediction_proba[0, 1] * 100
    confidence_not_high_potential = prediction_proba[0, 0] * 100

    if prediction[0] == 1:
        return (f"{short_name}, currently rated at {overall}, has the potential to improve by approximately {improvement_percentage:.2f}% "
                f"to reach a potential rating of {potential}. This indicates a promising future and capability for further development. "
                f"Confidence in this prediction of high potential is {confidence_high_potential:.2f}%, suggesting a strong likelihood of achieving such growth.")
    else:
        return (f"{short_name}, with a current rating of {overall}, is predicted not to have high potential for significant improvement, "
                f"expected to remain close to their current performance level. "
                f"Confidence in this prediction is {confidence_not_high_potential:.2f}%, indicating a high certainty that substantial improvement is unlikely.")


# instantiate the FastAPI class
app = FastAPI(
    title="Predict Soccer Players Potential API", openapi_url="/openapi.json"
)

api_router = APIRouter()

# API Key Name to invoke in the headers of our requests call
API_KEY_NAME = "PPP-API-KEY"

# API Key
PPP_API_KEY = '-sH4m4gr7-LzhDMdtinrr-5575urIoXBZwESUQL-uDU'

@api_router.get("/")
async def root():
    return {"message": "Hello There!"}

def get_api_key(request: Request):
    api_key = request.headers.get(API_KEY_NAME)
    API_KEY = PPP_API_KEY
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key

@api_router.get("/predict", status_code=200)
async def fetch_search_results(request: Request,
                               api_key: str = Depends(get_api_key),
                               value: str = Query(..., min_length=1, max_length=100)) -> JSONResponse:
    # Call the function to get the prediction
    result = predict_high_potential(value)

    # Return the result
    return JSONResponse(content={"result": result})

# add endpoints to fastAPI app
app.include_router(api_router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
