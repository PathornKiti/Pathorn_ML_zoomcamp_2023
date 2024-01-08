from fastapi import FastAPI, APIRouter
import uvicorn
import logging
from pydantic import BaseModel
from predict import run_classifier
logging.basicConfig(level = logging.INFO)



app = FastAPI(title="Brain MRI Alzheimer Classification", 
              description="API to predict Alzheimer Stages")
router = APIRouter()


@router.get("/")
async def home():
    return {"message": "Welcome"}

class Img(BaseModel):
    img_url: str

@router.post("/predict",status_code=200)
async def predict(request: Img):
        try:
            prediction = run_classifier(request.img_url)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error("Something went wrong")
        return prediction

app.include_router(router)
if __name__ == '__main__':
    uvicorn.run("main:app",
                host="0.0.0.0", 
                port=8000, 
                reload=True,
                log_level="debug",
                proxy_headers=True)

