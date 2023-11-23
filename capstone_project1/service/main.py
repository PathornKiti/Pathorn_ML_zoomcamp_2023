from fastapi import FastAPI, APIRouter
import uvicorn
from model import Model
from slot_filling import slot_filling
from intent_predict import intent_predict
import logging
logging.basicConfig(level = logging.INFO)

app = FastAPI(title="Intent Classification and Slots Filling", 
              description="API to predict the input text")
slot =slot_filling()
intent = intent_predict()
router = APIRouter()


@router.get("/")
async def home():
    return {"message": "Welcome"}

@router.post("/predict")
async def predict(data: dict):
    try:
        text = data["text"]
        intent_res=intent.get_intent(text)
        slot_res=slot.fill_slot(text)
        return {"Text":text,
                    "Intent":intent_res,
                    "Slot":slot_res
               }
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error("Something went wrong")


app.include_router(router)
if __name__ == '__main__':
    uvicorn.run("main:app",host="0.0.0.0", port=8000, reload=True)