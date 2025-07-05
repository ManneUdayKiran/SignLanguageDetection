from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model.predict import predict_sign as model_predict_sign

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def read_root():
    return {'Hello': 'World'}

@app.post('/predict')
async def predict_sign(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = model_predict_sign(contents)
    return JSONResponse(content={"predicted_sign": prediction})

