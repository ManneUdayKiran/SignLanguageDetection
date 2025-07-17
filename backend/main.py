from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model.predict import predict_sign as model_predict_sign

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://signlanguagedetection-g24p.onrender.com"],  # Or ["http://localhost:5173"]
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render will set PORT=8000
    uvicorn.run("main:app", host="0.0.0.0", port=port)