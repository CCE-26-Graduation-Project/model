from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/embed-text")
async def embed_text(text: str):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    outputs = model.get_text_features(**inputs)
    embedding = outputs[0].detach().numpy().tolist()  
    return {"embedding": embedding}

@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    inputs = processor(images=image, return_tensors="pt")
    embedding = outputs[0].detach().numpy().tolist() 
    return {"embedding": embedding}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
