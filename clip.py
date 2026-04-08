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

    if hasattr(outputs, "pooler_output"):
        tensor = outputs.pooler_output
        if hasattr(model, "text_projection"):
            tensor = model.text_projection(tensor)
    else:
        tensor = outputs

    embedding = tensor.squeeze(0).detach().cpu().tolist()
    return {"embedding": embedding, "length": len(embedding)}

@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)

    if isinstance(outputs, torch.Tensor):
        tensor = outputs
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        tensor = outputs.image_embeds
    elif hasattr(outputs, "pooler_output"):
        tensor = outputs.pooler_output
        # Only project if this is pre-projection size (usually 768)
        if tensor.shape[-1] == model.visual_projection.in_features:
            tensor = model.visual_projection(tensor)
    else:
        raise ValueError("Unsupported image output type")

    embedding = tensor.squeeze(0).detach().cpu().tolist()
    return {"embedding": embedding, "length": len(embedding)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
