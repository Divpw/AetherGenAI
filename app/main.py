from fastapi import FastAPI
from pydantic import BaseModel
import os

from .generator import generate_dummy_image

# TODO: Load the actual image generation model here
# model = None

app = FastAPI(
    title="AetherGenAI",
    description="A lightweight API for image generation.",
    version="0.1.0"
)

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    image_path: str
    prompt_received: str

@app.get("/", summary="Root endpoint", description="Returns a welcome message.")
async def read_root():
    return {"message": "Welcome to AetherGenAI! Visit /docs for API documentation."}

@app.post("/generate", response_model=GenerateResponse, summary="Generate an image", description="Generates an image based on the provided prompt. Currently returns a dummy image.")
async def generate_image_endpoint(request: GenerateRequest):
    """
    Endpoint to generate an image from a text prompt.

    - Accepts a `prompt` string in the request body.
    - Generates a dummy image and saves it as `output.png`.
    - Returns the path to the generated image.
    """
    output_filename = "output.png"

    # TODO: Replace with actual model prediction
    # For now, we call the dummy image generator
    generate_dummy_image(prompt=request.prompt, output_path=output_filename)

    return GenerateResponse(
        image_path=output_filename,
        prompt_received=request.prompt
    )

if __name__ == "__main__":
    import uvicorn
    # This is for running the app directly with `python app/main.py`
    # For production, use `uvicorn app.main:app --host 0.0.0.0 --port 8000`
    uvicorn.run(app, host="127.0.0.1", port=8000)
