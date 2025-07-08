import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import generator functions and model status variables/functions
from .generator import load_model, generate_real_image, generate_fallback_image, is_model_loaded, pipe as global_onnx_pipe # Import global pipe
# Naming the logger from generator to avoid conflicts if any
from .generator import model_loaded as generator_model_loaded # Import global model_loaded status

# --- Logging Configuration ---
# Configure basic logging for the main application.
# This should be one of the first things to run.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger for main.py

# --- Application Setup ---
app = FastAPI(
    title="AetherGenAI",
    description="A lightweight API for image generation using Stable Diffusion ONNX.",
    version="0.2.1" # Incremented version for this fix
)

# --- Model Loading ---
# Attempt to load the model when the application starts.
# load_model() will set global variables in app.generator (pipe, model_loaded)
logger.info("Application startup: Initiating ONNX model loading process...")
load_model() # This function now sets globals in generator.py

if generator_model_loaded: # Check status from generator.py
    logger.info("Application startup: ONNX Model loading attempted. Status: LOADED. Ready for real image generation.")
else:
    logger.warning(
        "Application startup: ONNX Model loading attempted. Status: FAILED. API will use fallback images."
        " Check logs from 'app.generator' for detailed error messages from the model loading process."
    )

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500, description="Text prompt for image generation.")

class GenerateResponse(BaseModel):
    image_path: str
    prompt_received: str
    status_message: str
    model_used: str # "real" or "fallback"

# --- API Endpoints ---
@app.get("/", summary="Root Endpoint", description="Returns a welcome message and API status.")
async def read_root():
    model_status_message = "Real image generation model loaded successfully." if generator_model_loaded else "Real image generation model FAILED to load. Using fallback images."
    return {
        "message": "Welcome to AetherGenAI!",
        "version": app.version,
        "model_status": model_status_message,
        "docs_url": "/docs"
    }

@app.post("/generate", response_model=GenerateResponse, summary="Generate Image from Prompt", description="Generates an image based on the provided text prompt. Uses a fallback if the main model is unavailable.")
async def generate_image_endpoint(request: GenerateRequest):
    output_filename = "output.png"
    prompt = request.prompt
    logger.info(f"Received image generation request for prompt: '{prompt}'")

    if not prompt or len(prompt.strip()) < 3: # Pydantic also handles this
        logger.warning(f"Invalid prompt received: '{prompt}'")
        raise HTTPException(status_code=400, detail="Prompt must be at least 3 characters long.")

    image_generated_successfully = False
    model_type_used = "fallback"

    # Use the global model_loaded and pipe from app.generator
    if generator_model_loaded and global_onnx_pipe is not None:
        logger.info(f"Attempting to generate image with real model for: '{prompt}'")
        # Pass the global pipe to generate_real_image, though generate_real_image also uses the global pipe.
        # For clarity, one could remove the 'pipe' argument from generate_real_image if it *only* uses the global.
        # However, current generate_real_image in generator.py is defined to take `pipe` as an argument,
        # but it also references the global `pipe` and `model_loaded`.
        # To strictly follow the user's snippet style where `generate_real_image` might not need `pipe` passed:
        # We will assume `generate_real_image` implicitly uses the global `pipe` set by `load_model`.
        # The `generate_real_image` signature was: `generate_real_image(prompt: str, ...)` after user's implied changes.
        # Let's ensure `generate_real_image` in `generator.py` uses its global `pipe`. Yes, it does.

        # My `generate_real_image` expects a prompt and output_path.
        # The global `pipe` is used within it.
        success = generate_real_image(prompt=prompt, output_path=output_filename)
        if success:
            image_generated_successfully = True
            model_type_used = "real"
            logger.info(f"Successfully generated real image for: '{prompt}'")
        else:
            logger.error(f"Real model generation failed for prompt: '{prompt}'. Will use fallback.")
    else:
        logger.warning(f"Real model not available/loaded. Proceeding with fallback for prompt: '{prompt}'")

    if not image_generated_successfully:
        status_message_detail = "Using fallback because real model is not loaded."
        if generator_model_loaded and global_onnx_pipe is not None: # Model was loaded but generation failed
            status_message_detail = "Using fallback because real model generation failed."

        generate_fallback_image(prompt=prompt, output_path=output_filename, message=f"Fallback: {status_message_detail}")
        logger.info(f"Generated fallback image for prompt: '{prompt}'")

    status_msg = f"Image generated using {model_type_used} model."
    if model_type_used == "fallback":
        if generator_model_loaded and global_onnx_pipe is not None: # Model loaded but real generation failed
            status_msg += " Real model generation attempt failed."
        else: # Model was not loaded at all
             status_msg += " Real model was not available/loaded."

    return GenerateResponse(
        image_path=output_filename,
        prompt_received=prompt,
        status_message=status_msg,
        model_used=model_type_used
    )

# --- Main Execution (for local debugging) ---
if __name__ == "__main__":
    logger.info("Starting AetherGenAI server for local debugging (uvicorn app.main:app)...")
    import uvicorn
    # Uvicorn will also configure its own logging.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
