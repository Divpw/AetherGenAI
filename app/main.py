import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import generator functions and model status checker
from .generator import load_model, generate_real_image, generate_fallback_image, get_model_status, logger as generator_logger

# Configure basic logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI(
    title="AetherGenAI",
    description="A lightweight API for image generation using Stable Diffusion ONNX.",
    version="0.2.0"
)

# --- Model Loading ---
# Attempt to load the model when the application starts.
# Using runwayml/stable-diffusion-v1-5 with ONNX revision as it's known to work.
# The original request mentioned CompVis/stable-diffusion-v1-4, but ONNX availability can be tricky.
# This can be parameterized or configured if needed.
MODEL_ID = "runwayml/stable-diffusion-v1-5"
logger.info(f"Attempting to load model '{MODEL_ID}' at application startup...")
model_pipeline = load_model(model_id=MODEL_ID)

if get_model_status():
    logger.info(f"Successfully loaded model '{MODEL_ID}' for image generation.")
else:
    logger.warning(
        f"Failed to load model '{MODEL_ID}'. The API will generate fallback images."
        " Check generator logs for more details (e.g., network issues, model compatibility, ONNX conversion)."
    )

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500, description="Text prompt for image generation.")
    # num_steps: int = Field(20, ge=5, le=50, description="Number of inference steps.") # Optional: add more controls

class GenerateResponse(BaseModel):
    image_path: str
    prompt_received: str
    status_message: str
    model_used: str # "real" or "fallback"

# --- API Endpoints ---
@app.get("/", summary="Root Endpoint", description="Returns a welcome message and API status.")
async def read_root():
    model_status_message = "Real image generation model loaded successfully." if get_model_status() else "Real image generation model FAILED to load. Using fallback images."
    return {
        "message": "Welcome to AetherGenAI!",
        "version": app.version,
        "model_status": model_status_message,
        "docs_url": "/docs"
    }

@app.post("/generate", response_model=GenerateResponse, summary="Generate Image from Prompt", description="Generates an image based on the provided text prompt. Uses a fallback if the main model is unavailable.")
async def generate_image_endpoint(request: GenerateRequest):
    """
    Endpoint to generate an image from a text prompt.

    - Validates the prompt.
    - Attempts to use the pre-loaded ONNX model to generate an image.
    - If the model isn't available or generation fails, a fallback dummy image is created.
    - Saves the generated image as `output.png` (can be made unique later).
    - Returns the path to the image and status information.
    """
    output_filename = "output.png" # Static filename for now
    prompt = request.prompt
    logger.info(f"Received image generation request for prompt: '{prompt}'")

    # Basic prompt validation (already handled by Pydantic Field constraints, but explicit check is fine)
    if not prompt or len(prompt.strip()) < 3:
        logger.warning(f"Invalid prompt received: '{prompt}'")
        raise HTTPException(status_code=400, detail="Prompt must be at least 3 characters long.")

    image_generated_successfully = False
    model_type_used = "fallback"

    if get_model_status() and model_pipeline:
        logger.info(f"Attempting to generate image with real model for: '{prompt}'")
        # Could pass request.num_steps here if added to GenerateRequest
        success = generate_real_image(pipe=model_pipeline, prompt=prompt, output_path=output_filename)
        if success:
            image_generated_successfully = True
            model_type_used = "real"
            logger.info(f"Successfully generated real image for: '{prompt}'")
        else:
            logger.error(f"Real model generation failed for prompt: '{prompt}'. Will use fallback.")
    else:
        logger.warning(f"Real model not available or not loaded. Proceeding with fallback for prompt: '{prompt}'")

    if not image_generated_successfully:
        status_message_detail = "Using fallback because real model is not loaded."
        if get_model_status() and model_pipeline : # Model was loaded but generation failed
            status_message_detail = "Using fallback because real model generation failed."

        generate_fallback_image(prompt=prompt, output_path=output_filename, message=f"Fallback: {status_message_detail}")
        logger.info(f"Generated fallback image for prompt: '{prompt}'")

    status_msg = f"Image generated using {model_type_used} model."
    if model_type_used == "fallback" and get_model_status():
        status_msg += " Real model generation attempt failed."
    elif model_type_used == "fallback":
         status_msg += " Real model was not available/loaded."


    return GenerateResponse(
        image_path=output_filename,
        prompt_received=prompt,
        status_message=status_msg,
        model_used=model_type_used
    )

# --- Main Execution (for local debugging) ---
if __name__ == "__main__":
    # This allows running directly with `python app/main.py`
    # For production, use: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
    logger.info("Starting AetherGenAI server for local debugging...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
