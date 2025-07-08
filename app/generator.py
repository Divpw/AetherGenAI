import logging
from PIL import Image, ImageDraw
import torch # Required by diffusers

# Import ONNX specific pipeline
try:
    from diffusers import OnnxStableDiffusionPipeline
except ImportError:
    logging.critical("Diffusers library not found. Please ensure it's installed.")
    # This allows the rest of the file to be parsed without runtime error if diffusers is missing,
    # though load_model will fail.
    OnnxStableDiffusionPipeline = None


# Configure basic logging
# Using the root logger for simplicity as per user's snippet, or could use a named logger.
# BasicConfig should ideally be called once at the application entry point (e.g. main.py)
# For now, let's ensure it's configured if this module is run standalone.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for the pipeline and its status
pipe = None
model_loaded = False # Renamed from _model_loaded_successfully to match user snippet
                     # and made it directly accessible.

def load_model():
    """
    Loads the Stable Diffusion ONNX pipeline using "stabilityai/stable-diffusion-onnx".
    Sets global `pipe` and `model_loaded` variables.
    """
    global pipe, model_loaded

    if model_loaded and pipe is not None:
        logging.info("ONNX model already loaded.")
        return

    if OnnxStableDiffusionPipeline is None:
        logging.error("Cannot load ONNX model: OnnxStableDiffusionPipeline could not be imported from diffusers.")
        model_loaded = False
        pipe = None
        return

    try:
        logging.info("Attempting to load ONNX model 'stabilityai/stable-diffusion-onnx'...")
        pipe = OnnxStableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-onnx",  # ✅ Use a public ONNX model as per user instruction
            provider="CPUExecutionProvider"       # ✅ CPU-only
            # revision="onnx" # Not typically needed if the main branch of the repo is already ONNX
            # torch_dtype=torch.float32 # Generally not needed for ONNX pipeline
        )
        model_loaded = True
        logging.info("ONNX model 'stabilityai/stable-diffusion-onnx' loaded successfully.")
    except Exception as e:
        logging.error(f"ONNX model 'stabilityai/stable-diffusion-onnx' failed to load: {e}")
        logging.error("This could be due to network issues, incorrect model ID, missing ONNX runtime dependencies, or issues with the ONNX model files themselves.")
        logging.error("Ensure 'onnxruntime' is installed and functional.")
        model_loaded = False
        pipe = None

def generate_real_image(prompt: str, output_path: str = "output.png", num_inference_steps: int = 20) -> bool:
    """
    Generates an image using the loaded global Stable Diffusion ONNX pipeline.

    Args:
        prompt (str): The text prompt for image generation.
        output_path (str): The path to save the generated image.
        num_inference_steps (int): Number of diffusion steps.

    Returns:
        bool: True if image generation and saving were successful, False otherwise.
    """
    global pipe, model_loaded
    if not model_loaded or pipe is None:
        logging.error("Image generation failed: Model pipeline is not loaded.")
        return False

    try:
        logging.info(f"Generating real image for prompt: '{prompt}' with {num_inference_steps} steps...")
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        image.save(output_path)
        logging.info(f"Real image saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error during real image generation or saving: {e}")
        return False

def generate_fallback_image(prompt: str, output_path: str = "output.png", message: str = "Fallback Image") -> None:
    """
    Generates a fallback image and saves it.
    """
    img_size = (512, 512)
    img = Image.new("RGB", img_size, color="black")
    draw = ImageDraw.Draw(img)

    text_lines = [
        message,
        f"Prompt: {prompt[:50]}..." if prompt else "No prompt provided."
    ]

    y_text = 10
    try:
        for line in text_lines:
            draw.text((10, y_text), line, fill="white")
            y_text += 20
    except Exception as e:
        logging.warning(f"Could not draw text on fallback image: {e}")

    img.save(output_path)
    logging.info(f"Fallback image saved to {output_path} (original prompt: '{prompt}')")

# Renamed from get_model_status to match the direct global variable usage pattern
# This function isn't strictly needed if app.main directly imports and uses `model_loaded`
def is_model_loaded():
    """Returns whether the model was loaded successfully."""
    global model_loaded
    return model_loaded

# Example usage for direct testing
if __name__ == '__main__':
    # Ensure logging is configured for standalone run
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Testing generator module directly...")
    load_model() # Attempt to load the model

    if is_model_loaded():
        logging.info("Model loaded. Attempting to generate a real image...")
        success = generate_real_image("A futuristic city skyline at dusk", "test_real_output.png", num_inference_steps=10)
        if success:
            logging.info("Test real image generated successfully: test_real_output.png")
        else:
            logging.error("Failed to generate real image. Generating fallback.")
            generate_fallback_image("Test prompt for fallback", "test_fallback_output.png", "Fallback: Real model failed")
    else:
        logging.warning("Model could not be loaded. Generating fallback image directly.")
        generate_fallback_image("Test prompt: Model load failed", "test_fallback_output_model_load_fail.png", "Fallback: Model Load Failed")

    logging.info("Generator module test finished.")
