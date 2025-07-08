import logging
from PIL import Image, ImageDraw
import torch # Required by diffusers

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to hold the pipeline
# This is to avoid loading the model multiple times if not necessary,
# though for a serverless/scaled environment, model loading per request or instance might occur.
_model_pipeline = None
_model_loaded_successfully = False

def load_model(model_id: str = "CompVis/stable-diffusion-v1-4", device: str = "cpu"):
    """
    Loads the Stable Diffusion ONNX pipeline.

    Args:
        model_id (str): The Hugging Face model ID for the ONNX model.
                        Example: "runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4"
                        Ensure the model has ONNX weights available.
        device (str): The device to run the model on ("cpu" or "cuda").

    Returns:
        OnnxStableDiffusionPipeline or None: The loaded pipeline, or None if loading fails.
    """
    global _model_pipeline, _model_loaded_successfully
    if _model_pipeline is not None:
        logger.info("Model already loaded.")
        return _model_pipeline

    try:
        # Import here to avoid import errors if diffusers is not installed during initial setup phases
        from diffusers import OnnxStableDiffusionPipeline

        logger.info(f"Loading ONNX Stable Diffusion model: {model_id} for {device}...")

        # For ONNX, especially on CPU, we use OnnxStableDiffusionPipeline.
        # It requires onnxruntime.
        # We explicitly use float32 for wider CPU compatibility.
        # The `provider` argument is crucial for ONNX Runtime.
        # Common providers: 'CPUExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider'

        # Note: Some models might be structured differently or require specific revisions or subfolders.
        # For "CompVis/stable-diffusion-v1-4", the ONNX weights might need to be converted or sourced
        # from a specific ONNX-converted repository if not directly available in the main one.
        # A common pattern is to use a model specifically converted to ONNX, e.g., from Hugging Face Hub.
        # Example: `pipeline = OnnxStableDiffusionPipeline.from_pretrained("username/stable-diffusion-v1-4-onnx", provider="CPUExecutionProvider")`

        # Let's try a known ONNX model. If "CompVis/stable-diffusion-v1-4" doesn't have direct ONNX weights,
        # we might need to point to a specific repo that does, or use one like `stabilityai/sd-onnx`
        # For now, we'll attempt with `CompVis/stable-diffusion-v1-4` and assume it can find ONNX files
        # or that the user has pre-downloaded/converted them to a local path.
        # A more robust solution would be to use a model explicitly marked as ONNX.
        # For example, if `stabilityai/stable-diffusion-2-1-base-onnx` was a thing:
        # pipeline = OnnxStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base-onnx", provider="CPUExecutionProvider")

        # Using a community ONNX version for Stable Diffusion 1.4
        # This one is known to have ONNX weights: "runwayml/stable-diffusion-v1-5" with revision "onnx"
        # Or "hf-internal-testing/tiny-stable-diffusion-pipe-onnx" for a very small test
        # Let's try a community-converted ONNX model for SD 1.4 or 1.5 if direct CompVis one is tricky.
        # For testing, a smaller model is better: "hf-internal-testing/tiny-stable-diffusion-onnx"
        # However, for better quality, a full model like "runwayml/stable-diffusion-v1-5" (if ONNX version exists)
        # or "CompVis/stable-diffusion-v1-4" (if ONNX weights are present or converted) is preferred.

        # Trying with a known ONNX model from Hugging Face.
        # "diffusers/stable-diffusion-v1-4-onnx" is a community model.
        # Using "runwayml/stable-diffusion-v1-5" with specific onnx revision
        # For this example, we will use "CompVis/stable-diffusion-v1-4" and assume it can find/use ONNX files.
        # If it fails, it highlights the need to ensure the model path *does* contain ONNX compatible files.
        # A common source for ONNX models is from users who have converted and uploaded them.
        # Let's use a model ID known to work with ONNX pipeline.
        # `model_id = "diffusers/stable-diffusion-onnx-v1-4"` - this is a made-up example, use a real one.
        # `model_id = "runwayml/stable-diffusion-v1-5"` with `revision="onnx"` and `provider="CPUExecutionProvider"`
        # The original request mentioned "CompVis/stable-diffusion-v1-4". We will stick to it.
        # The user might need to convert it to ONNX first if not directly available.
        # Diffusers will attempt to download the appropriate files.

        _model_pipeline = OnnxStableDiffusionPipeline.from_pretrained(
            model_id,
            revision="onnx", # Specify ONNX revision if available/needed
            provider="CPUExecutionProvider", # Explicitly use CPU
            # torch_dtype=torch.float32 # Usually not needed for ONNX as dtype is part of the ONNX model
                                      # but can be relevant for intermediate steps if any.
        )

        logger.info(f"Model {model_id} loaded successfully on {device}.")
        _model_loaded_successfully = True
        return _model_pipeline
    except ImportError:
        logger.error("Diffusers or ONNXRuntime not installed. Please install them with `pip install diffusers onnxruntime`.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model {model_id}: {e}")
        logger.error("This might be because the model does not have ONNX weights readily available under that ID,")
        logger.error("or due to a network issue, or missing local ONNX model files if a local path was intended.")
        logger.error("Consider using a model ID specifically converted to ONNX format, e.g., from the Hugging Face Hub.")
    _model_loaded_successfully = False
    _model_pipeline = None
    return None

def generate_real_image(pipe, prompt: str, output_path: str = "output.png", num_inference_steps: int = 20) -> bool:
    """
    Generates an image using the loaded Stable Diffusion ONNX pipeline.

    Args:
        pipe: The loaded ONNX Stable Diffusion pipeline.
        prompt (str): The text prompt for image generation.
        output_path (str): The path to save the generated image.
        num_inference_steps (int): Number of diffusion steps. Lower for faster, potentially lower quality.

    Returns:
        bool: True if image generation and saving were successful, False otherwise.
    """
    if pipe is None:
        logger.error("Image generation failed: Model pipeline is not loaded.")
        return False

    try:
        logger.info(f"Generating image for prompt: '{prompt}' with {num_inference_steps} steps...")
        # For ONNX pipeline, ensure arguments match its expected signature.
        # `torch.float32` is generally for PyTorch models; ONNX models have dtypes baked in.
        # The pipeline handles device placement based on how it was loaded (e.g., CPUExecutionProvider).
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error during image generation or saving: {e}")
        return False

def generate_fallback_image(prompt: str, output_path: str = "output.png", message: str = "Fallback Image") -> None:
    """
    Generates a fallback image (a black square with text) and saves it.
    Used when the main model fails or is unavailable.

    Args:
        prompt (str): The text prompt (used for context in the message).
        output_path (str): The path to save the generated image.
        message (str): A message to display on the fallback image.
    """
    img_size = (512, 512) # Standard SD size
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
            y_text += 20 # Spacing for next line
    except Exception as e:
        logger.warning(f"Could not draw text on fallback image: {e}")
        # Fallback image will still be created, just without text.

    img.save(output_path)
    logger.info(f"Fallback image saved to {output_path} (original prompt: '{prompt}')")

def get_model_status():
    """Returns whether the model was loaded successfully."""
    return _model_loaded_successfully

# Example usage (for testing the generator directly)
if __name__ == '__main__':
    logger.info("Testing generator module...")

    # Attempt to load the model
    # For local testing, you might need to be logged into Hugging Face CLI
    # or have the model already cached.
    # `huggingface-cli login`
    # Using a smaller, known ONNX model for quicker testing if available,
    # e.g., "hf-internal-testing/tiny-stable-diffusion-onnx"
    # For now, sticking to the requested one, which might be large to download.
    # test_model_id = "hf-internal-testing/tiny-stable-diffusion-onnx" # A very small model for testing
    test_model_id = "CompVis/stable-diffusion-v1-4" # As per original request

    # It's better to test with a model explicitly designed for ONNX like:
    # test_model_id = "runwayml/stable-diffusion-v1-5"
    # and then specify revision="onnx" in load_model.
    # Or an even smaller one for quick tests: "hf-internal-testing/tiny-stable-diffusion-onnx"
    # For this run, we'll try "runwayml/stable-diffusion-v1-5" with ONNX revision
    # as "CompVis/stable-diffusion-v1-4" might not have a direct ONNX revision.

    # Using a specific ONNX model from runwayml
    # This model is known to have an ONNX variant.
    pipeline = load_model(model_id="runwayml/stable-diffusion-v1-5")

    if get_model_status() and pipeline:
        logger.info("Model loaded. Attempting to generate a real image...")
        success = generate_real_image(pipeline, "A photo of an astronaut riding a horse on the moon", "test_real_output.png", num_inference_steps=10) # Low steps for faster test
        if success:
            logger.info("Test real image generated successfully.")
        else:
            logger.error("Failed to generate real image. Generating fallback.")
            generate_fallback_image("Test prompt for fallback", "test_fallback_output.png", "Fallback: Real model failed")
    else:
        logger.warning("Model could not be loaded. Generating fallback image directly.")
        generate_fallback_image("Test prompt: Model load failed", "test_fallback_output_model_load_fail.png", "Fallback: Model Load Failed")

    logger.info("Generator module test finished.")
