# AetherGenAI

**Mission:** To provide a lightweight and open-source API for image generation.

AetherGenAI is a FastAPI-based project designed to serve image generation models efficiently. This initial version provides the basic project structure and a placeholder API endpoint.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AetherGenAI
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.7+ installed. It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

## API Usage (Placeholder)

The `/generate` endpoint currently returns a dummy image.

**Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
-H "Content-Type: application/json" \
-d '{"prompt": "A beautiful sunset over a mountain range"}'
```

**Expected Response:**
The server will save an image named `output.png` in the project's root directory and return a JSON response:
```json
{
  "image_path": "output.png"
}
```

You can then open `output.png` to view the generated (dummy) image.

## Future Development
-   [ ] Integrate a lightweight image generation model (e.g., Stable Diffusion ONNX).
-   [ ] Implement actual image generation logic in `app/generator.py`.
-   [ ] Add comprehensive error handling and logging.
-   [ ] Include unit and integration tests.
-   [ ] Deploy to a platform like Hugging Face Spaces.
