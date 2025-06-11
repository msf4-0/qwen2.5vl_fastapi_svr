from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoTokenizer, TextStreamer
from qwen_vl_utils import process_vision_info
from PIL import Image
from pathlib import Path
import tempfile
import torch
import io
import logging

# To run:
# uvicorn qwen_vl_api_prod:app --host 0.0.0.0 --port 8000

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Qwen2.5 API")

# -----------------------------
# Setup FastAPI
# -----------------------------
app = FastAPI()

# Allow cross-origin requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://your-node-red-ip:1880"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model Setup
# -----------------------------
model_dir = Path("Qwen2.5-VL-7B-Instruct")  # Change as needed
logger.info(f"Loading model from {model_dir}")

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28

processor = AutoProcessor.from_pretrained(model_dir / "INT4", min_pixels=min_pixels, max_pixels=max_pixels)

# Load tokenizer and set chat template
if processor.chat_template is None:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor.chat_template = tokenizer.chat_template
else:
    tokenizer = processor.tokenizer

from optimum.intel.openvino import OVModelForVisualCausalLM
model = OVModelForVisualCausalLM.from_pretrained(model_dir / "INT4", device="CPU")

logger.info("Model loaded successfully")

# -----------------------------
# Inference Endpoint
# -----------------------------
@app.post("/infer")
async def infer(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        logger.info("Received inference request")
        logger.info(f"Prompt: {prompt}")

        # Read and prepare image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Save temporarily to disk
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
            image_pil.save(temp_image_file.name)
            image_path = Path(temp_image_file.name)

        # Build message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Inference
        logger.info("Running generation...")
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        output_ids = model.generate(**inputs, max_new_tokens=100, streamer=streamer)

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        logger.info(f"Generated answer: {output_text}")
        return JSONResponse({"answer": output_text})

    except Exception as e:
        logger.exception("Error during inference")
        return JSONResponse(content={"error": str(e)}, status_code=500)
