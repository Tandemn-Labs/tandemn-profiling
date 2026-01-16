"""
Just a Flask Server for Deploying SAM2
"""
import io 
import json
import time
import logging
import torch
from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import base64
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

app = Flask(__name__)

# we have a global model object 
predictor = None

def load_model():
    """Load the SAM2 Model"""
    global predictor
    logger.info("Loading SAM2 Model...")
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    predictor = SAM2ImagePredictor(sam2)
    logger.info(f"Model loaded in {time.time() - start:.2f} seconds")
    logger.info("Warming up the model...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    predictor.set_image(dummy_image)
    _ = predictor.predict(
        point_coords=np.array([[100, 100]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    logger.info("Model warmed up")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    if predictor is None:
        return jsonify({"status": "Model is not ready yet", "error": "Model not loaded"}), 503
    return jsonify({"status": "Model is ready"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint for SAM2

    Input - 
    {
        "Image" : "base64 encpded image",
        "prompts": [
            {"point": [x,y]. "label": 1},
            {"point": [x,y]. "label": 1}
        ]
    }

    Output - 
    {
        "success": true,
        "results": results in base64 encoded format
        "num_prompts": number of prompts
        "encode_time": time taken to encode the image
        "predict_time": time taken to predict the masks
        "total_time": total time taken to process the request
    }
    """
    start_time = time.time()
    data = request.get_json()
    image_b64 = data.get("image")
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    
    # the image is processed, now get the prompts
    prompts = data.get("prompts", [])

    # encode the image
    encode_start = time.time()
    predictor.set_image(image_np)
    encode_time = time.time() - encode_start

    # run the predictions
    predict_start = time.time()
    results = []

    for i, prompt in enumerate(prompts):
        point = np.array([[prompt["point"][0], prompt["point"][1]]])
        label = np.array([prompt["label"]])

        masks, scores, logits = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True,
        )
        prompt_result = {
            "prompt_id": 1,
            "num_masks": len(masks),
            "scores": scores.tolist(),
        }
        mask_images = []
        for mask in masks:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            buffer = io.BytesIO()
            mask_img.save(buffer, format='PNG')
            mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mask_images.append(mask_b64)
        prompt_result["masks_png"] = mask_images
        results.append(prompt_result)
    predict_time = time.time() - predict_start
    total_time = time.time() - start_time

    return jsonify({
        "success": True,
        "results": results,
        "num_prompts": len(prompts),
        "encode_time": encode_time * 1000,
        "predict_time": predict_time * 1000,
        "total_time": total_time * 1000,
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Get service info"""
    return jsonify({
        "model": "SAM2.1 Tiny",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ready": predictor is not None
    }), 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    
    load_model()
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)