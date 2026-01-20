"""
Just a Flask Server for Deploying SAM2
"""
import io
import time
import logging
import base64
import argparse

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from pycocotools import mask as mask_utils

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

app = Flask(__name__)

# we have a global model object 
predictor = None


def mask_to_rle(binary_mask):
    """
    Convert a binary mask to COCO RLE format.
    
    Args:
        binary_mask: numpy array of shape (H, W) with boolean or 0/1 values
        
    Returns:
        dict with 'size' [H, W] and 'counts' (RLE string)
    """
    # pycocotools requires Fortran-order array
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def load_model():
    """Load the SAM2 Model"""
    global predictor
    logger.info("Loading SAM2 Model...")
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device)
    if device == "cuda" and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile()...")
        sam2 = torch.compile(sam2, mode="reduce-overhead")
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
        "image" : "base64 encoded image",
        "prompts": [
            {"point": [x,y], "label": 1},
            {"point": [x,y], "label": 1}
        ],
        "multimask_output": true/false
    }

    Output - 
    {
        "success": true,
        "results": [
            {
                "prompt_id": 0,
                "num_masks": 3,
                "scores": [0.95, 0.87, 0.72],
                "masks": [
                    {
                        "size": [H, W],
                        "counts": "RLE encoded string"
                    },
                    ...
                ]
            },
            ...
        ],
        "image_size": [H, W],
        "num_prompts": number of prompts,
        "encode_time": time taken to encode the image (ms),
        "predict_time": time taken to predict the masks (ms),
        "total_time": total time taken to process the request (ms)
    }
    
    Note: Masks are returned in COCO RLE format. To decode on client side:
        from pycocotools import mask as mask_utils
        rle['counts'] = rle['counts'].encode('utf-8')
        binary_mask = mask_utils.decode(rle)
    """
    start_time = time.time()
    data = request.get_json()
    image_b64 = data.get("image")
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    
    # the image is processed, now get the prompts
    prompts = data.get("prompts", [])
    multimask_output = data.get("multimask_output", True)
    # encode the image
    encode_start = time.time()
    with torch.inference_mode():
        # predictor.set_image(image_np)
        with torch.autocast("cuda", dtype=torch.float16):
            predictor.set_image(image_np)
            encode_time = time.time() - encode_start
            predict_start = time.time()
            results = []

            # Batch all prompts for parallel GPU processing
            if len(prompts) > 0:
                # Stack all prompts into batched arrays
                batched_points = np.array([
                    [prompt["point"]]  # Shape: (num_prompts, 1, 2)
                    for prompt in prompts
                ])
                batched_labels = np.array([
                    [prompt["label"]]  # Shape: (num_prompts, 1)
                    for prompt in prompts
                ])
                
                masks, scores, logits = predictor.predict(
                    point_coords=batched_points,
                    point_labels=batched_labels,
                    multimask_output=multimask_output,
                )
                
                # Unpack results for each prompt
                for i in range(len(prompts)):
                    prompt_masks = masks[i]  # (num_masks, H, W)
                    prompt_scores = scores[i]  # (num_masks,)
                    
                    prompt_result = {
                        "prompt_id": i,
                        "num_masks": len(prompt_masks),
                        "scores": prompt_scores.tolist(),
                    }
                    mask_rles = []
                    for mask in prompt_masks:
                        rle = mask_to_rle(mask)
                        mask_rles.append(rle)
                    prompt_result["masks"] = mask_rles
                    results.append(prompt_result)
            
            predict_time = time.time() - predict_start
    total_time = time.time() - start_time

    return jsonify({
        "success": True,
        "results": results,
        "image_size": [image_np.shape[0], image_np.shape[1]],  # [H, W]
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

load_model()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
