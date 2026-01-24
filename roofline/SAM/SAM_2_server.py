"""
Just a Flask Server for Deploying SAM2
"""
import io
import time
import logging
import base64
import argparse
from queue import Queue, Empty, Full
import threading
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from pycocotools import mask as mask_utils
from dataclasses import dataclass
from concurrent.futures import Future
from typing import Any,Dict,List,Optional


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

app = Flask(__name__)

# we have a global model object 
predictor = None

#####################GPUWorker Code#####################

@dataclass
class Job:
    """
    A job needs:
        input: image_np, prompts, multimask_output
        output container: something the HTTP handler can wait on

    In Python, the simplest "waitable" is concurrent.futures.Future.
    """
    image_np: np.ndarray
    prompts: List[Dict[str, Any]]
    multimask_output: bool
    future: Future 
    t0: float # enqueue timestamp
# make the queue that will hold the jobs
MAX_QUEUE_SIZE = 500
job_queue: Queue[Job] = Queue(maxsize=MAX_QUEUE_SIZE)
# make the event that will be used to stop the worker thread
stop_event = threading.Event()
worker_thread = None


def start_gpu_worker():
    global worker_thread
    worker_thread = threading.Thread(
        target=gpu_worker_loop,
        args=(predictor, job_queue, stop_event),
        daemon=True
    )
    worker_thread.start()
    logger.info("GPU worker thread started")

# Make the GPU Worker Function that will be a loop that runs per process
def gpu_worker_loop(predictor, job_queue: Queue, stop_event: threading.Event):
    torch.backends.cudnn.benchmark = True

    while not stop_event.is_set():
        try:
            job = job_queue.get(timeout=0.1)
        except Empty:
            continue

        try:
            result = run_one_job(predictor, job)
            job.future.set_result(result)
        except Exception as e:
            job.future.set_exception(e)
        finally:
            job_queue.task_done()

########################################################

#####################Python Endpoints###################


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
        "decode_time": time to decode base64+PIL+numpy (ms),
        "encode_time": time to encode image features on GPU (ms),
        "predict_time": time to predict masks on GPU (ms),
        "gpu_time": encode_time + predict_time (ms),
        "rle_time": time to convert masks to RLE format (ms),
        "queue_wait_ms": time spent waiting in queue (ms),
        "total_time": total end-to-end time (ms)
    }
    
    Note: Masks are returned in COCO RLE format. To decode on client side:
        from pycocotools import mask as mask_utils
        rle['counts'] = rle['counts'].encode('utf-8')
        binary_mask = mask_utils.decode(rle)
    """
    if predictor is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    start_time = time.time()
    data = request.get_json()

    # decode image
    t_decode_start = time.time()
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"success": False, "error": "Missing image"}), 400

    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    decode_time = (time.time() - t_decode_start) * 1000

    prompts = data.get("prompts", [])
    multimask_output = bool(data.get("multimask_output", True))

    fut = Future() # this will be set by the GPU worker when it's done
    t_enqueue = time.time()
    job = Job(
        image_np=image_np,
        prompts=prompts,
        multimask_output=multimask_output,
        future=fut,
        t0=t_enqueue
    )

    try:
        job_queue.put(job, block=False)
    except Full:
        return jsonify({"success": False, "error": "Overloaded"}), 503

    try:
        out = fut.result(timeout=120)  # higher timeout for long running jobs
        # Add decode_time to the output
        out["decode_time"] = decode_time
        out["queue_wait_ms"] = (out.get("t_gpu_start", t_enqueue) - t_enqueue) * 1000
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def run_one_job(predictor, job: Job) -> dict:
    t_gpu_start = time.time()
    image_np = job.image_np
    prompts = job.prompts
    multimask_output = job.multimask_output

    with torch.inference_mode():
        # --- Encode image (embedding) ---
        t_enc0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            predictor.set_image(image_np)
        encode_time = (time.time() - t_enc0) * 1000

        # --- Predict for prompts ---
        t_pred0 = time.time()
        results = []

        if len(prompts) > 0:
            # Make batched prompt arrays
            batched_points = np.array([ [p["point"]] for p in prompts ], dtype=np.float32)
            batched_labels = np.array([ [p["label"]] for p in prompts ], dtype=np.int32)

            with torch.autocast("cuda", dtype=torch.float16):
                masks, scores, logits = predictor.predict(
                    point_coords=batched_points,
                    point_labels=batched_labels,
                    multimask_output=multimask_output,
                )

            predict_time = (time.time() - t_pred0) * 1000
            gpu_time = encode_time + predict_time

            # --- Convert masks to RLE (CPU-heavy) ---
            t_rle_start = time.time()
            for i in range(len(prompts)):
                prompt_masks = masks[i]
                prompt_scores = scores[i]
                mask_rles = [mask_to_rle(m) for m in prompt_masks]
                results.append({
                    "prompt_id": i,
                    "num_masks": len(prompt_masks),
                    "scores": prompt_scores.tolist(),
                    "masks": mask_rles
                })
            rle_time = (time.time() - t_rle_start) * 1000
        else:
            predict_time = 0
            gpu_time = encode_time
            rle_time = 0

    total_queue_to_done = (time.time() - job.t0) * 1000
    queue_wait = (t_gpu_start - job.t0) * 1000

    logger.info(f"Job timing - Queue: {queue_wait:.1f}ms | Encode: {encode_time:.1f}ms | "
                f"Predict: {predict_time:.1f}ms | RLE: {rle_time:.1f}ms | Total: {total_queue_to_done:.1f}ms")

    return {
        "success": True,
        "results": results,
        "image_size": [int(image_np.shape[0]), int(image_np.shape[1])],
        "num_prompts": len(prompts),
        "encode_time": encode_time,
        "predict_time": predict_time,
        "gpu_time": gpu_time,
        "rle_time": rle_time,
        "total_time": total_queue_to_done,
        "t_gpu_start": t_gpu_start,
    }


@app.route('/info', methods=['GET'])
def info():
    """Get service info"""
    return jsonify({
        "model": "SAM2.1 Tiny",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ready": predictor is not None
    }), 200
########################################################

#####################Main Code#########################
load_model()
start_gpu_worker()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
########################################################