from fastapi import FastAPI, File, UploadFile, HTTPException
from deepface import DeepFace
import shutil
import uuid
import os
import logging
import time
from PIL import Image
import pillow_heif
import traceback

# Enable HEIC support
pillow_heif.register_heif_opener()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


def convert_to_jpg(input_path, output_path):
    try:
        image = Image.open(input_path)
        rgb_image = image.convert("RGB")
        rgb_image.save(output_path, "JPEG")
    except Exception as e:
        raise Exception(f"Image conversion failed: {str(e)}")


@app.post("/verify")
async def verify(id_image: UploadFile = File(...), selfie: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] New verification request received")

    # temp raw files
    raw_id = f"raw_id_{request_id}"
    raw_selfie = f"raw_selfie_{request_id}"

    # final jpg files
    id_filename = f"id_{request_id}.jpg"
    selfie_filename = f"selfie_{request_id}.jpg"

    try:
        # Save raw files (any format)
        with open(raw_id, "wb") as f:
            shutil.copyfileobj(id_image.file, f)

        with open(raw_selfie, "wb") as f:
            shutil.copyfileobj(selfie.file, f)

        logger.info(f"[{request_id}] Raw images saved")

        # Convert to JPG (handles HEIC, PNG, etc.)
        convert_to_jpg(raw_id, id_filename)
        convert_to_jpg(raw_selfie, selfie_filename)

        logger.info(f"[{request_id}] Images converted to JPG")

        # Run verification (safe config)
        result = DeepFace.verify(
            id_filename,
            selfie_filename,
            model_name="ArcFace",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False
        )

        confidence = 1 - result["distance"]

        logger.info(f"[{request_id}] Verification done | Verified: {result['verified']} | Confidence: {confidence:.2f}")

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "verified": bool(result["verified"]),
            "confidence": round(confidence, 2),
            "processing_time_ms": processing_time
        }

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Verification failed")

    finally:
        # Cleanup all files
        for file in [raw_id, raw_selfie, id_filename, selfie_filename]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"[{request_id}] Deleted file: {file}")