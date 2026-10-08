import base64
import io
import os

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from torchvision import transforms

from model import ResNetUNet


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "fruit_resnet50_unet.pth")

MODEL_URL = (
    "https://huggingface.co/codefreak231/fruit_resnet50_unet/"
    "resolve/main/fruit_resnet50_unet.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


app = FastAPI(title="Fruit Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None


def download_model() -> None:
    """Download model weights from Hugging Face if they are not available locally."""
    if os.path.exists(MODEL_PATH):
        return

    print("Downloading model weights from Hugging Face...")

    response = requests.get(
        MODEL_URL,
        stream=True,
        timeout=600,
    )
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    print("Model weights downloaded successfully.")


def load_model() -> None:
    global model

    if model is not None:
        return

    # Download the model if it does not exist on the server.
    download_model()

    model_instance = ResNetUNet().to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_instance.load_state_dict(checkpoint)
    model_instance.eval()

    model = model_instance


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


async def read_rgb_image(
    file: UploadFile,
    field_name: str,
) -> Image.Image:

    if file is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required.",
        )

    if file.content_type and "image" not in file.content_type:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not a valid image file.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is empty.",
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)

        return image.convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: unable to read image data.",
        ) from exc


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def build_mask_and_overlay(
    original_rgb: Image.Image,
    predicted_mask: np.ndarray,
) -> tuple[Image.Image, Image.Image]:

    mask_image = Image.fromarray(
        (predicted_mask * 255).astype(np.uint8),
        mode="L",
    )

    resized_mask = (
        np.asarray(
            mask_image.resize(
                original_rgb.size,
                Image.Resampling.BILINEAR,
            )
        )
        / 255.0
    )

    mask_bool = resized_mask > 0.5

    overlay_rgba = np.array(
        original_rgb.convert("RGBA"),
        dtype=np.uint8,
    ).copy()

    overlay_rgba[mask_bool, 0] = 255
    overlay_rgba[mask_bool, 1] = 0
    overlay_rgba[mask_bool, 2] = 0
    overlay_rgba[mask_bool, 3] = 180

    overlay_image = Image.fromarray(
        overlay_rgba,
        mode="RGBA",
    )

    return mask_image, overlay_image


def compute_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> dict:

    pred = predicted_mask.astype(bool)
    gt = ground_truth_mask.astype(bool)

    true_positive = np.logical_and(
        pred,
        gt,
    ).sum()

    false_positive = np.logical_and(
        pred,
        ~gt,
    ).sum()

    false_negative = np.logical_and(
        ~pred,
        gt,
    ).sum()

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if iou_denominator > 0:
        iou = true_positive / iou_denominator
    else:
        iou = (
            1.0
            if not pred.any() and not gt.any()
            else 0.0
        )

    dice_denominator = pred.sum() + gt.sum()

    if dice_denominator > 0:
        dice = (2 * true_positive) / dice_denominator
    else:
        dice = (
            1.0
            if not pred.any() and not gt.any()
            else 0.0
        )

    precision_denominator = (
        true_positive + false_positive
    )

    if precision_denominator > 0:
        precision = (
            true_positive / precision_denominator
        )
    else:
        precision = (
            1.0
            if not pred.any()
            else 0.0
        )

    recall_denominator = (
        true_positive + false_negative
    )

    if recall_denominator > 0:
        recall = (
            true_positive / recall_denominator
        )
    else:
        recall = (
            1.0
            if not gt.any()
            else 0.0
        )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def to_binary_mask(
    image: Image.Image,
) -> np.ndarray:

    resized = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR,
    )

    grayscale = np.asarray(
        resized.convert("L")
    )

    return (
        grayscale > 127
    ).astype(np.uint8)


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    ground_truth_mask: UploadFile | None = File(default=None),
) -> dict:

    try:
        original_image = await read_rgb_image(
            image,
            "image",
        )

    except HTTPException:
        raise

    try:
        with torch.no_grad():

            tensor = (
                transform(original_image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(tensor)

            probability = torch.sigmoid(logits)

            predicted_mask = (
                probability > 0.5
            ).float()

            predicted_mask_array = (
                predicted_mask[0, 0]
                .cpu()
                .numpy()
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again.",
        ) from exc

    mask_image, overlay_image = build_mask_and_overlay(
        original_image,
        predicted_mask_array,
    )

    response: dict[str, object] = {
        "success": True,
        "original_image": encode_image(
            original_image
        ),
        "mask_image": encode_image(
            mask_image
        ),
        "overlay_image": encode_image(
            overlay_image
        ),
        "metrics_available": False,
    }

    if ground_truth_mask is not None:

        try:
            ground_truth_image = await read_rgb_image(
                ground_truth_mask,
                "ground_truth_mask",
            )

            ground_truth_binary = to_binary_mask(
                ground_truth_image
            )

            metrics = compute_metrics(
                predicted_mask_array,
                ground_truth_binary,
            )

            response["metrics_available"] = True
            response["metrics"] = metrics

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Ground-truth mask could not be processed.",
            ) from exc

    return response