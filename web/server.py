"""
server.py - High-Performance FastAPI Web Backend for 3D Volume Rendering Dashboard.
Serves interactive REST API for ray casting, 2D slice inspection, and ground truth evaluation.
"""

import os
import sys
import io
import time
import base64
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

from core.volume import Volume
from core.camera import Camera
from core.raycaster import (
    VolumeRenderer,
    RaycastSettings,
    RenderMethod,
    CompositingMode,
)
from core.levoy import LevoyRendererConfig, LevoySurface
from core.drebin import DrebinClassifier, DrebinMaterial
from core.shading import LightingConfig
from datasets.loader import load_volume_any
from datasets.downloader import DATASETS_CATALOG, download_dataset
from datasets.generator import create_marschner_lobb

app = FastAPI(title="3D Direct Volume Rendering & Surface Display Dashboard")

# Global Volume and Renderer State
CURRENT_DATASET_KEY = "brain_mri"
CURRENT_VOLUME: Optional[Volume] = None
CURRENT_RENDERER: Optional[VolumeRenderer] = None


def get_default_dataset_path(key: str) -> Optional[str]:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    if key in DATASETS_CATALOG:
        filename = DATASETS_CATALOG[key]["filename"]
        path = os.path.join(data_dir, filename)
        if os.path.isfile(path):
            return path
    # Check if key is a direct filename in data/
    direct_path = os.path.join(data_dir, key)
    if os.path.isfile(direct_path):
        return direct_path
    return None


def init_volume(key: str = "brain_mri"):
    global CURRENT_DATASET_KEY, CURRENT_VOLUME, CURRENT_RENDERER

    if key == "marschner_lobb":
        vol = create_marschner_lobb(res=80)
        vol.normalize_to_unit_cube()
        vol.compute_gradients()
        CURRENT_VOLUME = vol
        CURRENT_DATASET_KEY = key
        CURRENT_RENDERER = VolumeRenderer(CURRENT_VOLUME)
        return

    path = get_default_dataset_path(key)
    if path is None or not os.path.isfile(path):
        # Try downloading
        if key in DATASETS_CATALOG:
            path = download_dataset(key, data_dir=os.path.join(PROJECT_ROOT, "data"))
        else:
            raise FileNotFoundError(f"Dataset '{key}' not found locally or in catalog.")

    vol = load_volume_any(path)
    vol.normalize_to_unit_cube()
    vol.compute_gradients()

    CURRENT_VOLUME = vol
    CURRENT_DATASET_KEY = key
    CURRENT_RENDERER = VolumeRenderer(CURRENT_VOLUME)


# Initialize default volume on startup
try:
    init_volume("brain_mri")
except Exception as e:
    print(f"Initial volume load warning: {e}. Falling back to marschner_lobb.")
    init_volume("marschner_lobb")


class RenderRequest(BaseModel):
    dataset_key: Optional[str] = "brain_mri"
    method: int = 0                  # 0: Levoy, 1: Drebin, 2: MIP
    compositing: int = 1             # 0: Back-to-Front, 1: Front-to-Back
    step_size: float = 0.5
    width: int = 256
    height: int = 256
    # Camera
    azimuth: float = 45.0
    elevation: float = 25.0
    distance: float = 2.5
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.0
    fov_y: float = 40.0
    # Levoy parameters
    levoy_iso1: float = 0.32
    levoy_thick1: float = 1.5
    levoy_alpha1: float = 0.75
    levoy_color1: list = [0.92, 0.76, 0.65]
    levoy_iso2: float = 0.60
    levoy_thick2: float = 2.0
    levoy_alpha2: float = 0.90
    levoy_color2: list = [0.95, 0.95, 0.90]
    # Drebin parameters
    drebin_smin: float = 0.15
    drebin_sp1: float = 0.25
    drebin_sp2: float = 0.50
    drebin_smax: float = 0.85
    # Clipping / Matte
    clip_enabled: bool = False
    clip_plane_d: float = -9999.0
    wedge_cut: bool = False
    # Lighting
    ka: float = 0.2
    kd: float = 0.65
    ks: float = 0.35
    shininess: float = 30.0


@app.get("/api/datasets")
def list_datasets():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    local_files = os.listdir(data_dir) if os.path.isdir(data_dir) else []

    datasets = [
        {
            "key": "brain_mri",
            "name": "Clinical Human Brain MRI (T1-weighted)",
            "subject": "Adult Human Neuroanatomy (Healthy Volunteer)",
            "creator": "Chris Rorden / Neurolabusc / NIH Neuroimaging Benchmark",
            "date": "Clinical Acquisition (SPM12 / OASIS Project)",
            "modality": "3.0 Tesla MRI (T1-weighted 3D MPRAGE)",
            "shape": "140 x 160 x 124 voxels (2.77M voxels)",
            "spacing": "1.00 mm isotropic (dx=1.0, dy=1.0, dz=1.0)",
            "structures": "Cerebral cortex, sulci & gyri, corpus callosum, lateral ventricles, cerebellum, brainstem",
            "significance": "Demonstrates Levoy (1988) continuous isosurface display for separating brain soft tissue from ventricular fluid without polygonal mesh extraction.",
            "recommended": "Levoy (1988) Isosurface Display (Skin f_v=0.32, Ventricles f_v=0.60)",
            "cached": "brain_t1.nii.gz" in local_files,
        },
        {
            "key": "bonsai",
            "name": "Bonsai Tree CT Benchmark Scan",
            "subject": "Miniature Picea abies Bonsai Tree in Ceramic Pot",
            "creator": "Stefan Röttger (University of Stuttgart / The Volume Library)",
            "date": "1998 (Canonical Computer Graphics Benchmark)",
            "modality": "High-Resolution X-Ray Computed Tomography (Industrial Micro-CT)",
            "shape": "256 x 256 x 256 voxels (16.78M voxels, uint8)",
            "spacing": "0.50 mm isotropic",
            "structures": "Fine botanical needles, woody branch bifurcations, annular growth rings in trunk, root system, soil matrix, clay pot",
            "significance": "The gold-standard volume graphics benchmark for evaluating probabilistic material mixtures (Drebin 1988) and gradient-based transfer functions across complex organic boundaries.",
            "recommended": "Drebin (1988) Material Mixture (Soft Wood s=0.15, Dense Wood s=0.45, Clay Pot s=0.85)",
            "cached": "bonsai.nii.gz" in local_files,
        },
        {
            "key": "engine",
            "name": "Engine Block Industrial CT",
            "subject": "Two-Cylinder 4-Stroke Combustion Engine Block",
            "creator": "General Electric Corporate R&D / Stanford Volume Rendering Library",
            "date": "1995 (Stanford Volume Archive)",
            "modality": "High-Energy Industrial CT Scanner (Cast Metal Non-Destructive Testing)",
            "shape": "256 x 256 x 128 voxels (8.39M voxels)",
            "spacing": "1.00 mm isotropic",
            "structures": "Cylinder bore cavities, piston sleeves, water cooling jackets, crankcase casting, spark plug channels",
            "significance": "Classic industrial NDT volume for demonstrating Blinn-Phong specular metallic highlights, crisp normal estimation, and Drebin 3D wedge cutaways into hollow cavities.",
            "recommended": "Levoy (1988) Metallic Isosurface (f_v=0.35, Specular ks=0.50, Shininess=40)",
            "cached": "engine.nii.gz" in local_files,
        },
        {
            "key": "avm",
            "name": "Clinical CT Angiogram (AVM)",
            "subject": "Clinical Patient Brain with Cerebral Arteriovenous Malformation",
            "creator": "University of Zurich & Freiburg / Neurolabusc Prism Benchmark",
            "date": "Clinical Angiographic Repository",
            "modality": "3D Rotational Digital Subtraction CT Angiography (CT-DSA) with iodinated contrast",
            "shape": "337 x 319 x 154 voxels (16.56M voxels)",
            "spacing": "0.55 x 0.55 x 1.00 mm",
            "structures": "Feeding cerebral arteries, tangled vascular malformation nidus, draining veins, skull base bone",
            "significance": "Critical clinical neurosurgical planning volume demonstrating how direct ray casting exposes tortuous vascular malformations without manual surface segmentation.",
            "recommended": "Levoy (1988) Vascular Isosurface (f_v=0.22, High Diffuse kd=0.75)",
            "cached": "avm.nii.gz" in local_files,
        },
        {
            "key": "brain_flair",
            "name": "Clinical Human Brain MRI (FLAIR)",
            "subject": "Clinical Human Patient (Fluid-Attenuated Inversion Recovery)",
            "creator": "Neurolabusc Stroke & Neuroimaging Benchmark",
            "date": "Clinical Acquisition",
            "modality": "Fluid-Attenuated Inversion Recovery (FLAIR) 3D MRI",
            "shape": "140 x 160 x 124 voxels",
            "spacing": "1.00 mm isotropic",
            "structures": "Periventricular hyperintensities, cortical ribbon boundaries, gray/white matter interface",
            "significance": "Tests volume rendering contrast sensitivity when cerebrospinal fluid signal is nulled to highlight tissue boundary transitions.",
            "recommended": "Levoy (1988) Isosurface (f_v=0.28)",
            "cached": "brain_flair.nii.gz" in local_files,
        },
        {
            "key": "marschner_lobb",
            "name": "Marschner-Lobb Benchmark (1994)",
            "subject": "High-Frequency Cosine Modulated 3D Scalar Field",
            "creator": "Stephen R. Marschner & Richard J. Lobb (IEEE Visualization 1994)",
            "date": "1994 (Foundational Filter Quality Benchmark)",
            "modality": "Analytical Mathematical Scalar Field: f(x,y,z) = (1 - sin(pi*z/2) + alpha(1+rho(r))) / (2(1+alpha))",
            "shape": "80 x 80 x 80 voxels",
            "spacing": "Normalized [-1, 1]^3",
            "structures": "Complex high-frequency concentric wave ripples testing reconstruction filter sharpness and central difference gradient accuracy",
            "significance": "Standard benchmark used by every volume rendering paper to quantify reconstruction filter smoothing and gradient normal precision.",
            "recommended": "Levoy (1988) Isosurface (f_v=0.50, r=1.5 voxels)",
            "cached": True,
        },
    ]
    return {"datasets": datasets, "current": CURRENT_DATASET_KEY}


@app.post("/api/load_dataset")
def load_dataset_endpoint(payload: Dict[str, str]):
    key = payload.get("key", "brain_mri")
    try:
        t0 = time.time()
        init_volume(key)
        dur = time.time() - t0
        return {
            "status": "success",
            "dataset": key,
            "shape": list(CURRENT_VOLUME.shape),
            "load_time_sec": round(dur, 3),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/render")
def render_volume_endpoint(req: RenderRequest):
    global CURRENT_VOLUME, CURRENT_RENDERER, CURRENT_DATASET_KEY

    # Check if dataset switch requested
    if req.dataset_key and req.dataset_key != CURRENT_DATASET_KEY:
        init_volume(req.dataset_key)

    if CURRENT_VOLUME is None or CURRENT_RENDERER is None:
        raise HTTPException(status_code=500, detail="Volume engine not initialized.")

    # Configure Camera
    cam = Camera(
        target=(req.target_x, req.target_y, req.target_z),
        azimuth=req.azimuth,
        elevation=req.elevation,
        distance=req.distance,
        fov_y=req.fov_y,
    )

    # Configure Settings
    settings = RaycastSettings(
        method=RenderMethod(req.method),
        compositing=CompositingMode(req.compositing),
        step_size=req.step_size,
        clip_plane_d=req.clip_plane_d if req.clip_enabled else -9999.0,
        wedge_cut=req.wedge_cut,
    )
    CURRENT_RENDERER.settings = settings

    # Configure Lighting
    CURRENT_RENDERER.lighting.ka = req.ka
    CURRENT_RENDERER.lighting.kd = req.kd
    CURRENT_RENDERER.lighting.ks = req.ks
    CURRENT_RENDERER.lighting.shininess = req.shininess

    # Configure Levoy Multi-Surface
    if req.method == 0:
        c1 = np.array(req.levoy_color1, dtype=np.float32)
        c2 = np.array(req.levoy_color2, dtype=np.float32)
        CURRENT_RENDERER.levoy_config = LevoyRendererConfig([
            LevoySurface(isovalue=req.levoy_iso1, thickness=req.levoy_thick1, alpha_v=req.levoy_alpha1, color=c1),
            LevoySurface(isovalue=req.levoy_iso2, thickness=req.levoy_thick2, alpha_v=req.levoy_alpha2, color=c2),
        ])

    # Configure Drebin Classifier
    elif req.method == 1:
        # 3 materials: Soft Tissue, Bone, Dense Bone
        CURRENT_RENDERER.drebin_classifier = DrebinClassifier([
            DrebinMaterial(
                name="Soft Tissue",
                s_min=req.drebin_smin * 0.8,
                s_peak_start=req.drebin_smin,
                s_peak_end=req.drebin_sp1,
                s_max=req.drebin_sp2,
                color=np.array([0.90, 0.65, 0.50], dtype=np.float32),
                opacity=0.35,
                density_rho=1.0,
            ),
            DrebinMaterial(
                name="Bone",
                s_min=req.drebin_sp1,
                s_peak_start=req.drebin_sp2,
                s_peak_end=req.drebin_smax,
                s_max=min(1.0, req.drebin_smax + 0.15),
                color=np.array([0.95, 0.95, 0.90], dtype=np.float32),
                opacity=0.90,
                density_rho=2.0,
            ),
        ])

    t0 = time.time()
    img_rgba = CURRENT_RENDERER.render(cam, width=req.width, height=req.height)
    render_time = time.time() - t0

    # Encode to PNG Base64
    pil_img = Image.fromarray(img_rgba)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_png = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "render_time_ms": round(render_time * 1000.0, 1),
        "fps": round(1.0 / max(1e-4, render_time), 1),
        "width": req.width,
        "height": req.height,
        "image": f"data:image/png;base64,{b64_png}",
    }


@app.get("/api/slice")
def get_slice(axis: str = "axial", index_pct: float = 0.5):
    """
    Returns 2D orthogonal slice (axial: Z, coronal: Y, sagittal: X) as PNG base64.
    """
    if CURRENT_VOLUME is None:
        raise HTTPException(status_code=500, detail="No volume loaded.")

    data = CURRENT_VOLUME.data
    nz, ny, nx = data.shape

    pct = np.clip(index_pct, 0.0, 1.0)
    if axis.lower() == "axial":
        idx = int(pct * (nz - 1))
        slice_2d = data[idx, :, :]
    elif axis.lower() == "coronal":
        idx = int(pct * (ny - 1))
        slice_2d = data[:, idx, :]
    else:  # sagittal
        idx = int(pct * (nx - 1))
        slice_2d = data[:, :, idx]

    # Normalize to uint8
    s_min = slice_2d.min()
    s_max = slice_2d.max()
    if s_max > s_min:
        norm = ((slice_2d - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(slice_2d, dtype=np.uint8)

    # Apply colormap (Bone / Medical cyan-white)
    im = Image.fromarray(norm).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "axis": axis,
        "index": idx,
        "max_index": (nz if axis == "axial" else (ny if axis == "coronal" else nx)) - 1,
        "image": f"data:image/png;base64,{b64}",
    }


@app.get("/api/reference_image")
def get_reference_image(dataset_key: Optional[str] = None):
    key = dataset_key or CURRENT_DATASET_KEY
    ref_dir = os.path.join(PROJECT_ROOT, "reference_scans")

    # Map dataset key to reference file
    ref_map = {
        "brain_mri": "brain_t1_ref.png",
        "brain_flair": "brain_t1_ref.png",
        "bonsai": "bonsai_ref.png",
        "engine": "engine_ref.png",
        "avm": "avm_ref.png",
        "procedural_head": "ct_ref.jpg",
        "marschner_lobb": "things_ref.jpg",
    }
    filename = ref_map.get(key, "ct_ref.jpg")
    filepath = os.path.join(ref_dir, filename)

    if not os.path.isfile(filepath):
        # Fallback to any existing image
        files = [f for f in os.listdir(ref_dir) if f.endswith((".png", ".jpg"))]
        if files:
            filepath = os.path.join(ref_dir, files[0])
        else:
            raise HTTPException(status_code=404, detail="No reference image found.")

    im = Image.open(filepath)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "dataset": key,
        "filename": filename,
        "image": f"data:image/png;base64,{b64}",
    }


@app.post("/api/compare")
def compare_with_reference(payload: Dict[str, Any]):
    """
    Computes quantitative evaluation metrics (PSNR, SSIM, Pearson r) between
    the rendered frame and the ground-truth dataset reference image, and generates
    a pixel-difference heatmap.
    """
    current_img_b64 = payload.get("current_image", "")
    dataset_key = payload.get("dataset_key", CURRENT_DATASET_KEY)

    if not current_img_b64.startswith("data:image"):
        raise HTTPException(status_code=400, detail="Invalid current image data URI.")

    # Decode current render
    raw_b64 = current_img_b64.split(",")[1]
    curr_im = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")

    # Load reference image
    ref_resp = get_reference_image(dataset_key)
    ref_b64 = ref_resp["image"].split(",")[1]
    ref_im = Image.open(io.BytesIO(base64.b64decode(ref_b64))).convert("RGB")

    # Resize reference to match current render resolution
    ref_im_resized = ref_im.resize(curr_im.size, Image.Resampling.BILINEAR)

    arr_curr = np.array(curr_im, dtype=np.float32) / 255.0
    arr_ref = np.array(ref_im_resized, dtype=np.float32) / 255.0

    # 1. Mean Squared Error (MSE) & PSNR
    mse = float(np.mean((arr_curr - arr_ref) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(1e-10, mse))) if mse > 1e-10 else 99.9

    # 2. Pearson Correlation Coefficient
    c_flat = arr_curr.flatten()
    r_flat = arr_ref.flatten()
    c_mean = np.mean(c_flat)
    r_mean = np.mean(r_flat)
    num = np.sum((c_flat - c_mean) * (r_flat - r_mean))
    den = np.sqrt(np.sum((c_flat - c_mean) ** 2) * np.sum((r_flat - r_mean) ** 2))
    pearson_r = float(num / den) if den > 1e-8 else 0.0

    # 3. Approximate SSIM (Structural Similarity)
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    var_c = float(np.var(c_flat))
    var_r = float(np.var(r_flat))
    cov_cr = float(np.mean((c_flat - c_mean) * (r_flat - r_mean)))
    ssim = float(((2.0 * c_mean * r_mean + c1) * (2.0 * cov_cr + c2)) /
                 ((c_mean**2 + r_mean**2 + c1) * (var_c + var_r + c2)))

    # 4. Generate Diff Heatmap (Jet / Fire colormap)
    diff = np.abs(arr_curr - arr_ref).mean(axis=2)  # shape (H, W) in [0, 1]
    # Simple colormap: low diff = dark blue/green, high diff = yellow/red
    diff_rgb = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    diff_rgb[:, :, 0] = (np.clip(diff * 2.5, 0.0, 1.0) * 255).astype(np.uint8)  # Red
    diff_rgb[:, :, 1] = (np.clip((1.0 - np.abs(diff * 2.0 - 0.5)) * 1.5, 0.0, 1.0) * 255).astype(np.uint8)  # Green
    diff_rgb[:, :, 2] = (np.clip((1.0 - diff * 2.0), 0.0, 1.0) * 200).astype(np.uint8)  # Blue

    diff_im = Image.fromarray(diff_rgb)
    buf = io.BytesIO()
    diff_im.save(buf, format="PNG")
    diff_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "psnr_db": round(psnr, 2),
        "ssim": round(np.clip(ssim, -1.0, 1.0), 4),
        "pearson_r": round(np.clip(pearson_r, -1.0, 1.0), 4),
        "mse": round(mse, 5),
        "diff_image": f"data:image/png;base64,{diff_b64}",
    }


# Mount static files directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Volume Rendering Engine Web UI Loading...</h1>"
