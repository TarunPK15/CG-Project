"""
downloader.py - Automated Downloader & Cache for Real-World 3D Medical & Scientific Datasets.
Downloads real clinical CT, MRI, and benchmark volumes (in NIfTI / RAW format),
caches them locally in 'data/', and loads them into Volume instances.
Uses fast global CDN mirrors (jsDelivr / GitHub) with automatic fallback and retries.
"""

import os
import sys
import gzip
import time
import struct
import urllib.request
from typing import Dict, Optional, List
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.volume import Volume


DATASETS_CATALOG: Dict[str, dict] = {
    "brain_mri": {
        "name": "Clinical Human Brain MRI (T1-weighted)",
        "urls": [
            "https://cdn.jsdelivr.net/gh/neurolabusc/VolumeRenderingData@master/stroke/T1.nii.gz",
            "https://raw.githubusercontent.com/neurolabusc/VolumeRenderingData/master/stroke/T1.nii.gz",
        ],
        "filename": "brain_t1.nii.gz",
        "description": "Real clinical T1-weighted 3D brain MRI scan (140 x 160 x 124 voxels, 1mm isotropic).",
        "preset": "brain",
        "recommended_method": "levoy",
    },
    "brain_flair": {
        "name": "Clinical Human Brain MRI (FLAIR)",
        "urls": [
            "https://cdn.jsdelivr.net/gh/neurolabusc/VolumeRenderingData@master/stroke/FLAIR.nii.gz",
            "https://raw.githubusercontent.com/neurolabusc/VolumeRenderingData/master/stroke/FLAIR.nii.gz",
        ],
        "filename": "brain_flair.nii.gz",
        "description": "Clinical Fluid-Attenuated Inversion Recovery (FLAIR) brain scan highlighting tissue boundaries.",
        "preset": "brain",
        "recommended_method": "levoy",
    },
    "bonsai": {
        "name": "Bonsai Tree CT Scan",
        "urls": [
            "https://cdn.jsdelivr.net/gh/neurolabusc/VolumeRenderingData@master/OpenSciVis/bonsai.nii.gz",
            "https://raw.githubusercontent.com/neurolabusc/VolumeRenderingData/master/OpenSciVis/bonsai.nii.gz",
        ],
        "filename": "bonsai.nii.gz",
        "description": "The classic OpenSciVis / Stanford Bonsai Tree CT scan (256 x 256 x 128 voxels).",
        "preset": "bonsai",
        "recommended_method": "drebin",
    },
    "engine": {
        "name": "Engine Block Industrial CT",
        "urls": [
            "https://cdn.jsdelivr.net/gh/neurolabusc/VolumeRenderingData@master/OpenSciVis/engine.nii.gz",
            "https://raw.githubusercontent.com/neurolabusc/VolumeRenderingData/master/OpenSciVis/engine.nii.gz",
        ],
        "filename": "engine.nii.gz",
        "description": "Classic industrial CT scan of a motor engine block (256 x 256 x 110 voxels).",
        "preset": "engine",
        "recommended_method": "levoy",
    },
    "fuel": {
        "name": "Fuel Injection Volumetric Flow",
        "urls": [
            "https://cdn.jsdelivr.net/gh/neurolabusc/VolumeRenderingData@master/OpenSciVis/fuel.nii.gz",
            "https://raw.githubusercontent.com/neurolabusc/VolumeRenderingData/master/OpenSciVis/fuel.nii.gz",
        ],
        "filename": "fuel.nii.gz",
        "description": "Volumetric simulation of fuel injection into a combustion chamber (64 x 33 x 32 voxels).",
        "preset": "shells",
        "recommended_method": "drebin",
    },
    "avm": {
        "name": "Clinical CT Angiogram (AVM)",
        "urls": [
            "https://cdn.jsdelivr.net/gh/neurolabusc/VolumeRenderingData@master/Prism/AVM.nii.gz",
            "https://raw.githubusercontent.com/neurolabusc/VolumeRenderingData/master/Prism/AVM.nii.gz",
        ],
        "filename": "avm.nii.gz",
        "description": "Real clinical CT cerebral angiogram depicting an Arteriovenous Malformation (vascular structure).",
        "preset": "bone_only",
        "recommended_method": "levoy",
    },
}


def parse_nifti_bytes(data_bytes: bytes, filename: str = "volume") -> Volume:
    """
    Parses uncompressed NIfTI-1 bytes into a Volume instance with physical voxel spacing.
    Pure Python / NumPy implementation (zero external C++ dependencies).
    """
    if len(data_bytes) < 348:
        raise ValueError("File is too small to be a valid NIfTI header.")

    # 1. Read header fields
    sizeof_hdr = struct.unpack("<i", data_bytes[:4])[0]
    if sizeof_hdr != 348:
        sizeof_hdr_be = struct.unpack(">i", data_bytes[:4])[0]
        endian = ">" if sizeof_hdr_be == 348 else "<"
    else:
        endian = "<"

    dim = struct.unpack(f"{endian}8h", data_bytes[40:56])
    ndim = dim[0]
    dims = dim[1 : ndim + 1]

    datatype = struct.unpack(f"{endian}h", data_bytes[70:72])[0]
    pixdim = struct.unpack(f"{endian}8f", data_bytes[76:108])
    vox_offset = int(struct.unpack(f"{endian}f", data_bytes[108:112])[0])

    dtype_map = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        512: np.uint16,
    }
    voxel_dtype = dtype_map.get(datatype, np.uint8)

    # Physical spacing (dx, dy, dz)
    dx = float(pixdim[1]) if len(pixdim) > 1 and pixdim[1] > 0 else 1.0
    dy = float(pixdim[2]) if len(pixdim) > 2 and pixdim[2] > 0 else 1.0
    dz = float(pixdim[3]) if len(pixdim) > 3 and pixdim[3] > 0 else 1.0

    # 2. Extract voxel buffer
    raw_voxels = np.frombuffer(data_bytes[vox_offset:], dtype=voxel_dtype)

    expected_size = int(np.prod(dims))
    if raw_voxels.size < expected_size:
        raise ValueError(f"Incomplete voxel data: expected {expected_size}, got {raw_voxels.size}")

    raw_voxels = raw_voxels[:expected_size].reshape(dims[::-1])

    # 3. Normalize to [0, 1] float32
    v_min = float(raw_voxels.min())
    v_max = float(raw_voxels.max())
    if v_max > v_min:
        arr_norm = ((raw_voxels.astype(np.float32) - v_min) / (v_max - v_min)).astype(np.float32)
    else:
        arr_norm = np.zeros_like(raw_voxels, dtype=np.float32)

    # Center origin
    extent_x = (arr_norm.shape[2] - 1) * dx
    extent_y = (arr_norm.shape[1] - 1) * dy
    extent_z = (arr_norm.shape[0] - 1) * dz
    origin = (-0.5 * extent_x, -0.5 * extent_y, -0.5 * extent_z)

    vol = Volume(
        data=arr_norm,
        spacing=(dx, dy, dz),
        origin=origin,
        name=os.path.splitext(os.path.basename(filename))[0],
    )
    vol.normalize_to_unit_cube()
    return vol


def download_dataset(dataset_key: str, data_dir: str = "data") -> str:
    """
    Downloads a dataset by key from the catalog if not already cached.
    Uses mirror fallbacks and retry logic.
    Returns path to local cached file.
    """
    key = dataset_key.lower().strip()
    if key not in DATASETS_CATALOG:
        available = ", ".join(DATASETS_CATALOG.keys())
        raise ValueError(f"Unknown dataset '{key}'. Available: {available}")

    entry = DATASETS_CATALOG[key]
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, entry["filename"])

    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000:
        print(f"Dataset '{key}' already cached at: {local_path}")
        return local_path

    print(f"Downloading '{entry['name']}'...")
    urls = entry.get("urls", [])

    last_err = None
    for url in urls:
        for attempt in range(2):
            try:
                print(f" -> Connecting to {url} (attempt {attempt + 1})...")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    content = resp.read()

                with open(local_path, "wb") as f:
                    f.write(content)

                print(f"Download complete ({len(content):,} bytes) -> {local_path}")
                return local_path
            except Exception as e:
                last_err = e
                time.sleep(1.0)

    raise RuntimeError(f"Failed to download dataset '{key}' from all mirrors: {last_err}")


def load_real_dataset(dataset_key: str, data_dir: str = "data") -> Volume:
    """Downloads (if necessary) and loads a real dataset into a Volume instance."""
    local_path = download_dataset(dataset_key, data_dir)

    if local_path.endswith(".gz"):
        with gzip.open(local_path, "rb") as f:
            data_bytes = f.read()
    else:
        with open(local_path, "rb") as f:
            data_bytes = f.read()

    return parse_nifti_bytes(data_bytes, filename=local_path)


def list_available_datasets():
    """Prints catalog of available real datasets."""
    print("=" * 70)
    print("CATALOG OF REAL 3D MEDICAL & SCIENTIFIC DATASETS")
    print("=" * 70)
    for key, info in DATASETS_CATALOG.items():
        print(f" • Key: {key:<14} | Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Recommended: --method {info['recommended_method']} --preset {info['preset']}")
        print("-" * 70)
