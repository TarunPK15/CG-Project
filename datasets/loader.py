"""
loader.py - Flexible 3D Volume Data File Loaders.
Supports:
    - NIfTI medical format (.nii, .nii.gz)
    - NumPy .npy and .npz
    - Raw binary volumes (.raw, .dat)
    - 2D image slice stacks (.png, .jpg, .tif)
"""

import os
import gzip
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from core.volume import Volume


def load_volume_from_nii(path: str) -> Volume:
    """Loads a NIfTI-1 file (.nii or .nii.gz) into a Volume instance."""
    from datasets.downloader import parse_nifti_bytes

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data_bytes = f.read()
    else:
        with open(path, "rb") as f:
            data_bytes = f.read()

    return parse_nifti_bytes(data_bytes, filename=path)


def load_volume_from_npy(path: str, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Volume:
    """Loads a 3D NumPy array from .npy file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got shape {arr.shape}")

    arr = arr.astype(np.float32)
    val_max = arr.max()
    val_min = arr.min()
    if val_max > 1.0 or val_min < 0.0:
        if val_max > val_min:
            arr = (arr - val_min) / (val_max - val_min)

    basename = os.path.splitext(os.path.basename(path))[0]
    vol = Volume(arr, spacing=spacing, name=basename)
    vol.normalize_to_unit_cube()
    return vol


def load_volume_from_raw(
    path: str,
    shape: Tuple[int, int, int],  # (Nz, Ny, Nx)
    dtype: np.dtype = np.uint8,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    endian: str = "little",
) -> Volume:
    """Loads raw binary voxel data."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.fromfile(path, dtype=dtype)
    expected_size = shape[0] * shape[1] * shape[2]
    if arr.size != expected_size:
        raise ValueError(
            f"File size mismatch: got {arr.size} elements, expected {expected_size} for shape {shape}"
        )

    arr = arr.reshape(shape).astype(np.float32)
    val_max = arr.max()
    val_min = arr.min()
    if val_max > val_min:
        arr = (arr - val_min) / (val_max - val_min)

    basename = os.path.splitext(os.path.basename(path))[0]
    vol = Volume(arr, spacing=spacing, name=basename)
    vol.normalize_to_unit_cube()
    return vol


def load_volume_from_image_stack(
    folder_path: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_slices: Optional[int] = None,
) -> Volume:
    """Loads a 3D volume from a folder containing sorted 2D image slices."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Directory not found: {folder_path}")

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = sorted(
        [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
    )

    if not files:
        raise ValueError(f"No image slices found in {folder_path}")

    if max_slices is not None:
        files = files[:max_slices]

    slices = []
    for f in files:
        img_path = os.path.join(folder_path, f)
        img = Image.open(img_path).convert("L")  # Grayscale
        slices.append(np.array(img, dtype=np.float32))

    arr = np.stack(slices, axis=0)  # (Nz, Ny, Nx)
    arr = arr / 255.0

    basename = os.path.basename(os.path.normpath(folder_path))
    vol = Volume(arr, spacing=spacing, name=basename)
    vol.normalize_to_unit_cube()
    return vol


def load_volume_any(path: str) -> Volume:
    """Automatically detects file format and loads Volume."""
    lower = path.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_volume_from_nii(path)
    elif lower.endswith(".npy"):
        return load_volume_from_npy(path)
    elif os.path.isdir(path):
        return load_volume_from_image_stack(path)
    else:
        raise ValueError(f"Unsupported volume file format for '{path}'. Supported: .nii, .nii.gz, .npy, image stack directory.")


def save_volume_to_npy(volume: Volume, path: str):
    """Saves volume to .npy format."""
    np.save(path, volume.data)
