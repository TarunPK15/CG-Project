"""
main.py - Master Unified Entry Point for Direct Volume Rendering Engine.
Course: IS F311 Computer Graphics
Project: Direct Volume Rendering and Surface Display from 3D Voxel Data
Authors: Tarun P Krishnamurthy, Harsh Gunda, Ojas Dubey
References:
    - Robert A. Drebin, Loren Carpenter, Pat Hanrahan (1988). "Volume Rendering". ACM SIGGRAPH.
    - Marc Levoy (1988). "Display of Surfaces from Volume Data". IEEE Computer Graphics & Applications.

Usage:
    python main.py                     # Launch Modern Interactive Web Dashboard (FastAPI + HTML5)
    python main.py --web               # Launch Modern Interactive Web Dashboard
    python main.py --cli --dataset brain_mri --method levoy --output brain.png # Render CLI
    python main.py --compare           # Render comparison suite images to output/
    python main.py --benchmark         # Run performance benchmarks
    python main.py --test              # Run complete unit test suite
"""

import os
import sys
import argparse
import webbrowser
import threading
import time
import numpy as np
from PIL import Image

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.camera import Camera
from core.raycaster import (
    VolumeRenderer,
    RaycastSettings,
    RenderMethod,
    CompositingMode,
)
from core.shading import LightingConfig
from datasets.generator import create_marschner_lobb
from datasets.loader import load_volume_any
from datasets.downloader import (
    DATASETS_CATALOG,
    download_dataset,
    load_real_dataset,
    list_available_datasets,
)
from transfer_functions.presets import get_levoy_preset, get_drebin_preset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct Volume Rendering and Surface Display from 3D Voxel Data (Drebin 1988 & Levoy 1988)"
    )
    parser.add_argument("--web", action="store_true", help="Launch interactive Web Dashboard (default)")
    parser.add_argument("--port", type=int, default=8000, help="Port for web dashboard (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on launch")
    parser.add_argument("--cli", action="store_true", help="Run in Command-Line Interface mode")
    parser.add_argument("--list-datasets", action="store_true", help="List all available real medical/scientific datasets")
    parser.add_argument("--download", type=str, default=None, help="Download a real dataset by name (e.g. brain_mri, bonsai, engine, fuel, all)")
    parser.add_argument("--compare", action="store_true", help="Render all comparison suite images to output/")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks and FPS measurements")
    parser.add_argument("--test", action="store_true", help="Run unit test suite")

    # CLI rendering options
    parser.add_argument(
        "--dataset",
        type=str,
        default="brain_mri",
        help="3D volume dataset: 'brain_mri', 'bonsai', 'engine', 'avm', 'brain_flair', 'head', 'marschner', or path to .nii/.npy",
    )
    parser.add_argument("--file", type=str, default=None, help="Path to custom volume file (.nii, .nii.gz, .npy)")
    parser.add_argument("--method", type=str, default="levoy", choices=["levoy", "drebin", "mip"], help="Rendering algorithm")
    parser.add_argument("--compositing", type=str, default="f2b", choices=["b2f", "f2b"], help="Compositing mode: b2f (back-to-front) or f2b (front-to-back)")
    parser.add_argument("--preset", type=str, default="default", help="Transfer function preset (e.g. default, bone_only, skin_only, brain, bonsai, engine, shells)")
    parser.add_argument("--step", type=float, default=0.5, help="Ray casting step size in voxel units (default: 0.5)")
    parser.add_argument("--width", type=int, default=400, help="Output image width in pixels")
    parser.add_argument("--height", type=int, default=400, help="Output image height in pixels")
    parser.add_argument("--azimuth", type=float, default=45.0, help="Camera azimuth angle in degrees")
    parser.add_argument("--elevation", type=float, default=25.0, help="Camera elevation angle in degrees")
    parser.add_argument("--distance", type=float, default=2.5, help="Camera distance from target")
    parser.add_argument("--wedge", action="store_true", help="Apply Drebin 3D wedge cutaway to reveal interior")
    parser.add_argument("--output", type=str, default="output/volume_render.png", help="Output file path for CLI render")

    return parser.parse_args()


def load_dataset_by_name(name: str, custom_file: str = None):
    """Loads volume by name, catalog key, or file path."""
    if custom_file and os.path.isfile(custom_file):
        vol = load_volume_any(custom_file)
        vol.normalize_to_unit_cube()
        vol.compute_gradients()
        return vol

    clean_name = name.lower().strip()
    if clean_name in ["marschner", "marschner_lobb"]:
        vol = create_marschner_lobb(res=80)
    elif clean_name in DATASETS_CATALOG or clean_name.startswith("brain") or clean_name in ["bonsai", "engine", "fuel", "avm"]:
        key = "brain_mri" if clean_name == "brain" else clean_name
        vol = load_real_dataset(key)
    elif os.path.isfile(name):
        vol = load_volume_any(name)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS_CATALOG.keys())} or 'marschner'")

    vol.normalize_to_unit_cube()
    vol.compute_gradients()
    return vol


def run_web_dashboard(port: int = 8000, open_browser: bool = True):
    """Launches high-performance FastAPI web dashboard with Uvicorn."""
    import uvicorn
    from web.server import app

    url = f"http://127.0.0.1:{port}"
    print("=" * 65)
    print(" 3D DIRECT VOLUME RENDERING & SURFACE DISPLAY ENGINE")
    print(f" Web Dashboard Server running at: {url}")
    print("=" * 65)

    if open_browser:
        def _open():
            time.sleep(1.2)
            print(f"Opening browser at: {url}")
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def run_cli_render(args):
    """Renders a single frame from command line."""
    print(f"Loading dataset '{args.dataset}'...")
    vol = load_dataset_by_name(args.dataset, args.file)

    method_map = {"levoy": RenderMethod.LEVOY, "drebin": RenderMethod.DREBIN, "mip": RenderMethod.MIP}
    comp_map = {"b2f": CompositingMode.BACK_TO_FRONT, "f2b": CompositingMode.FRONT_TO_BACK}

    method = method_map[args.method.lower()]
    comp = comp_map[args.compositing.lower()]

    settings = RaycastSettings(
        method=method,
        compositing=comp,
        step_size=args.step,
        wedge_cut=args.wedge,
    )

    preset_name = args.preset
    if preset_name == "default":
        preset_name = "brain" if "brain" in args.dataset else ("bonsai" if "bonsai" in args.dataset else "default")

    levoy_cfg = get_levoy_preset(preset_name)
    drebin_cls = get_drebin_preset(preset_name)

    cam = Camera(
        target=(0.0, 0.0, 0.0),
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
        fov_y=40.0,
    )

    renderer = VolumeRenderer(vol, settings=settings, levoy_config=levoy_cfg, drebin_classifier=drebin_cls)

    print(f"Rendering {args.width}x{args.height} using {args.method.upper()} ({args.compositing.upper()})...")
    t0 = time.time()
    img_rgba = renderer.render(cam, width=args.width, height=args.height)
    dur = time.time() - t0
    print(f"Render complete in {dur:.2f}s ({1.0/max(1e-4, dur):.1f} FPS)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Image.fromarray(img_rgba).save(args.output)
    print(f"Saved output to: {args.output}")


def main():
    args = parse_args()

    if args.test:
        from tests.test_core import run_all_tests
        success = run_all_tests()
        sys.exit(0 if success else 1)

    if args.list_datasets:
        list_available_datasets()
        sys.exit(0)

    if args.download:
        dl_key = args.download.strip().lower()
        if dl_key == "all":
            for k in DATASETS_CATALOG:
                download_dataset(k)
        else:
            download_dataset(dl_key)
        sys.exit(0)

    if args.cli:
        run_cli_render(args)
        sys.exit(0)

    # Default action: Launch Modern Web Dashboard
    run_web_dashboard(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
