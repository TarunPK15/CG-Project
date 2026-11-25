# Direct Volume Rendering and Surface Display from 3D Voxel Data

**Course:** IS F311 Computer Graphics  
**Date:** November 2026  
**Authors:** Tarun P Krishnamurthy, Harsh Gunda, Ojas Dubey  

---

## 1. Executive Summary & Project Achievements

### What This Project Achieves
This project is an end-to-end, high-performance 3D Direct Volume Rendering (DVR) Engine implemented in Python with JIT-accelerated parallel ray casting. It solves the fundamental problem of visualizing 3D volumetric scalar fields (such as clinical Computed Tomography [CT] and Magnetic Resonance Imaging [MRI] scans) **directly from raw 3D voxel data without extracting intermediate polygonal meshes** (such as Marching Cubes).

By combining the seminal methodologies of:
1. **Marc Levoy (1988)**: *"Display of Surfaces from Volume Data"*, IEEE Computer Graphics & Applications
2. **Robert A. Drebin, Loren Carpenter, and Pat Hanrahan (1988)**: *"Volume Rendering"*, ACM SIGGRAPH Computer Graphics

the engine achieves:
- **Direct Surface & Translucent Mixture Rendering**: Visualizes dense structures (bone, enamel, metallic engine castings) simultaneously with translucent soft tissue layers (brain matter, skin, foliage, vascular trees) with correct depth sorting and Blinn-Phong lighting in a single pass.
- **Continuous Isosurface Display (Levoy 1988)**: Uses continuous gradient magnitude $|\nabla f(x)|$ and continuous trilinear interpolation to modulate isosurface thickness and opacity, producing silky-smooth surfaces free of discrete polygonal stepping or "Minecraft" voxel aliasing.
- **Probabilistic Material Classification (Drebin 1988)**: Decomposes voxels into continuous fractional material occupancies $p_i(x)$ using trapezoidal classification functions, computing effective physical density gradients $\nabla \rho$ and surface strength $S = |\nabla \rho|$ to accentuate physical tissue interfaces.
- **3D Matte Volumes & Cutaways (Drebin 1988)**: Provides interactive clipping planes and 3D wedge cutaways to peel back exterior structures and reveal hidden interior organs and cavities.
- **Dual Compositing Pipelines**:
  - **Back-to-Front Compositing** ($C_{\text{out}} = C_s \alpha_s + (1 - \alpha_s) C_{\text{in}}$): Strictly follows the IS F311 course project proposal specification.
  - **Front-to-Back Compositing**: Traces rays from the eye into the volume with Early Ray Termination ($\alpha \ge 0.99$) for massive computational speedups.
- **Modern Interactive Web Dashboard (FastAPI + HTML5 / Canvas)**: Replaced the slow, frozen Tkinter desktop GUI with a high-performance web dashboard featuring smooth mouse orbit navigation, progressive resolution refinement (instant ~30 ms interaction during mouse drag, high-res $320 \times 320$ on release), and full-width 2D multi-planar orthogonal slicing (Axial, Coronal, Sagittal).
- **Universal Unit Cube Normalization**: Eliminates the camera-inside-volume bug by automatically centering any dataset at $(0, 0, 0)$ and normalizing bounds into $[-1, 1]^3$, ensuring camera distances work uniformly across any dataset resolution.
- **Real Clinical & Benchmark Datasets with Provenance**: Features real clinical Human Brain T1 MRI, Bonsai Tree Micro-CT ($256^3$), Industrial Engine Block CT ($256 \times 256 \times 128$), CT Angiogram (AVM), and Brain FLAIR scans, accompanied by complete scientific provenance metadata (what was scanned, when, by whom, scanner modality, and voxel spacing).

---

## 2. Comparative Analysis: Why and Where Each Rendering Method is Better

The engine implements three distinct volumetric projection paradigms. Each method solves a different visualization challenge and excels in specific clinical, engineering, and scientific scenarios:

| Feature / Metric | 1. Marc Levoy (1988) Isosurface Display | 2. Robert Drebin et al. (1988) Material Mixture | 3. Maximum Intensity Projection (MIP) |
|---|---|---|---|
| **Underlying Philosophy** | Extracts continuous iso-intensity boundary surfaces based on local gradient magnitude. | Models voxels as partial-volume mixtures of multiple physical materials. | Projects the single maximum scalar value encountered along each viewing ray. |
| **Surface Definition** | Gradient-weighted distance falloff: $\alpha(x) = \alpha_v \left(1 - \frac{1}{r} \left\|\frac{f(x) - f_v}{\|\nabla f(x)\|}\right\|\right)$ | Trapezoidal material percentages $p_i(x)$ modulated by density surface strength $S = \|\nabla \rho\|$. | No surface definition; purely value-based ray search. |
| **Lighting & Shading** | Full Blinn-Phong illumination on normalized gradient normal $\vec{N} = \frac{\nabla f}{\|\nabla f\|}$. | Full Blinn-Phong illumination on density gradient normal $\vec{N} = \frac{\nabla \rho}{\|\nabla \rho\|}$. | None (unshaded scalar intensity mapping). |
| **Normal Computation** | Required (central differences of scalar field). | Required (central differences of density field $\rho$). | Not needed (zero gradient overhead). |
| **Computational Cost** | Moderate (~0.7s per frame at $200 \times 200$; ~1.8s at $320 \times 320$). | Moderate (~1.0s per frame at $200 \times 200$). | Blazingly Fast (~0.08s per frame; >12 FPS on CPU). |
| **View-Order Invariant?** | No (Back-to-Front or Front-to-Back accumulation). | No (Back-to-Front or Front-to-Back accumulation). | **Yes** (commutative maximum operation). |
| **Multi-Layer Capability** | Excellent for distinct concentric shells / surfaces (e.g., skin + bone). | Excellent for fuzzy, continuous multi-substance mixtures (wood, roots, soil). | Flattens 3D depth into a 2D maximum projection. |

---

### Detailed Method Breakdown:

### 1. Marc Levoy (1988) Gradient-Weighted Surface Display
- **Mathematical Principle**: Rather than fitting geometric polygons to an isovalue, Levoy assigns opacity based on the spatial distance to the target isosurface:
  $$\alpha(x) = \alpha_v \left( 1 - \frac{1}{r} \left| \frac{f(x) - f_v}{|\nabla f(x)|} \right| \right)$$
  when $|f(x) - f_v| \le r |\nabla f(x)|$, else $0$. Multiple isosurfaces (e.g. skin $f_{v1}=0.32$ and bone $f_{v2}=0.60$) are blended via:
  $$\alpha_{\text{total}} = 1 - \prod_{m=1}^M (1 - \alpha_m), \quad C_{\text{diffuse}} = \frac{\sum \alpha_m C_m}{\sum \alpha_m + \epsilon}$$
- **Why It Is Better**:
  - **Preserves Crisp Boundaries Without Polygonal Meshing**: Captures fine anatomical membranes, skull bone, and vascular structures without polygonal discretization errors.
  - **Controlled Surface Softness**: The parameter $r$ (in voxels) directly controls the visual softness/thickness of the surface boundary, preventing harsh aliasing.
  - **Simultaneous Multi-Surface Visualization**: Can render a semi-transparent outer skin layer alongside an opaque internal skeletal structure in a single ray march.
- **Where It Is Best Applied**:
  - **Clinical Brain MRI**: Visualizing the folded cerebral cortex, gyri, and deep ventricular interfaces.
  - **Industrial CT (Engine Block)**: Rendering metallic casting walls and piston sleeves with sharp Blinn-Phong specular reflections.
  - **Cerebral CT Angiography (AVM)**: Tracing contrast-enhanced arterial feeder vessels and vascular malformation nidus.

---

### 2. Robert Drebin et al. (1988) Material Mixture Classification
- **Mathematical Principle**: Models every voxel as a continuous fractional mixture of $M$ pure materials:
  $$C(x) = \sum_{i=1}^M p_i(x) C_i, \quad \alpha(x) = \sum_{i=1}^M p_i(x) \alpha_i, \quad \rho(x) = \sum_{i=1}^M p_i(x) \rho_i$$
  Surfaces occur where physical density transitions rapidly. The surface normal $\vec{N} = \frac{\nabla \rho}{|\nabla \rho|}$ and surface strength $S = |\nabla \rho|$ modulate specular highlights and opacity:
  $$\alpha_{\text{eff}} = \min\big(1.0, \, \alpha(x) \cdot (k_{\text{ambient}} + S \cdot k_{\text{surface}})\big) \cdot M(x, y, z)$$
  where $M(x, y, z)$ is a 3D Matte Volume mask.
- **Why It Is Better**:
  - **Partial-Volume Physical Fidelity**: In tomographic imaging, boundary voxels contain physical mixtures of adjacent substances due to detector point-spread functions. Drebin's probabilistic model reflects this physical acquisition reality.
  - **Preserves Internal Volumetric Translucency**: While Levoy suppresses opacity away from isosurfaces, Drebin allows semi-transparent material interiors to remain visible.
  - **Native 3D Cutaways & Matte Masking**: Seamlessly integrates 3D wedge cuts and clipping planes to slice through outer surfaces and inspect internal structures.
- **Where It Is Best Applied**:
  - **Bonsai Tree CT**: Distinguishing between porous wood, dense trunk core, fine foliage needles, root fibers, and ceramic pot material.
  - **Multi-Tissue Anatomical Scans**: Simultaneously distinguishing air, subcutaneous fat, soft muscle, spongy trabecular bone, and cortical bone.
  - **Surgical Cutaway Planning**: Revealing deep structures through 3D wedge cutouts.

---

### 3. Maximum Intensity Projection (MIP)
- **Mathematical Principle**: For each primary viewing ray, evaluates the maximum scalar value along the ray within the volume bounding box:
  $$I(u, v) = \max_{t_{\text{enter}} \le t \le t_{\text{exit}}} f\big(\vec{O} + t \vec{D}\big)$$
- **Why It Is Better**:
  - **Zero Parameter Tuning Required**: Requires no transfer function design, no isovalue selection, no material thresholds, and no lighting parameters.
  - **Ultra-Low Latency**: Omits gradient calculations, normal vectors, and Blinn-Phong shading loops entirely, rendering at over 12+ FPS on CPU.
  - **Depth Invariance**: Reveals high-density structures regardless of where they lie along the ray or whether they are obscured by lower-density soft tissue.
- **Where It Is Best Applied**:
  - **Clinical Angiography (MRA / CTA)**: Fast clinical detection of vascular stenosis, aneurysms, and feeding vessels.
  - **Rapid Reconnaissance Overview**: Instant orientation checks when loading a new or unfamiliar volume dataset before configuring Levoy or Drebin transfer parameters.
  - **Orthopedic Assessment**: Rapid localization of high-density bone fractures and foreign bodies.

---

## 3. Active Directory Structure (Files Used by `main.py`)

Below is the clean, complete directory structure containing **only the active files and directories that `main.py` and the web server actually use**:

```
project/
├── core/                               # Core mathematical & rendering engine
│   ├── __init__.py                     # Package initialization
│   ├── volume.py                       # Volume representation, AABB bounds, unit cube normalization
│   ├── camera.py                       # Orbit camera, view basis, parallel ray generation
│   ├── gradients.py                    # 3D central differences, normal vectors, gradient magnitude
│   ├── interpolation.py                # JIT-compiled trilinear interpolation (scalar and vector fields)
│   ├── shading.py                      # Blinn-Phong illumination with two-sided normal handling
│   ├── levoy.py                        # Marc Levoy (1988) isosurface equations & multi-surface blending
│   ├── drebin.py                       # Robert Drebin (1988) material mixtures, density rho, & mattes
│   └── raycaster.py                    # Parallel ray caster (MIP, Levoy, Drebin, B2F, F2B, Cutaways)
│
├── datasets/                           # Volumetric datasets and loaders
│   ├── __init__.py                     # Package initialization
│   ├── downloader.py                   # Automated dataset catalog downloader & NIfTI byte parser
│   ├── loader.py                       # Volume loaders for NIfTI (.nii, .nii.gz), .npy, .raw, and image stacks
│   └── generator.py                    # Marschner-Lobb benchmark field generator (1994)
│
├── transfer_functions/                 # Color and opacity transfer tables
│   ├── __init__.py                     # Package initialization
│   ├── transfer_function.py            # Piecewise linear 1D LUT evaluation
│   └── presets.py                      # Curated presets (Brain, Bonsai, Engine, Shells, Marschner)
│
├── web/                                # Modern High-Performance Web Dashboard (FastAPI + HTML5)
│   ├── server.py                       # FastAPI REST API (raycast, 2D slice, provenance endpoints)
│   └── static/                         # Frontend client files
│       ├── index.html                  # Responsive dashboard layout (Hero 3D view + Provenance + Slices)
│       ├── style.css                   # Glassmorphism dark theme & responsive layout styling
│       └── app.js                      # Mouse orbit controls, progressive refinement, slice scrubber
│
├── data/                               # Local cache for real 3D volumetric datasets
│   ├── brain_t1.nii.gz                 # Clinical Human Brain T1 MRI (140 x 160 x 124)
│   ├── bonsai.nii.gz                   # Bonsai Tree CT scan (256 x 256 x 256)
│   ├── engine.nii.gz                   # Engine Block Industrial CT scan (256 x 256 x 128)
│   ├── avm.nii.gz                      # Clinical CT Cerebral Angiogram (337 x 319 x 154)
│   ├── brain_flair.nii.gz              # Clinical Human Brain FLAIR MRI (140 x 160 x 124)
│   └── fuel.nii.gz                     # Fuel Injection volumetric flow simulation (64 x 33 x 32)
│
├── reference_scans/                    # Ground truth reference renders of actual datasets
│   ├── bonsai_ref.png                  # Official OpenSciVis / Roettger benchmark reference render
│   ├── engine_ref.png                  # Canonical GE engine block reference render
│   ├── brain_t1_ref.png                # Standard clinical 3D anatomical reference render
│   ├── avm_ref.png                     # Standard clinical angiogram reference render
│   └── ct_ref.jpg                      # Official clinical CT head reference scan
│
├── tests/                              # Rigorous verification unit test suite
│   ├── __init__.py                     # Package initialization
│   ├── test_interpolation.py           # Verification of trilinear interpolation exactness
│   ├── test_gradients.py               # Verification of central differences on quadratic fields
│   ├── test_raybox.py                  # Verification of slab method ray-box intersection
│   ├── test_levoy.py                   # Verification of Levoy isosurface opacity falloff
│   ├── test_drebin.py                  # Verification of Drebin trapezoidal membership & density
│   ├── test_compositing.py             # Verification of Back-to-Front vs Front-to-Back equivalence
│   └── test_core.py                    # Test suite runner
│
├── output/                             # Destination for exported CLI renders and screenshots
├── main.py                             # Unified master entry point (Web Dashboard, CLI, Tests)
├── run_tests.py                        # Standalone unit test suite runner
├── requirements.txt                    # Python project dependencies
├── DEPRECATED_AND_UNNECESSARY.md       # Technical audit of obsolete legacy components
└── README.md                           # Master project documentation
```

*(For documentation on obsolete legacy components—such as the deprecated Tkinter GUI and static PDF figures—refer to [`DEPRECATED_AND_UNNECESSARY.md`](file:///c:/Users/tarun/BITS/4-1/CG/project/DEPRECATED_AND_UNNECESSARY.md)).*

---

## 4. File-by-File Technical Breakdown

### `core/volume.py`
- **Purpose**: Defines `Volume` and `VolumeBoundingBox`.
- **Key Methods**:
  - `normalize_to_unit_cube()`: Centers volume at $(0, 0, 0)$ and scales physical spacing so maximum spatial dimension is $2.0$ (bounds fit in $[-1, 1]^3$). Eliminates camera-inside-volume bugs.
  - `compute_gradients()`: Lazily evaluates and caches 3D central difference gradient vectors, gradient magnitudes, and unit normal vectors.

### `core/camera.py`
- **Purpose**: 3D pinhole camera with spherical orbit navigation.
- **Key Methods**:
  - `get_view_basis()`: Computes orthonormal basis $(\vec{u}, \vec{v}, \vec{w})$ from spherical coordinates (Azimuth, Elevation, Distance) centered on target.
  - `generate_camera_rays()`: JIT-compiled parallel ray generator computing primary ray origins and normalized directions for every pixel.

### `core/gradients.py`
- **Purpose**: 3D central difference gradient approximation.
- **Formula**:
  $$\nabla f(x, y, z) = \left( \frac{f(x+1) - f(x-1)}{2 \Delta x}, \frac{f(y+1) - f(y-1)}{2 \Delta y}, \frac{f(z+1) - f(z-1)}{2 \Delta z} \right)$$
- Accelerated with Numba multi-threading (`parallel=True, fastmath=True`).

### `core/interpolation.py`
- **Purpose**: High-speed continuous field sampling.
- **Functions**: `sample_trilinear_scalar` and `sample_trilinear_vec3` (`@njit(fastmath=True, inline='always')`). Clamps coordinates outside boundaries to zero.

### `core/shading.py`
- **Purpose**: Local illumination on volumetric surfaces.
- **Formula**: Blinn-Phong model:
  $$C = I_a k_a C_d + I_p \left[ k_d (\vec{N}' \cdot \vec{L}) C_d + k_s (\vec{N}' \cdot \vec{H})^n C_s \right]$$
  Includes two-sided normal flipping ($\vec{N}' \leftarrow -\vec{N}$ if $\vec{N} \cdot \vec{V} < 0$) to correctly light interior and back-facing surfaces.

### `core/levoy.py`
- **Purpose**: Implements Marc Levoy's 1988 surface display formulations.
- **Formulas**: Continuous isosurface thickness falloff $\alpha(x)$ and multi-surface blending $\alpha_{\text{total}} = 1 - \prod (1 - \alpha_m)$.

### `core/drebin.py`
- **Purpose**: Implements Robert Drebin et al.'s 1988 material mixture classification.
- **Formulas**: Trapezoidal membership percentages $p_i(x)$, composite color/opacity, effective density $\rho$, surface strength $S = |\nabla \rho|$, and 3D matte cutaway evaluation.

### `core/raycaster.py`
- **Purpose**: Unified high-performance parallel ray-casting coordinator.
- **Architecture**:
  - `_render_mip_kernel`: Blazingly fast maximum intensity projection.
  - `_render_levoy_kernel`: Specialized Levoy isosurface ray caster with front-to-back early termination and back-to-front recurrence.
  - `_render_drebin_kernel`: Specialized Drebin material mixture ray caster with surface strength modulation and 3D matte cutaways.
  - All kernels compiled with `@njit(parallel=True, fastmath=True, cache=True)` for instant startup on subsequent runs.

### `web/server.py`
- **Purpose**: Asynchronous REST backend built on FastAPI and Uvicorn.
- **Endpoints**:
  - `GET /`: Serves the modern HTML5 dashboard.
  - `GET /api/datasets`: Returns active datasets with complete scientific provenance metadata.
  - `POST /api/load_dataset`: Normalizes and pre-loads a dataset into GPU/CPU memory.
  - `POST /api/render`: Executes ray casting with camera, algorithm, and lighting parameters, returning base64 PNG.
  - `GET /api/slice`: Extracts 2D orthogonal slices (Axial, Coronal, Sagittal) with interactive scrubbing.

### `web/static/` (`index.html`, `style.css`, `app.js`)
- **Purpose**: Client-side interactive dashboard.
- **Capabilities**:
  - Hero 3D raycast viewport with mouse orbit controls (Left drag: orbit, Right drag/wheel: zoom).
  - Progressive resolution refinement (fast preview during drag, crisp $320 \times 320$ on release).
  - Dynamic dataset provenance panel updating metadata on the fly.
  - Full-width 2D orthogonal slice viewer with live slider scrubbing.

---

## 5. How to Run the Codebase

### 1. Launch the Interactive Web Dashboard (Default)
```powershell
python main.py
```
*Automatically launches the local FastAPI server and opens your web browser at `http://127.0.0.1:8000`.*

To specify a custom port or prevent auto-opening the browser:
```powershell
python main.py --port 8080 --no-browser
```

### 2. Command-Line Interface (CLI) Rendering
Render any dataset directly to an image file:
```powershell
# Render clinical Brain MRI using Levoy isosurface display:
python main.py --cli --dataset brain_mri --method levoy --output output/brain_levoy.png

# Render Bonsai tree CT using Drebin material classification:
python main.py --cli --dataset bonsai --method drebin --output output/bonsai_drebin.png

# Render Industrial Engine Block with a 3D Wedge Cutaway:
python main.py --cli --dataset engine --method levoy --wedge --output output/engine_cutaway.png

# Render Brain MRI using Maximum Intensity Projection (MIP):
python main.py --cli --dataset brain_mri --method mip --output output/brain_mip.png
```

### 3. Download & Cache Real Datasets
```powershell
# List available real datasets in the catalog:
python main.py --list-datasets

# Download a specific dataset:
python main.py --download bonsai
python main.py --download engine
python main.py --download all
```

### 4. Run the Unit Test Suite
```powershell
python run_tests.py
```
*(Runs all 6 mathematical verification suites; verifies trilinear interpolation exactness, central differences, slab intersection, Levoy falloff, Drebin trapezoids, and compositing equivalence).*

---

## 6. Verification and Performance Summary

All 6 unit test suites pass with zero errors:
```
============================================================
RUNNING IS F311 COMPUTER GRAPHICS VOLUME RENDERING TEST SUITE
============================================================

[1/6] Testing Trilinear Interpolation...                               -> PASSED
[2/6] Testing Central Difference Gradients & Normal Vectors...         -> PASSED
[3/6] Testing Slab Method Ray-Box Intersection...                      -> PASSED
[4/6] Testing Marc Levoy (1988) Isosurface & Multi-Surface Blending... -> PASSED
[5/6] Testing Robert Drebin (1988) Material Classification & Mattes... -> PASSED
[6/6] Testing Back-to-Front vs Front-to-Back Compositing Equivalence.. -> PASSED

============================================================
ALL 6 TEST SUITES PASSED SUCCESSFULLY!
============================================================
```

### Execution Speed:
- **Maximum Intensity Projection (MIP)**: ~80 ms per frame (>12 FPS).
- **Levoy Isosurface Ray Casting ($200 \times 200$)**: ~700 ms per frame.
- **Drebin Material Mixture ($200 \times 200$, 16.7M voxels)**: ~1.07 seconds per frame.
- **Interactive Drag Preview Mode**: ~30 ms response time for smooth 30+ FPS mouse orbit navigation.
- **Early Ray Termination Speedup**: $1.8\times$ to $2.5\times$ speedup over full volume integration.
