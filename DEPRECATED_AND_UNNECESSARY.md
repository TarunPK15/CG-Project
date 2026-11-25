# Deprecated and Unnecessary Components

This document details all files, modules, and legacy scripts that are **no longer needed or used** following the complete architectural overhaul to the modern FastAPI/HTML5 web application and the focus on real clinical/scientific datasets.

---

## 1. Summary of Deprecated Directories & Files

| Directory / File | Status | Reason for Deprecation | Modern Replacement |
|---|---|---|---|
| `gui/` (`gui/app.py`, `gui/__init__.py`) | **Obsolete** | Legacy Tkinter desktop interface. Tkinter ran single-threaded, choked event queues during mouse drag, froze the UI, and blitted raw software bitmaps with severe latency. | `web/server.py` and `web/static/` (FastAPI + Uvicorn + HTML5 Canvas web dashboard with progressive refinement and 30+ FPS responsiveness). |
| `paper_reference_figures/` (all 16 images) | **Obsolete** | Static extracted figures from the 1988 papers (PDF screenshots). The user explicitly redirected away from paper snapshots in favor of official ground-truth reference renders of the *exact datasets being rendered*. | `reference_scans/` (`bonsai_ref.png`, `engine_ref.png`, `brain_t1_ref.png`, `avm_ref.png`, `ct_ref.jpg`). |
| `examples/` (`render_comparisons.py`, `benchmark.py`, `interactive_viewer.py`, `view_slices.py`) | **Obsolete** | Standalone Tkinter/Matplotlib demonstration scripts. Everything they performed is now natively available within the web dashboard and master CLI. | `main.py --cli`, `main.py --benchmark`, `web/server.py` (`/api/slice`, `/api/render`). |
| `datasets/generator.py` (`create_ct_head_phantom`, `create_concentric_shells`, `create_torso_phantom`) | **Unused by Main Dashboard** | Synthetic procedural phantoms generated using discrete step functions, causing voxel aliasing. The user requested removing the synthetic head phantom entirely in favor of real clinical scans. Only `create_marschner_lobb` is retained for filter quality benchmarking. | Real datasets in `data/` (`brain_t1.nii.gz`, `bonsai.nii.gz`, `engine.nii.gz`, `avm.nii.gz`, `brain_flair.nii.gz`). |
| `drebin.pdf`, `levoy.pdf` | **Reference Only** | The original 1988 research papers. Useful for theoretical study, but not executed by runtime code. | Core formulations in `core/levoy.py` and `core/drebin.py`. |
| `proposal.docx` | **Historical Artifact** | Initial course project proposal document. | `README.md`. |

---

## 2. Detailed Technical Rationale

### A. Why Tkinter Was Deprecated (`gui/app.py`)
1. **Single-Threaded Event Loop Freezes**: Tkinter's `root.mainloop()` shares the main thread with rendering callbacks. High-resolution volume ray casting locked up the operating system window, causing Windows to mark the application as "Not Responding".
2. **Software Blitting Bottleneck**: Tkinter's `ImageTk.PhotoImage` performs CPU-bound memory copies of full RGBA framebuffers into Windows GDI bitmaps on every mouse movement, destroying interactivity.
3. **Replaced By**: A lightweight local web architecture. Python runs as an asynchronous REST server (`FastAPI` + `uvicorn`), while the browser handles smooth mouse input, HTML5 Canvas rendering, and layout styling via modern CSS.

### B. Why Paper Snapshots Were Deprecated (`paper_reference_figures/`)
1. **Misaligned Comparison**: The extracted 1988 PDF screenshots depicted historical 1980s CT scanners from Chapel Hill and Pixar on unspecified deceased specimens. Comparing our engine's render of a modern clinical brain MRI against a 1988 paper screenshot was scientifically invalid.
2. **Replaced By**: Verified, canonical benchmark reference renders of the *exact same datasets* (the Stefan Röttger Bonsai tree render, the General Electric engine block CT render, and standard clinical MRIcroGL brain surfaces) located in `reference_scans/`.

### C. Why Procedural Phantoms Were Removed from the Dashboard
1. **Voxel Aliasing & Staircasing**: Synthetic phantoms constructed from geometric equations (`x^2 + y^2 < r^2`) exhibit sharp step transitions unless heavily oversampled. Real tomographic scans naturally contain continuous partial-volume transitions from physical beam attenuation.
2. **Replaced By**: Clinical Human Brain MRI (T1-weighted and FLAIR), High-Resolution Bonsai Tree Micro-CT (16.7M voxels), Industrial Engine Block CT (8.3M voxels), and Clinical CT Angiogram (AVM).

---

## 3. Workspace Cleanup Status

The deprecated components (`gui/`, `examples/`, and `paper_reference_figures/`) along with obsolete procedural phantom generators have been permanently purged from the workspace.

The active codebase in `main.py`, `core/`, `web/`, `data/`, `datasets/`, and `tests/` has zero dependencies on those components, operates completely cleanly, and all unit tests pass with 100% success rate.
