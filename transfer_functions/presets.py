"""
presets.py - Preset transfer function and material configurations.
Provides curated presets for both Levoy (1988) and Drebin et al. (1988).
"""

import numpy as np
from core.levoy import LevoySurface, LevoyRendererConfig
from core.drebin import DrebinMaterial, DrebinClassifier


def get_levoy_preset(preset_name: str) -> LevoyRendererConfig:
    """Returns a configured LevoyRendererConfig based on preset name."""
    preset = preset_name.lower().strip()

    if "head_bone_skin" in preset or "default" in preset:
        # Dual surface: Semi-transparent skin + Opaque skull bone
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.28, thickness=1.5, alpha_v=0.30, color=np.array([0.92, 0.76, 0.65], dtype=np.float32)), # Skin
            LevoySurface(isovalue=0.72, thickness=2.0, alpha_v=0.92, color=np.array([0.96, 0.94, 0.88], dtype=np.float32)), # Bone
        ])

    elif "bone_only" in preset or "bone" in preset:
        # Isolated cortical bone and teeth
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.75, thickness=1.8, alpha_v=0.95, color=np.array([0.98, 0.95, 0.90], dtype=np.float32)),
        ])

    elif "skin_only" in preset or "skin" in preset:
        # Outer surface morphology
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.25, thickness=1.2, alpha_v=0.85, color=np.array([0.90, 0.70, 0.58], dtype=np.float32)),
        ])

    elif "brain" in preset or "mri" in preset or "soft_tissue" in preset:
        # Brain MRI: White matter surface + Grey matter surface
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.30, thickness=1.4, alpha_v=0.35, color=np.array([0.75, 0.60, 0.55], dtype=np.float32)), # Grey matter
            LevoySurface(isovalue=0.60, thickness=1.8, alpha_v=0.90, color=np.array([0.95, 0.92, 0.85], dtype=np.float32)), # White matter
        ])

    elif "bonsai" in preset:
        # Bonsai tree: Leaves / Foliage + Trunk / Branches
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.20, thickness=1.5, alpha_v=0.45, color=np.array([0.25, 0.75, 0.30], dtype=np.float32)), # Green foliage
            LevoySurface(isovalue=0.55, thickness=2.0, alpha_v=0.95, color=np.array([0.65, 0.40, 0.20], dtype=np.float32)), # Wood trunk
        ])

    elif "engine" in preset:
        # Engine block: Outer chassis + Internal cylinders
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.30, thickness=1.5, alpha_v=0.35, color=np.array([0.70, 0.75, 0.85], dtype=np.float32)), # Outer cast
            LevoySurface(isovalue=0.68, thickness=2.0, alpha_v=0.95, color=np.array([0.95, 0.85, 0.40], dtype=np.float32)), # Heavy metal core
        ])

    elif "marschner" in preset or "lobb" in preset:
        # Classical Marschner-Lobb benchmark surface at f_v = 0.5
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.50, thickness=1.4, alpha_v=0.95, color=np.array([0.85, 0.85, 0.95], dtype=np.float32)),
        ])

    elif "shells" in preset:
        # Concentric shells
        return LevoyRendererConfig(surfaces=[
            LevoySurface(isovalue=0.25, thickness=1.2, alpha_v=0.25, color=np.array([0.3, 0.6, 0.9], dtype=np.float32)), # Outer blue
            LevoySurface(isovalue=0.50, thickness=1.5, alpha_v=0.45, color=np.array([0.9, 0.7, 0.2], dtype=np.float32)), # Middle gold
            LevoySurface(isovalue=0.85, thickness=1.8, alpha_v=0.95, color=np.array([0.9, 0.2, 0.2], dtype=np.float32)), # Inner red
        ])

    else:
        return LevoyRendererConfig()


def get_drebin_preset(preset_name: str) -> DrebinClassifier:
    """Returns a configured DrebinClassifier based on preset name."""
    preset = preset_name.lower().strip()

    if "head" in preset or "default" in preset:
        # CT Head materials: Air, Skin/Soft Tissue, Brain, Bone, Teeth
        return DrebinClassifier(materials=[
            DrebinMaterial("Air", -0.10, -0.05, 0.08, 0.16, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.00, 0.0),
            DrebinMaterial("Skin/Tissue", 0.18, 0.26, 0.36, 0.44, np.array([0.90, 0.65, 0.55], dtype=np.float32), 0.28, 1.0),
            DrebinMaterial("Brain Matter", 0.40, 0.45, 0.52, 0.58, np.array([0.80, 0.40, 0.40], dtype=np.float32), 0.35, 1.4),
            DrebinMaterial("Cranial Bone", 0.65, 0.72, 0.86, 0.92, np.array([0.96, 0.94, 0.88], dtype=np.float32), 0.92, 3.0),
            DrebinMaterial("Teeth/Enamel", 0.88, 0.94, 1.05, 1.10, np.array([1.00, 1.00, 0.95], dtype=np.float32), 0.98, 4.0),
        ])

    elif "brain" in preset or "mri" in preset:
        # Brain MRI: Background/Air, CSF, Grey Matter, White Matter
        return DrebinClassifier(materials=[
            DrebinMaterial("Background", -0.10, 0.0, 0.08, 0.14, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0),
            DrebinMaterial("CSF / Ventricles", 0.12, 0.18, 0.24, 0.30, np.array([0.3, 0.5, 0.8], dtype=np.float32), 0.20, 1.0),
            DrebinMaterial("Grey Matter", 0.28, 0.35, 0.48, 0.56, np.array([0.85, 0.50, 0.45], dtype=np.float32), 0.40, 1.8),
            DrebinMaterial("White Matter", 0.54, 0.62, 0.90, 0.98, np.array([0.98, 0.95, 0.88], dtype=np.float32), 0.90, 3.0),
        ])

    elif "bonsai" in preset:
        # Bonsai tree: Foliage (green), Trunk (brown), Pot (clay)
        return DrebinClassifier(materials=[
            DrebinMaterial("Background", -0.10, 0.0, 0.08, 0.12, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0),
            DrebinMaterial("Foliage", 0.12, 0.20, 0.35, 0.45, np.array([0.2, 0.8, 0.25], dtype=np.float32), 0.30, 1.2),
            DrebinMaterial("Trunk/Branches", 0.42, 0.52, 0.70, 0.78, np.array([0.65, 0.40, 0.20], dtype=np.float32), 0.85, 2.5),
            DrebinMaterial("Pot", 0.75, 0.82, 0.98, 1.05, np.array([0.80, 0.45, 0.35], dtype=np.float32), 0.95, 3.5),
        ])

    elif "engine" in preset:
        return DrebinClassifier(materials=[
            DrebinMaterial("Air", -0.10, 0.0, 0.15, 0.20, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0),
            DrebinMaterial("Outer Casing", 0.22, 0.32, 0.55, 0.65, np.array([0.5, 0.7, 0.9], dtype=np.float32), 0.40, 1.5),
            DrebinMaterial("Cylinders/Pistons", 0.62, 0.75, 0.98, 1.05, np.array([0.98, 0.85, 0.30], dtype=np.float32), 0.95, 3.5),
        ])

    elif "bone_only" in preset or "bone" in preset:
        return DrebinClassifier(materials=[
            DrebinMaterial("Background", -0.10, 0.0, 0.60, 0.65, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0),
            DrebinMaterial("Bone", 0.68, 0.75, 1.05, 1.10, np.array([0.96, 0.94, 0.88], dtype=np.float32), 0.95, 3.0),
        ])

    elif "shells" in preset:
        return DrebinClassifier(materials=[
            DrebinMaterial("Outer Shell", 0.20, 0.24, 0.28, 0.35, np.array([0.2, 0.6, 0.9], dtype=np.float32), 0.20, 1.0),
            DrebinMaterial("Middle Shell", 0.45, 0.49, 0.53, 0.60, np.array([0.9, 0.8, 0.2], dtype=np.float32), 0.45, 2.0),
            DrebinMaterial("Inner Core", 0.78, 0.83, 1.00, 1.05, np.array([0.9, 0.2, 0.2], dtype=np.float32), 0.95, 4.0),
        ])

    elif "marschner" in preset or "lobb" in preset:
        return DrebinClassifier(materials=[
            DrebinMaterial("Exterior", -0.1, 0.0, 0.45, 0.48, np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0),
            DrebinMaterial("Surface", 0.48, 0.50, 0.52, 0.55, np.array([0.8, 0.85, 0.95], dtype=np.float32), 0.90, 2.0),
            DrebinMaterial("Interior", 0.54, 0.58, 1.00, 1.05, np.array([0.5, 0.6, 0.8], dtype=np.float32), 0.95, 3.0),
        ])

    else:
        return DrebinClassifier()
