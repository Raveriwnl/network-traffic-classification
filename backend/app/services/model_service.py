from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np

from backend.app.core.config import PROJECT_ROOT, Settings


class ModelService:
    def __init__(self, settings: Settings, class_names: list[str]) -> None:
        self.settings = settings
        self.class_names = class_names
        self._torch = None
        self._model = None
        self._device = "cpu"

    def _ensure_project_root(self) -> None:
        project_root = str(PROJECT_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

    def _extract_state_dict(self, raw: object) -> dict[str, object]:
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
        elif isinstance(raw, dict):
            state_dict = raw
        else:
            raise RuntimeError("Unsupported model checkpoint format.")
        normalized: dict[str, object] = {}
        for key, value in state_dict.items():
            normalized[str(key).removeprefix("module.")] = value
        return normalized

    def _detect_num_mamba_layers(self, state_dict: dict[str, object]) -> int:
        indices: set[int] = set()
        for key in state_dict:
            if not key.startswith("mamba_blocks."):
                continue
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.add(int(parts[1]))
        return max(indices) + 1 if indices else 1

    def load(self) -> None:
        if self._model is not None:
            return
        self._ensure_project_root()
        torch = importlib.import_module("torch")
        module = importlib.import_module("training.cnn_bimobilemamba_attention")
        checkpoint = torch.load(self.settings.model_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_state_dict(checkpoint)
        num_layers = self._detect_num_mamba_layers(state_dict)
        model = module.build_cnn_bimobilemamba_attention_model(
            input_shape=(self.settings.bins, self.settings.feature_count),
            num_classes=len(self.class_names),
            num_mamba_layers=num_layers,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self._torch = torch
        self._model = model

    @property
    def device(self) -> str:
        return self._device

    def infer(self, tensor: np.ndarray) -> dict[str, object]:
        self.load()
        assert self._torch is not None
        assert self._model is not None
        torch = self._torch
        sample = torch.from_numpy(tensor.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            started = time.perf_counter()
            logits = self._model(sample)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            latency_ms = (time.perf_counter() - started) * 1000.0
        distribution = {name: round(float(prob), 4) for name, prob in zip(self.class_names, probs.tolist())}
        predicted_index = int(np.argmax(probs))
        predicted_class = self.class_names[predicted_index]
        confidence = round(float(probs[predicted_index]), 4)
        if confidence < self.settings.other_traffic_threshold:
            predicted_index = -1
            predicted_class = "other"
        return {
            "class_id": predicted_index,
            "class_name": predicted_class,
            "confidence": confidence,
            "distribution": distribution,
            "inference_latency_ms": round(latency_ms, 3),
            "device": self._device,
        }