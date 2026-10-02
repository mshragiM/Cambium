"""Configuration models for Cambium."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class HLSBackendConfig:
    """Backend configuration for HLS tools (Vivado/Vitis)."""

    VIVADO_HLS = "vivado_hls"
    VITIS_HLS = "vitis_hls"

    backend: str = VIVADO_HLS

    def __post_init__(self) -> None:
        if self.backend not in {self.VIVADO_HLS, self.VITIS_HLS}:
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'vivado_hls' or 'vitis_hls'")
        self.config = self._get_backend_config()

    def _get_backend_config(self) -> Dict:
        configs = {
            self.VIVADO_HLS: {
                "version": "2020.1",
                "tool_command": "vivado_hls",
                "project_upgrade": False,
                "flow_target": None,
                "requires_const_rom": False,
                "default_pipeline_loops": None,
                "export_format": "ip_catalog",
                "cosim_trace": "port",
            },
            self.VITIS_HLS: {
                "version": "2023.1",
                "tool_command": "vitis_hls",
                "project_upgrade": True,
                "flow_target": "vitis",
                "requires_const_rom": True,
                "default_pipeline_loops": 0,
                "export_format": "xo",
                "cosim_trace": "all",
                "axi_bundle": "gmem",
                "axi_latency": 64,
            },
        }
        return configs[self.backend]

    def get_define_macro(self) -> str:
        if self.backend == self.VITIS_HLS:
            return "__VITIS_HLS__"
        return "__VIVADO_HLS__"

    def is_vitis(self) -> bool:
        return self.backend == self.VITIS_HLS


class CambiumConfig:
    """User/project configuration container."""

    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_default_config()
        self.backend = HLSBackendConfig()
        if config_file:
            self.load_config(config_file)

    def set_backend(self, backend: str) -> None:
        self.backend = HLSBackendConfig(backend)
        self.config["hls"]["backend"] = backend

    def _load_default_config(self) -> Dict:
        return {
            "project": {
                "name": "cambium_project",
                "output_dir": "cambium_output",
                "target": "hls",
            },
            "model": {
                "type": "random_forest",
                "task": "regression",
                "n_estimators": 40,
                "max_depth": 6,
                "max_leaf_nodes": None,
                "random_state": 42,
            },
            "data": {
                "test_size": 0.2,
                "scaler": "minmax",
                "feature_cols": [],
                "target_cols": [],
            },
            "export": {
                "precision": "fixed",
                "max_nodes": 128,
                "optimization": "speed",
                "format": "default",
                "implementation": "auto",
            },
            "hls": {
                "backend": "vivado_hls",
                "clock_period": "5ns",
                "target_device": "xc7z020clg400-1",
                "board_part": "www.digilentinc.com:pynq-z1:part0:1.0",
                "pipeline_ii": 1,
                "unroll_loops": True,
                "vitis_flow_target": "hw",
                "axi_bundle_name": "gmem",
                "axi_latency": 64,
            },
        }

    def load_config(self, config_file: str) -> None:
        path = Path(config_file)
        with path.open("r", encoding="utf-8") as file:
            if path.suffix.lower() in {".yaml", ".yml"}:
                user_config = yaml.safe_load(file)
            else:
                user_config = json.load(file)
        self._deep_merge(self.config, user_config or {})
        self.set_backend(self.config["hls"].get("backend", "vivado_hls"))

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save_config(self, output_file: str) -> None:
        with Path(output_file).open("w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
