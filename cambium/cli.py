#!/usr/bin/env python3
"""Cambium command line interface."""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

from .framework import CambiumFramework, create_sample_config
from .logos import get_logo

logger = logging.getLogger(__name__)


def _parse_csv_columns(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


class CambiumCLI:
    def __init__(self):
        self.parser = self._create_parser()

    @staticmethod
    def _normalize_precision(value: str) -> str:
        """Convert user-friendly precision input into config format."""
        if value is None:
            return value

        normalized = value.strip()
        if not normalized:
            raise ValueError("Precision cannot be empty.")

        lowered = normalized.lower()
        if lowered in {"float", "fixed"}:
            return lowered

        if lowered.startswith("ap_fixed<") and lowered.endswith(">"):
            match = re.fullmatch(r"ap_fixed<\s*(\d+)\s*,\s*(\d+)\s*>", lowered)
            if not match:
                raise ValueError(
                    "Invalid ap_fixed format. Use ap_fixed<total_bits,int_bits>, e.g. ap_fixed<18,8>."
                )
            total_bits = int(match.group(1))
            int_bits = int(match.group(2))
        else:
            match = re.fullmatch(r"(\d+)\.(\d+)", normalized)
            if not match:
                raise ValueError(
                    "Invalid precision format. Use float, fixed, ap_fixed<18,8>, or shorthand like 18.8."
                )
            total_bits = int(match.group(1))
            int_bits = int(match.group(2))

        if total_bits <= 0 or int_bits <= 0:
            raise ValueError("Precision bit widths must be positive.")
        if int_bits > total_bits:
            raise ValueError("Integer bits cannot be greater than total bits.")

        return f"ap_fixed<{total_bits},{int_bits}>"

    def _apply_precision_override(self, framework: CambiumFramework, precision: Optional[str]) -> None:
        if not precision:
            return

        normalized = self._normalize_precision(precision)
        framework.config.config["export"]["precision"] = normalized
        logger.info("Using precision: %s", normalized)

    def _create_parser(self) -> argparse.ArgumentParser:
        parent_parser = argparse.ArgumentParser(add_help=False)
        verbosity_group = parent_parser.add_mutually_exclusive_group()
        verbosity_group.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
        verbosity_group.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

        parser = argparse.ArgumentParser(
            description="Cambium - Generate optimized ML inference code",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[parent_parser],
            epilog="""
Examples:

# Vivado HLS (default)
cambium quick-start --data data.csv --features f1,f2 --targets t1 --output myproject

# Vitis HLS
cambium quick-start --data data.csv --features f1,f2 --targets t1 --backend vitis_hls --output myproject

# With config file
cambium quick-start --data data.csv --config cambium_config_vitis.yaml
            """,
        )

        parser.add_argument("--version", action="version", version="Cambium 1.1")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        self._add_train_parser(subparsers, parent_parser)
        self._add_export_parser(subparsers, parent_parser)
        self._add_quickstart_parser(subparsers, parent_parser)
        self._add_config_parser(subparsers, parent_parser)
        self._add_synthesis_parser(subparsers, parent_parser)
        return parser

    def _add_train_parser(self, subparsers, parent_parser) -> None:
        train_parser = subparsers.add_parser("train", help="Train a machine learning model", parents=[parent_parser])
        data_group = train_parser.add_argument_group("Data Options")
        data_group.add_argument("--data", "-d", required=True, help="Path to dataset file or URL")
        data_group.add_argument("--features", "-f", help="Comma-separated list of feature columns")
        data_group.add_argument("--targets", "-t", help="Comma-separated list of target columns")
        data_group.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")

        model_group = train_parser.add_argument_group("Model Options")
        model_group.add_argument("--n-estimators", type=int, default=20, help="Number of estimators")
        model_group.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
        model_group.add_argument("--max-leaf-nodes", type=int, help="Maximum leaf nodes per trained tree")
        model_group.add_argument(
            "--task",
            choices=["regression", "classification"],
            help="Learning task type (regression or classification)",
        )
        model_group.add_argument(
            "--precision",
            help="Precision override: float, fixed, ap_fixed<18,8>, or shorthand like 18.8",
        )

        train_parser.add_argument("--config", "-c", help="Configuration file (YAML/JSON)")
        train_parser.add_argument("--output", "-o", default="cambium_output", help="Output directory")
        train_parser.add_argument("--save-model", help="Save trained model to file")
        train_parser.add_argument("--backend", choices=["vivado_hls", "vitis_hls"], default="vivado_hls", help="HLS synthesis backend")

    def _add_export_parser(self, subparsers, parent_parser) -> None:
        export_parser = subparsers.add_parser("export", help="Export trained model", parents=[parent_parser])
        export_parser.add_argument("--model", "-m", required=True, help="Path to trained model file")
        export_parser.add_argument("--config", "-c", help="Saved Cambium YAML/JSON config (defaults to model directory)")
        export_parser.add_argument("--data", "-d", help="Dataset used for training, when saved test vectors are unavailable")
        export_parser.add_argument("--features", "-f", help="Comma-separated feature columns")
        export_parser.add_argument("--targets", "-t", help="Comma-separated target columns")
        export_parser.add_argument("--task", choices=["regression", "classification"], help="Model task when no saved config exists")
        export_parser.add_argument("--target", choices=["hls"], default="hls", help="Export target format")
        export_parser.add_argument("--output", "-o", default="cambium_export", help="Output directory")
        export_parser.add_argument("--export-format", choices=["default", "legacy"], default="legacy", help="Export format")
        export_parser.add_argument("--jinja2", action="store_true", help="Use Jinja2 templates")
        export_parser.add_argument(
            "--precision",
            help="Precision override: float, fixed, ap_fixed<18,8>, or shorthand like 18.8",
        )
        export_parser.add_argument("--backend", choices=["vivado_hls", "vitis_hls"], help="HLS synthesis backend")
        export_parser.add_argument("--max-nodes", type=int, help="Minimum HLS node-array capacity per tree")
        export_parser.add_argument("--implementation", choices=["auto", "struct", "integer_axi", "float_axi"],
                                   help="HLS datapath (auto selects primitive ROM AXI for Vitis models)")
        export_parser.add_argument("--vitis-flow", choices=["hw", "hw_emu"], default="hw", help="Vitis flow target")

    def _add_quickstart_parser(self, subparsers, parent_parser) -> None:
        quick_parser = subparsers.add_parser("quick-start", help="Quick start with automatic configuration", parents=[parent_parser])
        quick_parser.add_argument("--data", "-d", help="Path to dataset file or URL (optional if provided in --config)")
        quick_parser.add_argument("--output", "-o", default="cambium_quickstart", help="Output directory")
        quick_parser.add_argument("--features", "-f", help="Comma-separated list of feature columns")
        quick_parser.add_argument("--targets", "-t", help="Comma-separated list of target columns")
        quick_parser.add_argument("--export-format", choices=["default", "legacy"], default="legacy", help="Export format")
        quick_parser.add_argument("--jinja2", action="store_true", help="Use Jinja2 templates")
        quick_parser.add_argument("--n-estimators", type=int, default=None, help="Number of estimators (overrides config when set)")
        quick_parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth (overrides config when set)")
        quick_parser.add_argument("--max-leaf-nodes", type=int, help="Maximum leaf nodes per trained tree")
        quick_parser.add_argument("--max-nodes", type=int, help="Minimum HLS node-array capacity per tree")
        quick_parser.add_argument(
            "--task",
            choices=["regression", "classification"],
            help="Learning task type (regression or classification)",
        )
        quick_parser.add_argument(
            "--precision",
            help="Precision override: float, fixed, ap_fixed<18,8>, or shorthand like 18.8",
        )
        quick_parser.add_argument("--config", "-c", help="Configuration file (YAML/JSON)")
        quick_parser.add_argument("--backend", choices=["vivado_hls", "vitis_hls"], default="vivado_hls", help="HLS synthesis backend")
        quick_parser.add_argument("--implementation", choices=["auto", "struct", "integer_axi", "float_axi"],
                                  help="HLS datapath (auto selects primitive ROM AXI for Vitis models)")
        quick_parser.add_argument("--vitis-flow", choices=["hw", "hw_emu"], default="hw", help="Vitis flow target")
        quick_parser.add_argument("--fpga-part", type=str, help="FPGA part number override")
        quick_parser.add_argument("--clock-period", type=float, help="Clock period in ns")

    def _add_config_parser(self, subparsers, parent_parser) -> None:
        config_parser = subparsers.add_parser("create-config", help="Create configuration file", parents=[parent_parser])
        config_parser.add_argument("--template", choices=["default", "energy", "automotive"], default="default")
        config_parser.add_argument("--output", "-o", default="cambium_config.yaml", help="Output config file")
        config_parser.add_argument("--backend", choices=["vivado_hls", "vitis_hls"], default="vivado_hls", help="Target HLS backend")

    def _add_synthesis_parser(self, subparsers, parent_parser) -> None:
        synth_parser = subparsers.add_parser("hls-report", help="Print Vivado/Vitis HLS synthesis summary", parents=[parent_parser])
        synth_parser.add_argument("--project-dir", "-p", default="cambium_output", help="HLS project directory")

    def run(self, args=None) -> int:
        args = self.parser.parse_args(args)
        self._configure_logging(args)

        handlers = {
            "train": self._handle_train,
            "export": self._handle_export,
            "quick-start": self._handle_quickstart,
            "create-config": self._handle_create_config,
            "hls-report": self._handle_synthesis_report,
        }

        if args.command not in handlers:
            self.parser.print_help()
            return 1

        try:
            return handlers[args.command](args)
        except KeyboardInterrupt:
            logger.error("Operation interrupted by user")
            return 1
        except Exception as exc:
            logger.error("Command failed: %s", exc)
            if getattr(args, "verbose", False):
                logger.debug("Full traceback:", exc_info=True)
            return 1

    def _configure_logging(self, args) -> None:
        is_quiet = getattr(args, "quiet", False)
        is_verbose = getattr(args, "verbose", False)
        log_level = logging.ERROR if is_quiet else logging.DEBUG if is_verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

    def _build_framework(self, args) -> CambiumFramework:
        if args.config:
            framework = CambiumFramework(args.config)
            logger.info("Using configuration from: %s", args.config)
            if getattr(args, "command", None) == "export" and getattr(args, "backend", None):
                framework.config.set_backend(args.backend)
        else:
            framework = CambiumFramework()
            framework.config.config["project"]["output_dir"] = args.output
            framework.config.config["export"]["format"] = getattr(args, "export_format", "legacy")

            if getattr(args, "n_estimators", None) is not None:
                framework.config.config["model"]["n_estimators"] = args.n_estimators
            if getattr(args, "max_depth", None) is not None:
                framework.config.config["model"]["max_depth"] = args.max_depth
            if getattr(args, "max_leaf_nodes", None) is not None:
                framework.config.config["model"]["max_leaf_nodes"] = args.max_leaf_nodes
            if getattr(args, "max_nodes", None) is not None:
                framework.config.config["export"]["max_nodes"] = args.max_nodes
            if getattr(args, "task", None):
                framework.config.config["model"]["task"] = args.task
            if hasattr(args, "test_size"):
                framework.config.config["data"]["test_size"] = args.test_size

            backend = getattr(args, "backend", None) or "vivado_hls"
            framework.config.set_backend(backend)
            if backend == "vitis_hls":
                framework.config.config["hls"]["vitis_flow_target"] = getattr(args, "vitis_flow", "hw")
                if getattr(args, "fpga_part", None):
                    framework.config.config["export"]["fpga_part"] = args.fpga_part
                    framework.config.config["hls"]["target_device"] = args.fpga_part
                if getattr(args, "clock_period", None):
                    framework.config.config["export"]["clock_period"] = args.clock_period
                    framework.config.config["hls"]["clock_period"] = f"{args.clock_period}ns"

        self._apply_precision_override(framework, getattr(args, "precision", None))
        if getattr(args, "max_nodes", None) is not None:
            framework.config.config["export"]["max_nodes"] = args.max_nodes
        if getattr(args, "max_leaf_nodes", None) is not None:
            framework.config.config["model"]["max_leaf_nodes"] = args.max_leaf_nodes
        if getattr(args, "implementation", None):
            framework.config.config["export"]["implementation"] = args.implementation
        return framework

    def _handle_train(self, args) -> int:
        logger.info("Cambium Training")
        logger.info("=" * 40)
        framework = self._build_framework(args)

        feature_cols = _parse_csv_columns(args.features)
        target_cols = _parse_csv_columns(args.targets)
        logger.info("Loading data from: %s", args.data)
        framework.load_data(args.data, feature_cols, target_cols)

        logger.info("Training model...")
        metrics = framework.train()
        logger.info("\nTraining Results:")
        for key, value in metrics.items():
            logger.info("%s: %.4f", key, value) if isinstance(value, float) else logger.info("%s: %s", key, value)

        if args.save_model:
            with Path(args.save_model).open("wb") as file:
                pickle.dump(framework.model, file)
            logger.info("\nModel saved to: %s", args.save_model)

        framework.generate_report()
        logger.info("\nTraining completed! Check results in: %s", framework.config.config["project"]["output_dir"])
        return 0

    def _handle_export(self, args) -> int:
        logger.info("Cambium Export")
        logger.info("=" * 40)
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error("Error: Model file not found: %s", args.model)
            return 1

        logger.info("Loading model from: %s", args.model)
        with model_path.open("rb") as file:
            model = pickle.load(file)

        if not args.config:
            sibling_config = model_path.parent / "cambium_config.yaml"
            if sibling_config.exists():
                args.config = str(sibling_config)
        framework = self._build_framework(args)
        framework.model = model
        if args.task:
            framework.config.config["model"]["task"] = args.task
        elif not args.config:
            framework.config.config["model"]["task"] = (
                "classification" if hasattr(model, "classes_") else "regression"
            )

        saved_x = model_path.parent / "X_test.npy"
        if saved_x.exists() and args.config and not args.data:
            framework.data_manager.feature_cols = list(framework.config.config["data"].get("feature_cols") or [])
            framework.data_manager.target_cols = list(framework.config.config["data"].get("target_cols") or [])
            if not framework.data_manager.feature_cols or not framework.data_manager.target_cols:
                raise ValueError("Saved config lacks feature or target columns; provide --data, --features, and --targets")
            framework._split_cache = (None, np.load(saved_x), None, None)
            for name in ("scaler_x", "scaler_y"):
                path = model_path.parent / f"{name}.pkl"
                if path.exists():
                    with path.open("rb") as file:
                        setattr(framework.data_manager, name, pickle.load(file))
        else:
            source = args.data or framework.config.config["data"].get("source")
            if not source:
                raise ValueError("Export requires saved X_test.npy or --data")
            features = _parse_csv_columns(args.features) or framework.config.config["data"].get("feature_cols")
            targets = _parse_csv_columns(args.targets) or framework.config.config["data"].get("target_cols")
            framework.load_data(source, features, targets)
            framework._split_cache = framework.data_manager.prepare_data()
        logger.info("Exporting to HLS (%s)...", framework.config.backend.backend)
        framework.export_to_hls_j2(args.output)

        logger.info("\nHLS Export completed!")
        logger.info("Generated files in: %s", args.output)
        if framework.config.backend.backend == "vitis_hls":
            logger.info("Run: cd %s && vitis_hls -f *.tcl", args.output)
        else:
            logger.info("Run: cd %s && vivado_hls -f *.tcl", args.output)
        return 0

    def _handle_quickstart(self, args) -> int:
        logger.info("Cambium Quick Start")
        logger.info("=" * 40)
        framework = self._build_framework(args)
        if args.config:
            if args.n_estimators is not None:
                framework.config.config["model"]["n_estimators"] = args.n_estimators
            if args.max_depth is not None:
                framework.config.config["model"]["max_depth"] = args.max_depth
            if args.max_leaf_nodes is not None:
                framework.config.config["model"]["max_leaf_nodes"] = args.max_leaf_nodes
            if args.max_nodes is not None:
                framework.config.config["export"]["max_nodes"] = args.max_nodes
            if args.task:
                framework.config.config["model"]["task"] = args.task

        logger.info("Using HLS backend: %s", framework.config.backend.backend)
        config_data = framework.config.config.get("data", {})

        data_source = args.data
        if not data_source:
            data_source = config_data.get("source") or config_data.get("url") or config_data.get("path")
        if not data_source:
            logger.error("No dataset provided. Use --data/-d or define one of data.source/data.url/data.path in config.")
            return 1

        feature_cols = _parse_csv_columns(args.features) if args.features else (config_data.get("feature_cols") or config_data.get("features"))
        target_cols = _parse_csv_columns(args.targets) if args.targets else (config_data.get("target_cols") or config_data.get("targets"))

        logger.info("Loading data from: %s", data_source)
        framework.load_data(data_source, feature_cols, target_cols)

        logger.info("Training model with automatic configuration...")
        metrics = framework.train()

        logger.info("\nQuick Start Results:")
        for key, value in metrics.items():
            logger.info("%s: %.4f", key, value) if isinstance(value, float) else logger.info("%s: %s", key, value)

        backend = framework.config.backend.backend
        logger.info("\nExporting to %s...", backend)
        framework.export_to_hls_j2()
        framework.generate_report()

        output_dir = framework.config.config.get("project", {}).get("output_dir", args.output)
        logger.info("\nQuick start completed!")
        logger.info("Check results in: %s", output_dir)
        logger.info("Backend: %s", backend)
        if backend == "vitis_hls":
            logger.info("\nTo run Vitis HLS synthesis:")
            logger.info("   cd %s", output_dir)
            logger.info("   vitis_hls -f %s.tcl", framework.config.config["project"]["name"])
        else:
            logger.info("\nTo run Vivado HLS synthesis:")
            logger.info("   cd %s", output_dir)
            logger.info("   vivado_hls -f %s.tcl", framework.config.config["project"]["name"])
            logger.info("\nAfter HLS export, generate bitstream:")
            logger.info(
                "   cd %s && vivado -mode batch -source vivado_block_design.tcl",
                output_dir,
            )
        return 0

    def _handle_create_config(self, args) -> int:
        logger.info("Cambium Create Configuration")
        create_sample_config(args.output)

        if args.backend != "vitis_hls":
            logger.info("Configuration template (Vivado HLS) saved to: %s", args.output)
            return 0

        with Path(args.output).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        config.setdefault("hls", {})
        config.setdefault("export", {})
        config["hls"]["backend"] = "vitis_hls"
        config["hls"]["vitis_flow_target"] = "hw"
        config["export"]["fpga_part"] = "xcvu9p-flga2104-2-i"
        config["export"]["clock_period"] = 3.33

        with Path(args.output).open("w", encoding="utf-8") as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        logger.info("Configuration template (Vitis HLS) saved to: %s", args.output)
        return 0

    def _handle_synthesis_report(self, args) -> int:
        project_dir = Path(args.project_dir)
        report_file = project_dir / "solution1" / "syn" / "report" / "predict_array_csynth.rpt"
        if not report_file.exists():
            logger.error("Report file not found: %s", report_file)
            return 1

        print(f"\nHLS Synthesis Report (summary): {report_file}\n{'=' * 60}")
        print_lines = []
        record = False
        with report_file.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                if "== Performance Estimates" in line:
                    record = True
                if record:
                    print_lines.append(line)
                if "== Utilization Estimates" in line:
                    for _ in range(15):
                        next_line = file.readline()
                        if not next_line:
                            break
                        print_lines.append(next_line)
                    break
        print("".join(print_lines))
        print(f"{'=' * 60}\nEnd of summary\n")
        return 0


def main(argv=None) -> int:
    try:
        print(get_logo("bold"))
    except UnicodeEncodeError:
        print("Cambium")
    cli = CambiumCLI()
    try:
        return cli.run(argv)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
