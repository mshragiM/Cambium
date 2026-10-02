"""Regression tests for the Vitis fixed-point ROM export path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from cambium.framework import CambiumFramework


class IntegerAxiExportTests(unittest.TestCase):
    def make_framework(self, output, implementation="auto"):
        x = np.array([[-1.0, 0.5], [-0.5, -0.5], [0.0, 0.2],
                      [0.5, -0.2], [1.0, 0.9], [1.5, -0.9]], dtype=np.float32)
        y = np.array([-0.75, -0.4, 0.15, 0.4, 0.85, 0.95], dtype=np.float32)
        model = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
        model.fit(x, y)
        framework = CambiumFramework()
        framework.config.set_backend("vitis_hls")
        framework.config.config["project"]["output_dir"] = str(output)
        framework.config.config["model"].update(task="regression", n_estimators=3, max_depth=3)
        framework.config.config["export"].update(
            precision="ap_fixed<10,4>", implementation=implementation)
        framework.data_manager.feature_cols = ["a", "b"]
        framework.data_manager.target_cols = ["y"]
        framework.model = model
        framework._split_cache = (x, x, y, y)
        return framework

    def test_auto_export_has_strict_rom_testbench_and_cosimulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            generated = self.make_framework(output).export_to_hls_j2()
            self.assertEqual([Path(p).name for p in generated["implementation"]],
                             ["cambium_integer_axi.cpp"])
            vectors = np.loadtxt(output / "integer_axi_vectors.txt", dtype=int)
            self.assertEqual(vectors.shape, (6, 3))
            self.assertGreater(len(np.unique(vectors[:, -1])), 1)
            tcl = (output / "cambium_project.tcl").read_text()
            self.assertIn("cosim_design -rtl verilog", tcl)
            self.assertIn("integer_axi_vectors.txt", tcl)
            self.assertIn("export_design -format ip_catalog", tcl)
            self.assertNotIn("myproj_core.cpp", tcl)
            tb = (output / "integer_axi_tb.cpp").read_text()
            self.assertIn("int(actual) != expected", tb)
            self.assertIn("result.keep != 15", tb)

    def test_explicit_struct_export_keeps_legacy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            generated = self.make_framework(output, "struct").export_to_hls_j2()
            self.assertIn("myproj_core.cpp", [Path(p).name for p in generated["implementation"]])
            self.assertNotIn("integer_axi_vectors.txt", (output / "cambium_project.tcl").read_text())

    def test_test_vector_overflow_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            framework = self.make_framework(Path(tmp))
            x = framework._split_cache[1].copy()
            x[0, 0] = 100
            framework._split_cache = (x, x, None, None)
            with self.assertRaisesRegex(ValueError, "overflows"):
                framework.export_to_hls_j2()

    def test_default_fixed_precision_exports_integer_axi(self):
        with tempfile.TemporaryDirectory() as tmp:
            framework = self.make_framework(Path(tmp))
            framework.config.config["export"]["precision"] = "fixed"
            generated = framework.export_to_hls_j2()
            self.assertEqual([Path(p).name for p in generated["implementation"]],
                             ["cambium_integer_axi.cpp"])
            self.assertIn("WIDTH = 16", (Path(tmp) / "cambium_integer_axi.cpp").read_text())


if __name__ == "__main__":
    unittest.main()
