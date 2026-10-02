"""Regression tests for the Vitis fixed-point ROM export path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from cambium.framework import CambiumFramework
from cambium.cli import CambiumCLI


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

    def test_classification_exports_one_exact_code_per_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            framework = self.make_framework(output)
            x = framework._split_cache[1]
            labels = np.array([0, 0, 1, 1, 2, 2])
            model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
            model.fit(x, labels)
            framework.model = model
            framework.config.config["model"]["task"] = "classification"
            framework.export_to_hls_j2()
            vectors = np.loadtxt(output / "integer_axi_vectors.txt", dtype=int)
            self.assertEqual(vectors.shape, (6, 5))
            self.assertGreater(len(np.unique(vectors[:, -3:], axis=0)), 1)
            source = (output / "cambium_integer_axi.cpp").read_text()
            self.assertIn("VALUE[TOTAL_TREES][NODES][OUTPUTS]", source)
            self.assertIn("batch_last && c == OUTPUTS-1", source)

    def test_float_rom_export_for_regression_and_classification(self):
        for classification in (False, True):
            with self.subTest(classification=classification), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                framework = self.make_framework(output)
                framework.config.config["export"]["precision"] = "float"
                if classification:
                    x = framework._split_cache[1]
                    model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
                    model.fit(x, np.array([0, 0, 1, 1, 2, 2]))
                    framework.model = model
                    framework.config.config["model"]["task"] = "classification"
                generated = framework.export_to_hls_j2()
                self.assertEqual([Path(p).name for p in generated["implementation"]],
                                 ["cambium_float_axi.cpp"])
                vectors = np.loadtxt(output / "float_axi_vectors.txt")
                self.assertEqual(vectors.shape, (6, 5 if classification else 3))
                self.assertIn("cosim_design -rtl verilog", (output / "cambium_project.tcl").read_text())

    def test_tree_depth_and_node_controls_remain_independent(self):
        cli = CambiumCLI()
        args = cli.parser.parse_args([
            "quick-start", "--n-estimators", "5", "--max-depth", "3",
            "--max-leaf-nodes", "4", "--max-nodes", "31",
        ])
        framework = cli._build_framework(args)
        self.assertEqual(framework.config.config["model"]["n_estimators"], 5)
        self.assertEqual(framework.config.config["model"]["max_depth"], 3)
        self.assertEqual(framework.config.config["model"]["max_leaf_nodes"], 4)
        self.assertEqual(framework.config.config["export"]["max_nodes"], 31)
        x = np.linspace(-1, 1, 20)[:, None]
        model = framework.trainer.train_model(x, x ** 2)
        self.assertEqual(len(model.estimators_), 5)
        self.assertTrue(all(tree.tree_.max_depth <= 3 and tree.tree_.n_leaves <= 4
                            for tree in model.estimators_))

    def test_multioutput_regression_exports_each_forest(self):
        for precision in ("ap_fixed<10,4>", "float"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                framework = self.make_framework(output)
                x = framework._split_cache[1]
                y = np.column_stack((x[:, 0] * 0.4, x[:, 1] * -0.3))
                model = MultiOutputRegressor(RandomForestRegressor(
                    n_estimators=3, max_depth=3, random_state=42))
                model.fit(x, y)
                framework.model = model
                framework.data_manager.target_cols = ["y1", "y2"]
                framework.config.config["export"]["precision"] = precision
                generated = framework.export_to_hls_j2()
                name = "float" if precision == "float" else "integer"
                vectors = np.loadtxt(output / f"{name}_axi_vectors.txt")
                self.assertEqual(vectors.shape, (6, 4))
                source = Path(generated["implementation"][0]).read_text()
                self.assertIn("FORESTS = 2", source)
                self.assertIn("c * TREES + tree", source)

    def test_export_command_restores_saved_model_config_and_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trained = root / "trained"
            exported = root / "exported"
            framework = self.make_framework(trained)
            framework.config.config["data"].update(feature_cols=["a", "b"], target_cols=["y"])
            x = framework._split_cache[1]
            framework._save_training_artifacts(x, np.zeros(len(x)))
            code = CambiumCLI().run([
                "export", "--model", str(trained / "cambium_model.pkl"),
                "--output", str(exported), "--quiet",
            ])
            self.assertEqual(code, 0)
            self.assertEqual(np.loadtxt(exported / "integer_axi_vectors.txt").shape, (6, 3))
            self.assertIn("cosim_design -rtl verilog", (exported / "cambium_project.tcl").read_text())


if __name__ == "__main__":
    unittest.main()
