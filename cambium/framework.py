"""Top-level Cambium framework orchestration."""

from __future__ import annotations

import json
import logging
import pickle as pkl
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .codegen import Jinja2HLSCodeGenerator
from .config import CambiumConfig
from .data import DataManager
from .training import ModelTrainer

logger = logging.getLogger(__name__)


class CambiumFramework:
    """Main Cambium framework class."""

    def __init__(self, config_file: Optional[str] = None):
        self.config = CambiumConfig(config_file)
        self.data_manager = DataManager(self.config)
        self.trainer = ModelTrainer(self.config)
        self.code_generator: Optional[Jinja2HLSCodeGenerator] = None
        self.model = None
        self.metrics: Dict[str, Any] = {}
        self._split_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    def load_data(
        self,
        data_source: Union[str, pd.DataFrame],
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
    ):
        return self.data_manager.load_data(data_source, feature_cols, target_cols)

    def train(self, save_model: bool = True) -> Dict[str, Any]:
        x_train, x_test, y_train, y_test = self.data_manager.prepare_data()
        self._split_cache = (x_train, x_test, y_train, y_test)
        self.model = self.trainer.train_model(x_train, y_train)
        self.metrics = self.trainer.evaluate_model(x_test, y_test, self.data_manager.target_cols)
        if save_model:
            self._save_training_artifacts(x_test, y_test)
        return self.metrics

    def export_to_hls_j2(self, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if output_dir:
            self.config.config["project"]["output_dir"] = output_dir

        self.code_generator = Jinja2HLSCodeGenerator(self.config)
        implementation = self.config.config["export"].get("implementation", "auto")
        if implementation not in {"auto", "struct", "integer_axi", "float_axi"}:
            raise ValueError(f"Unknown HLS implementation: {implementation}")
        if implementation == "auto":
            implementation = "struct"
        integer_axi = implementation == "integer_axi"
        float_axi = implementation == "float_axi"
        rom_axi = integer_axi or float_axi
        generated_files = {"headers": [], "implementation": []}
        if integer_axi:
            generated_files["implementation"] = [
                self.code_generator.generate_integer_axi(self.model, self.data_manager)
            ]
        elif float_axi:
            generated_files["implementation"] = [
                self.code_generator.generate_float_axi(self.model, self.data_manager)
            ]
        else:
            generated_files["headers"] = self.code_generator.generate_headers(self.model, self.data_manager)
            generated_files["implementation"] = self.code_generator.generate_implementation(self.model, self.data_manager)

        if self._split_cache:
            _, x_test, _, _ = self._split_cache
        else:
            _, x_test, _, _ = self.data_manager.prepare_data()
        if integer_axi:
            generated_files["test"] = self.code_generator.generate_integer_testbench(self.model, x_test)
        elif float_axi:
            generated_files["test"] = self.code_generator.generate_float_testbench(self.model, x_test)
        else:
            hls_x_test = x_test
            if self.config.backend.is_vitis():
                limit = int(self.config.config["export"].get("hls_test_samples", 32))
                if limit < 1:
                    raise ValueError("export.hls_test_samples must be positive")
                hls_x_test = x_test[:limit]
            generated_files["test"] = [
                self.code_generator.generate_x_test_header(hls_x_test),
                self.code_generator.generate_rfr_tb(self._get_output_names(), self.config.config["model"]["task"]),
            ]
        self._integer_axi_export = integer_axi
        self._float_axi_export = float_axi
        self._rom_axi_export = rom_axi

        generated_files["build"] = self._generate_build_scripts()
        generated_files["vivado_tcl"] = [str(self._generate_vivado_tcl())]
        if self.config.backend.is_vitis():
            part = self.config.config["export"].get("fpga_part", "")
            generated_files["vivado_block_design"] = (
                [self._generate_zcu104_block_design_tcl()]
                if part == "xczu7ev-ffvc1156-2-e" else []
            )
        else:
            generated_files["vivado_block_design"] = [self._generate_vivado_block_design_tcl()]
        generated_files["pynq_driver"] = [self._generate_pynq_driver()]

        logger.info(
            "Code generation completed. Files saved to: %s",
            self.config.config["project"]["output_dir"],
        )
        return generated_files

    def _save_training_artifacts(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        output_dir = Path(self.config.config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X_test.npy", x_test)
        np.save(output_dir / "Y_test.npy", y_test)
        y_pred = self.model.predict(x_test)
        np.save(output_dir / "Y_pred.npy", y_pred)
        if self.config.config["model"]["task"] == "classification" and hasattr(self.model, "predict_proba"):
            np.save(output_dir / "Y_pred_proba.npy", self.model.predict_proba(x_test))

        with (output_dir / "cambium_model.pkl").open("wb") as file:
            pkl.dump(self.model, file)
        logger.info("sklearn model saved to: %s", output_dir / "cambium_model.pkl")

        if self.data_manager.scaler_x:
            with (output_dir / "scaler_x.pkl").open("wb") as file:
                pkl.dump(self.data_manager.scaler_x, file)

        if self.data_manager.scaler_y:
            with (output_dir / "scaler_y.pkl").open("wb") as file:
                pkl.dump(self.data_manager.scaler_y, file)

        self.config.save_config(str(output_dir / "cambium_config.yaml"))
        with (output_dir / "cambium_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(self.metrics, file, indent=2)

    def _get_output_names(self) -> List[str]:
        task = self.config.config["model"]["task"]
        if task == "classification":
            if self.model is None or not hasattr(self.model, "classes_"):
                raise ValueError("Classification output names require a trained model with classes_.")
            return [f"class_{cls}" for cls in self.model.classes_]
        return list(self.data_manager.target_cols or [])

    def _generate_build_scripts(self) -> List[str]:
        # Placeholder for future build-script generation.
        return []

    def _generate_pynq_driver(self) -> str:
        """Write a driver with the exact model dimensions, precision and scalers."""
        output_dir = Path(self.config.config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        precision = self.code_generator.precision_type if self.code_generator else self.config.config["export"]["precision"]
        if precision == "fixed":
            precision = "ap_fixed<16,6>"
        if precision == "float":
            width, integer_bits, is_float = 32, 32, True
        else:
            width, integer_bits = Jinja2HLSCodeGenerator._parse_ap_fixed_type(precision)
            is_float = False

        scaler_x = self.data_manager.scaler_x
        if scaler_x is None:
            x_spec = {"kind": "identity"}
        elif hasattr(scaler_x, "data_min_"):
            x_spec = {"kind": "minmax", "scale": scaler_x.scale_.tolist(),
                      "offset": scaler_x.min_.tolist()}
        elif hasattr(scaler_x, "mean_"):
            x_spec = {"kind": "standard", "center": scaler_x.mean_.tolist(),
                      "scale": scaler_x.scale_.tolist()}
        elif hasattr(scaler_x, "center_"):
            x_spec = {"kind": "robust", "center": scaler_x.center_.tolist(),
                      "scale": scaler_x.scale_.tolist()}
        else:
            raise ValueError("Cannot generate PYNQ driver for this feature scaler")
        scaler_y = self.data_manager.scaler_y
        y_spec = None if scaler_y is None else {
            "min": scaler_y.data_min_.tolist(), "range": scaler_y.data_range_.tolist()}
        spec = {
            "n_features": len(self.data_manager.feature_cols),
            "n_targets": len(self._get_output_names()),
            "width": width, "int_bits": integer_bits, "float": is_float,
            "x_scaler": x_spec, "y_scaler": y_spec,
        }
        source = (Path(__file__).with_name("pynq_driver_runtime.py")).read_text(encoding="utf-8")
        driver_path = output_dir / "axi_rfr_driver.py"
        driver_path.write_text(source.replace("MODEL_CONFIG = {}", f"MODEL_CONFIG = {spec!r}", 1),
                               encoding="utf-8")
        return str(driver_path)

    def _generate_zcu104_block_design_tcl(self) -> str:
        """Generate a ZCU104 AXI DMA overlay for the Vitis HLS IP export."""
        output_dir = Path(self.config.config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        project_name = self.config.config["project"]["name"]
        period = float(self.config.config["export"].get("clock_period", 5))
        if period <= 0:
            raise ValueError("export.clock_period must be positive")
        frequency_mhz = 1000.0 / period

        tcl_content = f'''# ZCU104 AXI4-Stream + simple-DMA overlay for Cambium.
# Run after: vitis_hls -f {project_name}.tcl
# Validate only: vivado -mode batch -source vivado_block_design.tcl -tclargs validate
# Full build:    vivado -mode batch -source vivado_block_design.tcl
set script_dir [file dirname [file normalize [info script]]]
cd $script_dir
set project_dir [file join $script_dir vivado_project]
set ip_repo [file join $script_dir {project_name} solution1 impl ip]
if {{![file exists [file join $ip_repo component.xml]]}} {{
    error "HLS IP is missing at $ip_repo; run Vitis HLS first"
}}
create_project -force cambium_zcu104 $project_dir -part xczu7ev-ffvc1156-2-e
set_param general.maxThreads 2
set_property board_part xilinx.com:zcu104:part0:1.1 [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog

create_bd_design design_1
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {{apply_board_preset "1"}} [get_bd_cells ps]
set_property -dict [list \\
    CONFIG.PSU__USE__M_AXI_GP1 {{0}} \\
    CONFIG.PSU__USE__S_AXI_GP2 {{1}} \\
    CONFIG.PSU__USE__S_AXI_GP3 {{1}} \\
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {{{frequency_mhz:g}}}] [get_bd_cells ps]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0
set_property -dict [list \\
    CONFIG.c_include_sg {{0}} CONFIG.c_sg_length_width {{26}} \\
    CONFIG.c_m_axi_mm2s_data_width {{32}} CONFIG.c_m_axi_s2mm_data_width {{32}} \\
    CONFIG.c_m_axis_mm2s_tdata_width {{32}} CONFIG.c_s_axis_s2mm_tdata_width {{32}} \\
    CONFIG.c_mm2s_burst_size {{256}} CONFIG.c_s2mm_burst_size {{256}}] [get_bd_cells axi_dma_0]
create_bd_cell -type ip -vlnv xilinx.com:hls:predict_axi:1.0 predict_axi_0
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins predict_axi_0/in_r]
connect_bd_intf_net [get_bd_intf_pins predict_axi_0/out_r] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# One PS master controls the DMA. Two HP ports carry the two memory directions.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {{Master "/ps/M_AXI_HPM0_FPD" Clk "Auto"}} [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {{Master "/axi_dma_0/M_AXI_MM2S" Slave "/ps/S_AXI_HP0_FPD" Clk "Auto"}} [get_bd_intf_pins ps/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {{Master "/axi_dma_0/M_AXI_S2MM" Slave "/ps/S_AXI_HP1_FPD" Clk "Auto"}} [get_bd_intf_pins ps/S_AXI_HP1_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config {{Clk "/ps/pl_clk0"}} [get_bd_pins predict_axi_0/ap_clk]

# Prevent an unnoticed clock-domain change by connection automation.
set pl_clock_net [get_bd_nets -of_objects [get_bd_pins ps/pl_clk0]]
foreach clock_pin {{predict_axi_0/ap_clk axi_dma_0/s_axi_lite_aclk axi_dma_0/m_axi_mm2s_aclk axi_dma_0/m_axi_s2mm_aclk}} {{
    set pin [get_bd_pins -quiet $clock_pin]
    if {{[llength $pin] && [get_bd_nets -of_objects $pin] ne $pl_clock_net}} {{
        error "Clock pin $clock_pin is not driven by ps/pl_clk0"
    }}
}}
assign_bd_address
validate_bd_design
save_bd_design
set bd_file [get_files design_1.bd]
generate_target all $bd_file
add_files -norecurse [make_wrapper -files $bd_file -top]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
if {{[llength $argv] && [lindex $argv 0] eq "validate"}} {{
    puts "SUCCESS: ZCU104 block design validated"
    close_project
    exit 0
}}

set report_dir [file join $script_dir vivado_reports]
file mkdir $report_dir
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {{[get_property PROGRESS [get_runs synth_1]] ne "100%"}} {{error "Synthesis failed"}}
open_run synth_1
report_utilization -file [file join $report_dir utilization_synth.rpt]
report_power -file [file join $report_dir power_synth.rpt]
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {{[get_property PROGRESS [get_runs impl_1]] ne "100%"}} {{error "Implementation failed"}}
open_run impl_1
report_timing_summary -file [file join $report_dir timing.rpt]
report_utilization -file [file join $report_dir utilization_routed.rpt]
report_power -file [file join $report_dir power_routed.rpt]
file copy -force [file join [get_property DIRECTORY [get_runs impl_1]] design_1_wrapper.bit] [file join $script_dir output.bit]
file copy -force [file join $project_dir cambium_zcu104.gen sources_1 bd design_1 hw_handoff design_1.hwh] [file join $script_dir output.hwh]
puts "SUCCESS: output.bit and output.hwh generated in $script_dir"
close_project
'''
        tcl_path = output_dir / "vivado_block_design.tcl"
        tcl_path.write_text(tcl_content, encoding="utf-8")
        logger.info("ZCU104 Vivado block design TCL generated: %s", tcl_path)
        return str(tcl_path)

    def _generate_vivado_block_design_tcl(self) -> str:
        """Generate Vivado block design TCL for FPGA integration."""
        project_name = self.config.config["project"]["name"]
        output_dir = Path(self.config.config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        part_name = self.config.config["hls"].get("target_device", "xc7z020clg400-1")
        board = self.config.config["hls"].get("board_part", "www.digilentinc.com:pynq-z1:part0:1.0")
        hls_ip_path = f"{project_name}/solution1/impl/ip"

        tcl_content = f"""# Project setup
set design_name design_1
create_project project_1 rf_accelerator -part {part_name} -force
set_property BOARD_PART {board} [current_project]
set_property IP_REPO_PATHS "{hls_ip_path}" [current_project]
update_ip_catalog

create_bd_design $design_name

# Create PS7 with board preset first
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {{make_external "FIXED_IO, DDR" apply_board_preset "1"}} [get_bd_cells processing_system7_0]

# Now configure HP0 and GP0
set_property -dict [list \\
  CONFIG.PCW_USE_S_AXI_HP0 {{1}} \\
  CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {{64}} \\
  CONFIG.PCW_USE_M_AXI_GP0 {{1}}] [get_bd_cells processing_system7_0]

# Other IPs
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list CONFIG.c_include_sg {{0}} CONFIG.c_sg_include_stscntrl_strm {{0}}] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv xilinx.com:hls:predict_axi:1.0 predict_axi_0
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ps7_0_axi_periph
set_property CONFIG.NUM_MI {{1}} [get_bd_cells ps7_0_axi_periph]

# Connections
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins predict_axi_0/in_r]
connect_bd_intf_net [get_bd_intf_pins predict_axi_0/out_r] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins ps7_0_axi_periph/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ps7_0_axi_periph/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# Clocks
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \\
  [get_bd_pins axi_dma_0/s_axi_lite_aclk] \\
  [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \\
  [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \\
  [get_bd_pins predict_axi_0/ap_clk] \\
  [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \\
  [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \\
  [get_bd_pins ps7_0_axi_periph/ACLK] \\
  [get_bd_pins ps7_0_axi_periph/S00_ACLK] \\
  [get_bd_pins ps7_0_axi_periph/M00_ACLK] \\
  [get_bd_pins rst_ps7_0_100M/slowest_sync_clk] \\
  [get_bd_pins smartconnect_0/aclk]

# Resets
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \\
  [get_bd_pins axi_dma_0/axi_resetn] \\
  [get_bd_pins predict_axi_0/ap_rst_n] \\
  [get_bd_pins ps7_0_axi_periph/ARESETN] \\
  [get_bd_pins ps7_0_axi_periph/S00_ARESETN] \\
  [get_bd_pins ps7_0_axi_periph/M00_ARESETN] \\
  [get_bd_pins smartconnect_0/aresetn]

# Addresses
assign_bd_address -offset 0x00000000 -range 0x20000000 [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -target_address_space [get_bd_addr_spaces axi_dma_0/Data_MM2S]
assign_bd_address -offset 0x00000000 -range 0x20000000 [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM] -target_address_space [get_bd_addr_spaces axi_dma_0/Data_S2MM]
assign_bd_address -offset 0x40400000 -range 0x00010000 [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] -target_address_space [get_bd_addr_spaces processing_system7_0/Data]

validate_bd_design
save_bd_design

# Build
make_wrapper -files [get_files ${{design_name}}.bd] -top
add_files -norecurse rf_accelerator/project_1.srcs/sources_1/bd/${{design_name}}/hdl/${{design_name}}_wrapper.v
launch_runs impl_1 -to_step write_bitstream -jobs 6
wait_on_run impl_1

file copy -force rf_accelerator/project_1.runs/impl_1/${{design_name}}_wrapper.bit ${{design_name}}.bit
file copy -force rf_accelerator/project_1.srcs/sources_1/bd/${{design_name}}/hw_handoff/${{design_name}}.hwh ${{design_name}}.hwh

puts "SUCCESS: Build complete!"
"""

        tcl_path = output_dir / "vivado_block_design.tcl"
        tcl_path.write_text(tcl_content, encoding="utf-8")
        logger.info("Vivado block design TCL generated: %s", tcl_path)
        return str(tcl_path)

    def _generate_vivado_tcl(self) -> Path:
        """Generate backend-aware TCL script."""
        project_name = self.config.config["project"]["name"]
        output_dir = Path(self.config.config["project"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        top_function = self.config.config["export"].get("top_function", "predict_axi")
        part_name = self.config.config["export"].get("fpga_part", "xc7z020clg400-1")
        clock_period = self.config.config["export"].get("clock_period", 5)
        backend = self.config.backend
        tcl_path = output_dir / f"{project_name}.tcl"

        if getattr(self, "_integer_axi_export", False):
            design_files = [output_dir / "cambium_integer_axi.cpp"]
            tb_files = [output_dir / "integer_axi_tb.cpp", output_dir / "integer_axi_vectors.txt"]
        elif getattr(self, "_float_axi_export", False):
            design_files = [output_dir / "cambium_float_axi.cpp"]
            tb_files = [output_dir / "float_axi_tb.cpp", output_dir / "float_axi_vectors.txt"]
        else:
            cpp_files = sorted(output_dir.glob("*.cpp"))
            header_files = sorted(output_dir.glob("*.h"))
            tb_files = [file for file in cpp_files + header_files if "tb" in file.stem.lower() or "test" in file.stem.lower()]
            design_files = [file for file in cpp_files + header_files if file not in tb_files]

        lines: List[str] = []
        lines.append("############################################################")
        lines.append(f"## Generated for {backend.config['tool_command']}")
        lines.append("## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.")
        lines.append("############################################################")

        lines.append("set script_dir [file dirname [file normalize [info script]]]")
        lines.append(f'set project_dir [file join $script_dir "{project_name}"]')
        lines.append("cd $script_dir")
        lines.append("if {[file exists $project_dir]} {")
        lines.append("    file delete -force $project_dir")
        lines.append("}")
        lines.append("file mkdir $project_dir")
        lines.append(f"open_project {project_name}")

        lines.append(f"set_top {top_function}")
        lines.extend(f'add_files [file join $script_dir "{file.name}"]' for file in design_files)
        lines.extend(f'add_files -tb [file join $script_dir "{file.name}"]' for file in tb_files)

        # AXI IP is integrated through Vivado's IP catalog, including when
        # Vitis HLS is the compiler for the original struct-based datapath.
        lines.append('open_solution "solution1"')

        lines.append(f"set_part {{{part_name}}}")
        lines.append(f"create_clock -period {clock_period} -name default")

        if backend.is_vitis() and backend.config["default_pipeline_loops"] is not None:
            lines.append(f"config_compile -pipeline_loops {backend.config['default_pipeline_loops']}")

        lines.append('#source "./solution1/directives.tcl"')
        lines.append("csim_design")
        lines.append("csynth_design")
        if backend.is_vitis():
            lines.append("cosim_design -rtl verilog")
        else:
            lines.append(f"cosim_design -trace_level {backend.config['cosim_trace']}")
        lines.append("export_design -format ip_catalog")
        lines.append("exit")

        tcl_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("%s TCL file generated at: %s", backend.config["tool_command"], tcl_path)
        return tcl_path

    def print_synthesis_report(self, project_dir: str) -> None:
        """Print Vivado/Vitis HLS synthesis utilization and timing reports."""
        project_path = Path(project_dir).resolve()

        if not project_path.exists():
            logger.error("Project directory not found: %s", project_path)
            logger.info("Please specify a valid project directory or run quick-start first.")
            return

        solution_dir = project_path / "solution1"
        if not solution_dir.exists():
            logger.error("No 'solution1' directory found in: %s", project_path)
            logger.info("Please run synthesis first using:")
            logger.info("  cd %s && vivado_hls -f %s.tcl", project_path, self.config.config["project"]["name"])
            return

        report_dir = solution_dir / "syn" / "report"
        if not report_dir.exists():
            logger.error("No synthesis report directory found: %s", report_dir)
            logger.info("Make sure 'csynth_design' completed successfully.")
            return

        print(f"\n{'=' * 60}")
        print(f" SYNTHESIS REPORT: {project_path.name}")
        print(f"{'=' * 60}")
        self._print_utilization_report(report_dir)
        self._print_timing_report(report_dir)
        self._print_synthesis_summary(project_path)

    def _print_utilization_report(self, report_dir: Path) -> None:
        util_files = sorted(report_dir.glob("*utilization*"))
        if not util_files:
            logger.warning("Utilization report not found.")
            return

        util_file = util_files[0]
        print("\nRESOURCE UTILIZATION:")
        print(f"{'-' * 50}")
        print(f"File: {util_file.name}")

        try:
            lines = util_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            in_table = False
            for line in lines:
                stripped = line.strip()
                if "Utilization Estimates" in stripped or "== Utilization ==" in stripped:
                    in_table = True
                    continue
                if in_table and ("====" in stripped or "Latency" in stripped):
                    break
                if in_table and stripped and any(key in stripped for key in ["BRAM", "DSP", "FF", "LUT", "URAM"]):
                    print(f"  {stripped}")
        except Exception as exc:
            logger.error("Failed to parse utilization report %s: %s", util_file, exc)

    def _print_timing_report(self, report_dir: Path) -> None:
        timing_files = sorted(report_dir.glob("*timing*"))
        if not timing_files:
            logger.warning("Timing report not found.")
            return

        timing_file = timing_files[0]
        print("\n TIMING SUMMARY:")
        print(f"{'-' * 50}")
        print(f" File: {timing_file.name}")

        try:
            lines = timing_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                stripped = line.strip()
                if any(
                    keyword in stripped.lower()
                    for keyword in ["clock", "target", "achieved", "slack", "timing", "latency", "ns"]
                ) and ("|" in stripped or ":" in stripped):
                    print(f"  {stripped}")
        except Exception as exc:
            logger.error(" Failed to parse timing report %s: %s", timing_file, exc)

    def _print_synthesis_summary(self, project_path: Path) -> None:
        print(f"\n PROJECT STATUS: {project_path.name}")
        print(f"{'-' * 50}")
        stages = {
            "C Simulation": (project_path / "solution1" / "csim").exists(),
            "Synthesis": (project_path / "solution1" / "syn").exists(),
            "Co-Simulation": (project_path / "solution1" / "sim").exists(),
            "IP Export": (project_path / "solution1" / "impl").exists(),
        }
        for stage, done in stages.items():
            status = " COMPLETED" if done else "NOT RUN"
            print(f"  {stage:<18} {status}")

        if all(stages.values()):
            print("\n All synthesis stages completed successfully!")
        else:
            print("\n To complete synthesis, run:")
            print(f"  cd {project_path} && vivado_hls -f {self.config.config['project']['name']}.tcl")

    def generate_report(self, output_file: Optional[str] = None) -> str:
        if not output_file:
            output_dir = Path(self.config.config["project"]["output_dir"])
            output_file = str(output_dir / "cambium_report.md")

        report = self._create_markdown_report() or "# Error: Report generation failed"
        Path(output_file).write_text(report, encoding="utf-8")
        logger.info("Report generated: %s", output_file)
        return str(output_file)

    def _create_markdown_report(self) -> str:
        project_name = self.config.config["project"]["name"]
        output_dir = Path(self.config.config["project"]["output_dir"])

        header_files = list(output_dir.glob("*.h"))
        cpp_files = list(output_dir.glob("*.cpp"))
        build_files = list(output_dir.glob("*.tcl")) + list(output_dir.glob("Makefile")) + list(output_dir.glob("CMakeLists.txt"))

        header_list = "\n".join(f"- `{file.name}`" for file in header_files)
        cpp_list = "\n".join(f"- `{file.name}`" for file in cpp_files)
        build_list = "\n".join(f"- `{file.name}`" for file in build_files)

        metrics_info = ""
        if self.metrics:
            metrics_info = "\n### Model Performance\n"
            for key, value in self.metrics.items():
                metrics_info += f"- **{key}**: {value:.4f}\n" if isinstance(value, float) else f"- **{key}**: {value}\n"

        feature_lines = (
            "\n".join(f"- {column}" for column in self.data_manager.feature_cols)
            if self.data_manager.feature_cols
            else "No features specified"
        )
        target_lines = (
            "\n".join(f"- {column}" for column in self.data_manager.target_cols)
            if self.data_manager.target_cols
            else "No targets specified"
        )

        return f"""# Cambium Report: {project_name}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview

### Configuration Summary
- **Model Type**: {self.config.config['model']['type']}
- **Task**: {self.config.config['model']['task']}
- **Estimators**: {self.config.config['model']['n_estimators']}
- **Max Depth**: {self.config.config['model']['max_depth']}
- **Features**: {len(self.data_manager.feature_cols) if self.data_manager.feature_cols else 'N/A'}
- **Targets**: {len(self.data_manager.target_cols) if self.data_manager.target_cols else 'N/A'}
- **Precision**: {self.config.config['export']['precision']}
- **Output Directory**: {self.config.config['project']['output_dir']}
{metrics_info}

## Generated Files

### Header Files ({len(header_files)} files)
{header_list if header_list else 'No header files generated.'}

### Implementation Files ({len(cpp_files)} files)
{cpp_list if cpp_list else 'No implementation files generated.'}

### Build Files ({len(build_files)} files)
{build_list if build_list else 'No build files generated.'}

## Usage Instructions

### HLS Synthesis
```bash
cd {output_dir.name}
./run_synthesis.sh
```

### Software Simulation
```bash
cd {output_dir.name}
make
./rfr_tb
```

## Model Information

### Features
{feature_lines}

### Targets
{target_lines}

Generated by Cambium v1.1
"""


def create_sample_config(output_file: str = "cambium_config.yaml") -> None:
    config = CambiumConfig()
    config.save_config(output_file)
    print(f"Sample configuration saved to: {output_file}")


def quick_start_example() -> Optional[CambiumFramework]:
    try:
        logger.info("Initializing Cambium...")
        framework = CambiumFramework()
        framework.config.config["export"]["format"] = "legacy"

        data_url = "https://raw.githubusercontent.com/AbuAli3/ee/main/alldata.csv"
        feature_cols = ["Occupancy", "Rel_Hum", "Room_Temp", "Air_Flow_Rat", "Air_Temp"]
        target_cols = ["Elec_Cons", "Therm_Eng_Cons", "PMV"]

        logger.info("Loading data...")
        framework.load_data(data_url, feature_cols, target_cols)

        logger.info("Training model...")
        metrics = framework.train()
        logger.info("Training completed with metrics: %s", metrics)

        logger.info("Exporting to HLS with Jinja2...")
        generated_files = framework.export_to_hls_j2("cambium_output")
        logger.info("Generated files: %s", list(generated_files.keys()))

        logger.info("Generating Vivado TCL script...")
        generated_tcl = framework._generate_vivado_tcl()
        logger.info("Vivado TCL file generated: %s", generated_tcl)

        logger.info("Generating Markdown report...")
        report_file = framework.generate_report()
        logger.info("Report generated: %s", report_file)

        return framework
    except Exception as exc:
        logger.error("Error in quick start: %s", exc)
        return None
