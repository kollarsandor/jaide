# ============================================================================
# JAIDE v40 ASIC Synthesis Script
# Tool: Synopsys Design Compiler
# ============================================================================
# Description: Complete synthesis flow from RTL to gate-level netlist
# Technology: Generic (configure for target PDK)
# Clock: 100 MHz (10ns period)
# ============================================================================

# ============================================================================
# Setup and Configuration
# ============================================================================

set DESIGN_NAME "top_level"
set CLK_PERIOD 10.0
set CLK_PORT "clk"
set RST_PORT "rst_n"

set RTL_DIR "../fpga"
set TECH_LIB_DIR "/path/to/technology/library"
set OUTPUT_DIR "./output"

file mkdir $OUTPUT_DIR

# Suppress specific warnings
set_app_var sh_enable_page_mode false
suppress_message LINT-1
suppress_message LINT-28
suppress_message LINT-29

# ============================================================================
# Technology Library Setup
# ============================================================================

# Target technology library (customize for your PDK)
set target_library [list \
    ${TECH_LIB_DIR}/slow.db \
    ${TECH_LIB_DIR}/typical.db \
    ${TECH_LIB_DIR}/fast.db \
]

# Link library includes target + standard cells + macros
set link_library [list \
    "*" \
    ${TECH_LIB_DIR}/slow.db \
    ${TECH_LIB_DIR}/typical.db \
    ${TECH_LIB_DIR}/fast.db \
    ${TECH_LIB_DIR}/memory_compiler.db \
]

# Symbol library for schematic generation
set symbol_library [list \
    ${TECH_LIB_DIR}/symbols.sdb \
]

# Physical library for placement info
set mw_reference_library ${TECH_LIB_DIR}/mw_lib
set mw_design_library ${OUTPUT_DIR}/mw_${DESIGN_NAME}

# ============================================================================
# Read and Elaborate Design
# ============================================================================

echo "============================================"
echo "Reading RTL Design"
echo "============================================"

# Read Verilog RTL files
read_verilog -rtl [list \
    ${RTL_DIR}/top_level.v \
]

# If using Clash-generated Verilog (after clash compilation)
# read_verilog -rtl ${RTL_DIR}/MemoryArbiter.topEntity.v
# read_verilog -rtl ${RTL_DIR}/SSISearch.topEntity.v
# read_verilog -rtl ${RTL_DIR}/RankerCore.topEntity.v

current_design $DESIGN_NAME
link

echo "============================================"
echo "Elaborating Design"
echo "============================================"

elaborate $DESIGN_NAME
current_design $DESIGN_NAME
link

check_design > ${OUTPUT_DIR}/check_design.rpt

# ============================================================================
# Define Design Environment
# ============================================================================

echo "============================================"
echo "Setting Design Constraints"
echo "============================================"

# Define clock
create_clock -name $CLK_PORT -period $CLK_PERIOD [get_ports $CLK_PORT]

# Clock uncertainty (jitter + skew)
set_clock_uncertainty -setup 0.2 [get_clocks $CLK_PORT]
set_clock_uncertainty -hold 0.1 [get_clocks $CLK_PORT]

# Clock transition
set_clock_transition 0.1 [get_clocks $CLK_PORT]

# Clock latency
set_clock_latency -source 0.5 [get_clocks $CLK_PORT]
set_clock_latency 0.3 [get_clocks $CLK_PORT]

# Input/Output delays relative to clock
set_input_delay -clock $CLK_PORT -max 2.0 [all_inputs]
set_input_delay -clock $CLK_PORT -min 0.5 [all_inputs]
set_output_delay -clock $CLK_PORT -max 2.0 [all_outputs]
set_output_delay -clock $CLK_PORT -min 0.5 [all_outputs]

# Exception: async reset
set_input_delay 0 -clock $CLK_PORT [get_ports $RST_PORT]
set_false_path -from [get_ports $RST_PORT]

# Exceptions: status LEDs and interrupts
set_output_delay 0 -clock $CLK_PORT [get_ports led_*]
set_output_delay 0 -clock $CLK_PORT [get_ports irq_out]
set_false_path -to [get_ports led_*]
set_false_path -to [get_ports irq_out]

# Multi-cycle paths (as defined in constraints.pcf)
# Memory arbiter: 4 cycles
set_multicycle_path -setup 4 -from [get_pins -hierarchical *arbiter*/*] \
                               -to [get_pins -hierarchical mem_*]
set_multicycle_path -hold 3 -from [get_pins -hierarchical *arbiter*/*] \
                             -to [get_pins -hierarchical mem_*]

# SSI search: 32 cycles (max tree depth)
set_multicycle_path -setup 32 -from [get_pins -hierarchical *ssi*/*] \
                                -to [get_pins -hierarchical mem_*]
set_multicycle_path -hold 31 -from [get_pins -hierarchical *ssi*/*] \
                              -to [get_pins -hierarchical mem_*]

# ============================================================================
# Design Rules and Optimization Goals
# ============================================================================

# Operating conditions
set_operating_conditions -max slow -max_library slow \
                         -min fast -min_library fast

# Wire load model
set_wire_load_model -name "estimated" -library typical
set_wire_load_mode top

# Drive strength for inputs
set_driving_cell -lib_cell BUFX4 -library typical [all_inputs]
remove_driving_cell [get_ports $CLK_PORT]
remove_driving_cell [get_ports $RST_PORT]

# Load capacitance for outputs
set_load 0.05 [all_outputs]

# Max transition time
set_max_transition 0.5 $DESIGN_NAME

# Max fanout
set_max_fanout 16 $DESIGN_NAME

# Max capacitance
set_max_capacitance 0.5 [all_outputs]

# Area constraint (soft)
set_max_area 0

# ============================================================================
# Compile Strategy
# ============================================================================

echo "============================================"
echo "Compiling Design - Initial Mapping"
echo "============================================"

# Initial compile with medium effort
compile_ultra -gate_clock -no_autoungroup

# ============================================================================
# Optimization for Power
# ============================================================================

echo "============================================"
echo "Power Optimization"
echo "============================================"

# Enable clock gating
set_clock_gating_style -sequential_cell latch \
                       -minimum_bitwidth 4 \
                       -control_point before

# Compile with clock gating
compile_ultra -gate_clock -incremental

# Dynamic power optimization
set_dynamic_optimization true

# Multi-Vt optimization (if available)
# set_multi_vt_optimization true

# ============================================================================
# Incremental Optimization
# ============================================================================

echo "============================================"
echo "Incremental Optimization"
echo "============================================"

# Focus on critical paths
compile_ultra -incremental -only_design_rule

# ============================================================================
# Reports
# ============================================================================

echo "============================================"
echo "Generating Reports"
echo "============================================"

report_timing -max_paths 10 -transition_time -nets -attributes \
    > ${OUTPUT_DIR}/timing.rpt

report_area -hierarchy > ${OUTPUT_DIR}/area.rpt

report_power -hierarchy > ${OUTPUT_DIR}/power.rpt

report_constraint -all_violators > ${OUTPUT_DIR}/constraints.rpt

report_qor > ${OUTPUT_DIR}/qor.rpt

report_resources > ${OUTPUT_DIR}/resources.rpt

report_clock_gating -gated -ungated > ${OUTPUT_DIR}/clock_gating.rpt

check_design > ${OUTPUT_DIR}/check_design_final.rpt

# ============================================================================
# Write Output Files
# ============================================================================

echo "============================================"
echo "Writing Output Files"
echo "============================================"

# Gate-level netlist (Verilog)
change_names -rules verilog -hierarchy
write -format verilog -hierarchy -output ${OUTPUT_DIR}/${DESIGN_NAME}_synth.v

# DDC format (Design Compiler internal)
write -format ddc -hierarchy -output ${OUTPUT_DIR}/${DESIGN_NAME}.ddc

# SDC constraints for back-end tools
write_sdc ${OUTPUT_DIR}/${DESIGN_NAME}.sdc

# SDF for timing simulation
write_sdf ${OUTPUT_DIR}/${DESIGN_NAME}.sdf

# Design constraints
write_script > ${OUTPUT_DIR}/${DESIGN_NAME}_constraints.tcl

# ============================================================================
# Summary
# ============================================================================

echo "============================================"
echo "Synthesis Complete"
echo "============================================"
echo "Outputs written to: ${OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - ${DESIGN_NAME}_synth.v  : Gate-level netlist"
echo "  - ${DESIGN_NAME}.ddc      : Design database"
echo "  - ${DESIGN_NAME}.sdc      : Timing constraints"
echo "  - timing.rpt              : Timing report"
echo "  - area.rpt                : Area report"
echo "  - power.rpt               : Power report"
echo ""

# Print QoR summary
report_qor

exit
