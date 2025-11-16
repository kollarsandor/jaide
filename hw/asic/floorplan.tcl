# ============================================================================
# JAIDE v40 ASIC Floorplanning Script
# Tool: Synopsys IC Compiler / ICC2
# ============================================================================
# Description: Complete floorplanning flow for ASIC physical design
# Technology: Generic (configure for target PDK)
# Die Size: 5mm x 5mm (example)
# ============================================================================

# ============================================================================
# Setup and Configuration
# ============================================================================

set DESIGN_NAME "top_level"
set ICC_OUTPUT_DIR "./icc_output"
set NETLIST_DIR "./output"

file mkdir $ICC_OUTPUT_DIR

# Technology files (customize for your PDK)
set TECH_FILE "/path/to/technology/tech.tf"
set RC_TECH_FILE "/path/to/technology/captable"
set MW_REFERENCE_LIB "/path/to/mw_lib"

# Timing constraints from synthesis
set SDC_FILE "${NETLIST_DIR}/${DESIGN_NAME}.sdc"

# ============================================================================
# Library and Design Setup
# ============================================================================

# Create Milkyway design library
create_mw_lib -technology $TECH_FILE \
              -mw_reference_lib $MW_REFERENCE_LIB \
              ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_mw

open_mw_lib ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_mw

# Import gate-level netlist from synthesis
import_designs ${NETLIST_DIR}/${DESIGN_NAME}_synth.v \
               -format verilog \
               -top $DESIGN_NAME

# Read timing constraints
read_sdc $SDC_FILE

# Set TLU+ models for RC extraction
set_tlu_plus_files -max_tluplus ${RC_TECH_FILE}/max.tluplus \
                   -min_tluplus ${RC_TECH_FILE}/min.tluplus \
                   -tech2itf_map ${RC_TECH_FILE}/tech2itf.map

check_library

# ============================================================================
# Floorplan Specifications
# ============================================================================

echo "============================================"
echo "Creating Floorplan"
echo "============================================"

# Die size: 5mm x 5mm
set DIE_WIDTH 5000
set DIE_HEIGHT 5000

# Core utilization: 70% (leaves room for routing)
set CORE_UTILIZATION 0.70

# Core-to-die spacing (um)
set CORE_MARGIN_LEFT 100
set CORE_MARGIN_RIGHT 100
set CORE_MARGIN_TOP 100
set CORE_MARGIN_BOTTOM 100

# Create rectangular floorplan
create_floorplan -core_utilization $CORE_UTILIZATION \
                 -core_aspect_ratio 1.0 \
                 -left_io2core $CORE_MARGIN_LEFT \
                 -right_io2core $CORE_MARGIN_RIGHT \
                 -top_io2core $CORE_MARGIN_TOP \
                 -bottom_io2core $CORE_MARGIN_BOTTOM \
                 -start_first_row

# Alternative: specify explicit die and core dimensions
# create_floorplan -core_width [expr $DIE_WIDTH - $CORE_MARGIN_LEFT - $CORE_MARGIN_RIGHT] \
#                  -core_height [expr $DIE_HEIGHT - $CORE_MARGIN_TOP - $CORE_MARGIN_BOTTOM] \
#                  -die_width $DIE_WIDTH \
#                  -die_height $DIE_HEIGHT \
#                  -left_io2core $CORE_MARGIN_LEFT \
#                  -bottom_io2core $CORE_MARGIN_BOTTOM

# ============================================================================
# Power Grid Design
# ============================================================================

echo "============================================"
echo "Creating Power Grid"
echo "============================================"

# Power/Ground nets
set POWER_NET "VDD"
set GROUND_NET "VSS"

# Define power domains (if using UPF)
# create_power_domain PD_TOP -include_scope

# Create power rings around core
create_rectangular_rings \
    -nets {VDD VSS} \
    -left_offset 5 \
    -right_offset 5 \
    -top_offset 5 \
    -bottom_offset 5 \
    -left_segment_layer METAL5 \
    -left_segment_width 10 \
    -right_segment_layer METAL5 \
    -right_segment_width 10 \
    -top_segment_layer METAL6 \
    -top_segment_width 10 \
    -bottom_segment_layer METAL6 \
    -bottom_segment_width 10

# Create power straps (vertical)
create_power_straps \
    -direction vertical \
    -nets {VDD VSS} \
    -layer METAL5 \
    -width 2 \
    -spacing 2 \
    -start_offset 50 \
    -number_of_straps 20

# Create power straps (horizontal)
create_power_straps \
    -direction horizontal \
    -nets {VDD VSS} \
    -layer METAL6 \
    -width 2 \
    -spacing 2 \
    -start_offset 50 \
    -number_of_straps 20

# Power mesh for uniform distribution
create_power_mesh \
    -nets {VDD VSS} \
    -layers {METAL5 METAL6} \
    -pitch_x 100 \
    -pitch_y 100 \
    -width 2

# Connect power grid
preroute_standard_cells \
    -connect horizontal \
    -port_filter_mode off \
    -cell_master_filter_mode off \
    -cell_instance_filter_mode off \
    -voltage_area_filter_mode off

# ============================================================================
# Macro Placement
# ============================================================================

echo "============================================"
echo "Placing Hard Macros"
echo "============================================"

# Identify macros (memory blocks, large cells)
set MACROS [get_cells -hierarchical -filter "is_hard_macro==true"]

if {[sizeof_collection $MACROS] > 0} {
    echo "Found [sizeof_collection $MACROS] hard macros"
    
    # Example: Place SSI search memory at bottom left
    # set_cell_location -coordinates {200 200} -fixed ssi_search/memory_block
    
    # Example: Place ranker memory at top right
    # set_cell_location -coordinates {4000 4000} -fixed ranker/memory_block
    
    # Auto-place remaining macros
    place_fp_macros -auto_blockages all
    
    # Create blockages around macros for routing
    create_fp_placement_blockage -type hard -bbox {190 190 600 600}
    
} else {
    echo "No hard macros found in design"
}

# ============================================================================
# Pin Placement
# ============================================================================

echo "============================================"
echo "Placing I/O Pins"
echo "============================================"

# Remove any existing pin constraints
remove_pin_constraint -all

# Create pin guides for different sides
# Left side: AXI write channels
set_pin_constraint -side 4 \
                   -allowed_layers {METAL5 METAL6} \
                   -pin_spacing 5 \
                   -ports [get_ports "axi_aw* axi_w* axi_b*"]

# Right side: AXI read channels
set_pin_constraint -side 2 \
                   -allowed_layers {METAL5 METAL6} \
                   -pin_spacing 5 \
                   -ports [get_ports "axi_ar* axi_r*"]

# Top side: Memory interface
set_pin_constraint -side 1 \
                   -allowed_layers {METAL5 METAL6} \
                   -pin_spacing 5 \
                   -ports [get_ports "mem_*"]

# Bottom side: Status and control
set_pin_constraint -side 3 \
                   -allowed_layers {METAL5 METAL6} \
                   -pin_spacing 5 \
                   -ports [get_ports "led_* irq_out"]

# Clock and reset on specific locations
set_individual_pin_constraints -ports $CLK_PORT \
                               -side 1 \
                               -offset 2500 \
                               -allowed_layers {METAL6}

set_individual_pin_constraints -ports $RST_PORT \
                               -side 1 \
                               -offset 2550 \
                               -allowed_layers {METAL6}

# Place pins according to constraints
place_pins -self

# ============================================================================
# Placement Blockages and Keep-Out Regions
# ============================================================================

echo "============================================"
echo "Creating Placement Blockages"
echo "============================================"

# Hard blockage: prevent standard cell placement in specific areas
# create_fp_placement_blockage -type hard -bbox {x1 y1 x2 y2} -name BLOCK_1

# Soft blockage: discourage placement but allow if necessary
# create_fp_placement_blockage -type soft -bbox {x1 y1 x2 y2} -name BLOCK_2

# Partial blockage for specific cell types
# create_fp_placement_blockage -type partial -blocked_percentage 50 \
#                              -bbox {x1 y1 x2 y2} -name BLOCK_3

# ============================================================================
# Virtual Flat Placement
# ============================================================================

echo "============================================"
echo "Virtual Flat Placement (Coarse)"
echo "============================================"

# Initial placement to estimate congestion and timing
create_fp_placement -timing_driven -no_legalize

# ============================================================================
# Congestion Analysis
# ============================================================================

echo "============================================"
echo "Analyzing Routing Congestion"
echo "============================================"

# Route estimation for congestion analysis
route_fp_proto

# Report congestion
report_congestion > ${ICC_OUTPUT_DIR}/congestion.rpt

# Visualize congestion map
set_route_zrt_common_options -congestion_map_output both
set_route_zrt_global_options -congestion_map_effort medium

# ============================================================================
# Timing Analysis (Pre-Placement)
# ============================================================================

echo "============================================"
echo "Pre-Placement Timing Analysis"
echo "============================================"

# Update timing with estimated wire loads
update_timing -full

report_timing -max_paths 10 > ${ICC_OUTPUT_DIR}/timing_preplacement.rpt
report_constraint -all_violators > ${ICC_OUTPUT_DIR}/constraints_preplacement.rpt

# ============================================================================
# Power Planning Verification
# ============================================================================

echo "============================================"
echo "Verifying Power Grid"
echo "============================================"

# Check power grid connectivity
verify_pg_nets -error_view ${DESIGN_NAME}_pg_errors

# Power grid analysis (if license available)
# analyze_power_plan -nets {VDD VSS}

# ============================================================================
# Reports and Output
# ============================================================================

echo "============================================"
echo "Generating Floorplan Reports"
echo "============================================"

report_design -physical > ${ICC_OUTPUT_DIR}/design_physical.rpt
report_utilization > ${ICC_OUTPUT_DIR}/utilization.rpt
report_pin_placement > ${ICC_OUTPUT_DIR}/pin_placement.rpt
report_placement > ${ICC_OUTPUT_DIR}/placement.rpt

# ============================================================================
# Save Floorplan
# ============================================================================

echo "============================================"
echo "Saving Floorplan"
echo "============================================"

save_mw_cel -as ${DESIGN_NAME}_floorplan

# Write DEF file for exchange with other tools
write_def ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_floorplan.def

# Write floorplan script for reuse
write_floorplan -all ${ICC_OUTPUT_DIR}/${DESIGN_NAME}_floorplan.tcl

# ============================================================================
# Summary
# ============================================================================

echo "============================================"
echo "Floorplanning Complete"
echo "============================================"
echo "Outputs written to: ${ICC_OUTPUT_DIR}"
echo ""
echo "Key files:"
echo "  - ${DESIGN_NAME}_floorplan.def     : DEF file"
echo "  - ${DESIGN_NAME}_floorplan.tcl     : Floorplan script"
echo "  - design_physical.rpt              : Physical design report"
echo "  - utilization.rpt                  : Core utilization"
echo "  - congestion.rpt                   : Routing congestion"
echo ""
echo "Next steps:"
echo "  1. Review congestion and timing reports"
echo "  2. Adjust macro placement if needed"
echo "  3. Proceed to placement optimization"
echo "  4. Run detailed routing"
echo ""

# Print utilization summary
report_utilization

exit
