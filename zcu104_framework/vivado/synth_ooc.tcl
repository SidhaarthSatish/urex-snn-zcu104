# synth_ooc.tcl -- quick out-of-context synthesis of the SNN core alone (no PS) for resource and timing estimates.
# Usage: vivado -mode batch -source synth_ooc.tcl -tclargs [lanes] [clk_mhz]
set LANES  [expr {[llength $argv] > 0 ? [lindex $argv 0] : 16}]
set MHZ    [expr {[llength $argv] > 1 ? [lindex $argv 1] : 250}]
set HERE   [file normalize [file dirname [info script]]]
file mkdir $HERE/out
read_verilog [list $HERE/../rtl/snn_core.v $HERE/../rtl/snn_axi.v]
synth_design -top snn_axi -part xczu7ev-ffvc1156-2-e -mode out_of_context -generic LANES=$LANES
create_clock -period [expr {1000.0 / $MHZ}] [get_ports s_axi_aclk]
opt_design; place_design; route_design
report_utilization    -file $HERE/out/ooc_util_L$LANES.rpt
report_timing_summary -file $HERE/out/ooc_timing_L${LANES}_${MHZ}MHz.rpt
puts "WNS: [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
