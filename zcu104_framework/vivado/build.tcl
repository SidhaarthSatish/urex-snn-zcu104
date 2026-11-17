# build.tcl -- ZCU104 project: Zynq UltraScale+ PS + snn_axi (AXI-Lite, IRQ) on the PL.
# Works with Vivado 2018.2 and newer. Usage from this folder (batch), or in the GUI Tcl console:
#     set argv {16 200}; source build.tcl
# Batch:
#     vivado -mode batch -source build.tcl -tclargs [lanes] [pl_clk_mhz]
# Outputs in ./out: urex_snn.bit, urex_snn.hwh (for PYNQ), urex_snn.xsa (for Vitis/PetaLinux),
# and utilisation, timing and power reports.
# NOT YET RUN: written without Vivado access. Expect to fix small things on the first run.

set LANES      [expr {[llength $argv] > 0 ? [lindex $argv 0] : 16}]
set PL_MHZ     [expr {[llength $argv] > 1 ? [lindex $argv 1] : 200}]
set PART       xczu7ev-ffvc1156-2-e
set BOARD      ""   ;# picked below: newest installed xilinx.com:zcu104 board part
set HERE       [file normalize [file dirname [info script]]]
set PROJ       $HERE/build
set OUT        $HERE/out
file mkdir $OUT

create_project urex_snn $PROJ -part $PART -force
set bp [lsort [get_board_parts -quiet xilinx.com:zcu104:*]]
if {[llength $bp] > 0} {
    set BOARD [lindex $bp end]
    set_property board_part $BOARD [current_project]
    puts "Using board part $BOARD"
} else {
    puts "WARNING: no ZCU104 board part installed; using bare part $PART (PS presets will be generic)"
}
add_files -norecurse [list $HERE/../rtl/snn_core.v $HERE/../rtl/snn_axi.v]
update_compile_order -fileset sources_1

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $PL_MHZ \
] $ps

set snn [create_bd_cell -type module -reference snn_axi snn_0]
set_property -dict [list CONFIG.LANES $LANES CONFIG.NH 128 CONFIG.T 10] $snn

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
    Clk_master {/zynq_ultra_ps_e_0/pl_clk0 } Clk_slave {/zynq_ultra_ps_e_0/pl_clk0 } Clk_xbar {/zynq_ultra_ps_e_0/pl_clk0 } \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/snn_0/s_axi} ddr_seg {Auto} intc_ip {New AXI Interconnect} master_apm {0}] \
    [get_bd_intf_pins snn_0/s_axi]
connect_bd_net [get_bd_pins snn_0/irq_done] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] -filter {NAME =~ *snn_0*}]
set_property offset 0xA0000000 $seg
set_property range 256K $seg
validate_bd_design
save_bd_design

set wrap [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrap
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

open_run impl_1
report_utilization        -hierarchical -file $OUT/utilisation_hier.rpt
report_utilization                      -file $OUT/utilisation.rpt
report_timing_summary                   -file $OUT/timing.rpt
report_power                            -file $OUT/power_vectorless.rpt

file copy -force $PROJ/urex_snn.runs/impl_1/design_1_wrapper.bit $OUT/urex_snn.bit
# hand-off file lives in urex_snn.srcs (Vivado <= 2019.1) or urex_snn.gen (2019.2+)
set hwh [glob -nocomplain $PROJ/urex_snn.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh $PROJ/urex_snn.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh]
if {[llength $hwh] > 0} { file copy -force [lindex $hwh 0] $OUT/urex_snn.hwh } else { puts "WARNING: no .hwh found" }
if {[llength [info commands write_hw_platform]] > 0} {
    write_hw_platform -fixed -include_bit -force -file $OUT/urex_snn.xsa
} else {
    write_hwdef -force -file $OUT/urex_snn.hdf      ;# Vivado 2018.x
}
puts "DONE: bitstream, hwh and hand-off file in $OUT"
