# build2.tcl -- ZCU104 build WITHOUT IP-integrator module references (works around Vivado installs whose
# hierarchy parser cannot resolve snn_axi as a module). The BD contains PS + reset + AXI interconnect and exports
# an AXI4-Lite master (M_AXI_SNN), pl_clk, pl_resetn and an irq input; rtl/urex_top.v wires snn_axi to them.
# Vivado 2018.2 or newer. Batch:  vivado -mode batch -source build2.tcl -tclargs [lanes_unused] [pl_clk_mhz]
set PL_MHZ [expr {[llength $argv] > 1 ? [lindex $argv 1] : 150}]
set PART   xczu7ev-ffvc1156-2-e
set HERE   [file normalize [file dirname [info script]]]
set PROJ   $HERE/build2
set OUT    $HERE/out2
file mkdir $OUT

create_project urex_snn2 $PROJ -part $PART -force
set bp [lsort [get_board_parts -quiet xilinx.com:zcu104:*]]
if {[llength $bp] > 0} { set_property board_part [lindex $bp end] [current_project]; puts "Using board part [lindex $bp end]" } else { puts "WARNING: no ZCU104 board part installed; PS preset will be generic, do NOT use this bitstream on a board" }
add_files -norecurse [list $HERE/../rtl/snn_core.v $HERE/../rtl/snn_axi.v $HERE/../rtl/urex_top.v]

create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__IRQ0 {1} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $PL_MHZ] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_0]
set ic  [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect ic_0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ic

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins rst_0/slowest_sync_clk] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins ic_0/ACLK] [get_bd_pins ic_0/S00_ACLK] [get_bd_pins ic_0/M00_ACLK]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_0/ext_reset_in]
connect_bd_net [get_bd_pins rst_0/interconnect_aresetn] [get_bd_pins ic_0/ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins ic_0/S00_ARESETN] [get_bd_pins ic_0/M00_ARESETN]
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins ic_0/S00_AXI]

# exported ports
set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_SNN]
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.FREQ_HZ [expr {int($PL_MHZ * 1000000)}]] $m
connect_bd_intf_net [get_bd_intf_pins ic_0/M00_AXI] $m
set pclk [create_bd_port -dir O -type clk pl_clk]
set_property CONFIG.ASSOCIATED_BUSIF {M_AXI_SNN} $pclk
set prst [create_bd_port -dir O -type rst pl_resetn]
set irqp [create_bd_port -dir I -from 0 -to 0 irq_done]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] $pclk
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] $prst
connect_bd_net $irqp [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

assign_bd_address
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ultra_ps_e_0/Data]] {
    if {[string match *SNN* $seg]} { set_property offset 0xA0000000 $seg; set_property range 256K $seg; puts "SNN segment: $seg" }
}
validate_bd_design
save_bd_design
generate_target all [get_files design_1.bd]

set_property source_mgmt_mode DisplayOnly [current_project]
set_property top urex_top [current_fileset]
launch_runs synth_1 -jobs 2
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
open_run impl_1
report_utilization -file $OUT/utilisation.rpt
report_utilization -hierarchical -file $OUT/utilisation_hier.rpt
report_timing_summary -file $OUT/timing.rpt
report_power -file $OUT/power_vectorless.rpt
file copy -force $PROJ/urex_snn2.runs/impl_1/urex_top.bit $OUT/urex_snn.bit
# hand-off file lives in urex_snn2.srcs (Vivado <= 2019.1) or urex_snn2.gen (2019.2+)
set hwh [glob -nocomplain $PROJ/urex_snn2.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh $PROJ/urex_snn2.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh]
if {[llength $hwh] > 0} { file copy -force [lindex $hwh 0] $OUT/urex_snn.hwh } else { puts "WARNING: no .hwh found" }
if {[llength [info commands write_hw_platform]] > 0} { write_hw_platform -fixed -include_bit -force -file $OUT/urex_snn.xsa } else { write_hwdef -force -file $OUT/urex_snn.hdf }
puts "DONE2: outputs in $OUT"
