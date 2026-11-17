# dbg_synth.tcl -- synthesis only; prints which register groups survived. Usage: vivado -mode batch -source dbg_synth.tcl -tclargs [lanes]
set LANES [expr {[llength $argv] > 0 ? [lindex $argv 0] : 16}]
set HERE  [file normalize [file dirname [info script]]]
file mkdir $HERE/out
read_verilog [list $HERE/../rtl/snn_core.v $HERE/../rtl/snn_axi.v]
synth_design -top snn_axi -part xczu7ev-ffvc1156-2-e -mode out_of_context -generic LANES=$LANES
set f [open $HERE/out/dbg_survivors.txt w]
foreach pat {mem_reg acc_reg fc_q_reg ps_q_reg sm_q_reg lk_q_reg rd_data_reg rd_addr_reg st_reg g_reg k_reg fifo wp_reg rp_reg cycles_reg decision_reg acc_out_reg lk_v_reg rd_v_reg rd_v_d_reg sm_v_reg} {
    set c [get_cells -quiet -hier -filter "NAME =~ *${pat}*"]
    puts $f "$pat : [llength $c] cells"
}
set all [get_cells -hier -filter {IS_PRIMITIVE}]
puts $f "total primitives: [llength $all]"
foreach c [get_cells -hier -filter {REF_NAME == FDRE}] { puts $f "FDRE [get_property NAME $c]" }
close $f
write_verilog -force $HERE/out/dbg_netlist.v
puts "DBG-SYNTH-DONE"
