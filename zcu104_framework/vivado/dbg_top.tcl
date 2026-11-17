# dbg_top.tcl -- why does Vivado not see snn_axi as a module? Prints find_top for variants.
set HERE [file normalize [file dirname [info script]]]
create_project -in_memory -part xczu7ev-ffvc1156-2-e
add_files -norecurse $HERE/../rtl/snn_core.v
update_compile_order -fileset sources_1
puts "DBG core only: top=[get_property TOP [current_fileset]] find_top=[find_top -quiet]"
add_files -norecurse $HERE/../rtl/snn_axi.v
update_compile_order -fileset sources_1
puts "DBG core+axi : top=[get_property TOP [current_fileset]] find_top=[find_top -quiet]"
puts "DBG files: [get_files]"
puts "DBG types: [get_property FILE_TYPE [get_files]]"
foreach f [get_files] { puts "DBG $f used_in=[get_property USED_IN $f] enabled=[get_property IS_ENABLED $f]" }
catch {report_compile_order -sources} r; puts "DBG order: $r"
exit
