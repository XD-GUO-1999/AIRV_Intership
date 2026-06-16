set_property SRC_FILE_INFO {cfile:/home/imtuser/AIRV/cva6_mac8im_a1/corev_apu/fpga/xilinx/xlnx_clk_gen/xlnx_clk_gen.gen/sources_1/ip/xlnx_clk_gen/xlnx_clk_gen.xdc rfile:../corev_apu/fpga/xilinx/xlnx_clk_gen/xlnx_clk_gen.gen/sources_1/ip/xlnx_clk_gen/xlnx_clk_gen.xdc id:1 order:EARLY scoped_inst:i_xlnx_clk_gen/inst} [current_design]
set_property SRC_FILE_INFO {cfile:/home/imtuser/AIRV/cva6_mac8im_a1/corev_apu/fpga/constraints/cva6_fpga.xdc rfile:../corev_apu/fpga/constraints/cva6_fpga.xdc id:2} [current_design]
current_instance i_xlnx_clk_gen/inst
set_property src_info {type:SCOPED_XDC file:1 line:54 export:INPUT save:INPUT read:READ} [current_design]
set_input_jitter [get_clocks -of_objects [get_ports clk_in1]] 0.050
current_instance
set_property src_info {type:XDC file:2 line:4 export:INPUT save:INPUT read:READ} [current_design]
set_input_jitter tck 1.000
set_property src_info {type:XDC file:2 line:5 export:INPUT save:INPUT read:READ} [current_design]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets tck_IBUF]
set_property src_info {type:XDC file:2 line:14 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_src/data_src_q_reg*/C] -to [get_pins i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_dst/data_dst_q_reg*/D] 20.000
set_property src_info {type:XDC file:2 line:15 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_src/req_src_q_reg/C] -to [get_pins i_dmi_jtag/i_dmi_cdc/i_cdc_resp/i_dst/req_dst_q_reg/D] 20.000
set_property src_info {type:XDC file:2 line:16 export:INPUT save:INPUT read:READ} [current_design]
set_max_delay -datapath_only -from [get_pins i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_dst/ack_dst_q_reg/C] -to [get_pins i_dmi_jtag/i_dmi_cdc/i_cdc_req/i_src/ack_src_q_reg/D] 20.000
set_property src_info {type:XDC file:2 line:20 export:INPUT save:INPUT read:READ} [current_design]
set_multicycle_path -hold -from [get_pins {i_rstgen_main/i_rstgen_bypass/synch_regs_q_reg[3]/C}] 3
