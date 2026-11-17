// urex_top.v -- PL top for the ZCU104 build: block design (PS + AXI interconnect) wired to snn_axi.
// Used by vivado/build2.tcl, which exports the PS AXI-Lite master as an external interface and attaches the SNN
// here in RTL (no IP-integrator module reference; Vivado 2018.2 on some machines cannot resolve those).
`timescale 1ns/1ps
module urex_top;
    wire        clk, rstn, irq;
    wire [31:0] awaddr, wdata, araddr, rdata;
    wire [2:0]  awprot, arprot;
    wire [3:0]  wstrb;
    wire [1:0]  bresp, rresp;
    wire        awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;

    design_1 bd (
        .M_AXI_SNN_awaddr(awaddr),   .M_AXI_SNN_awprot(awprot),  .M_AXI_SNN_awvalid(awvalid), .M_AXI_SNN_awready(awready),
        .M_AXI_SNN_wdata(wdata),     .M_AXI_SNN_wstrb(wstrb),    .M_AXI_SNN_wvalid(wvalid),   .M_AXI_SNN_wready(wready),
        .M_AXI_SNN_bresp(bresp),     .M_AXI_SNN_bvalid(bvalid),  .M_AXI_SNN_bready(bready),
        .M_AXI_SNN_araddr(araddr),   .M_AXI_SNN_arprot(arprot),  .M_AXI_SNN_arvalid(arvalid), .M_AXI_SNN_arready(arready),
        .M_AXI_SNN_rdata(rdata),     .M_AXI_SNN_rresp(rresp),    .M_AXI_SNN_rvalid(rvalid),   .M_AXI_SNN_rready(rready),
        .pl_clk(clk), .pl_resetn(rstn), .irq_done(irq));

    snn_axi #(.LANES(16)) snn (
        .s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr[17:0]), .s_axi_awprot(awprot), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr[17:0]), .s_axi_arprot(arprot), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .irq_done(irq));
endmodule
