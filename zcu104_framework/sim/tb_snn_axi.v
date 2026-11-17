// Testbench: loads the exported INT8 weights over AXI-Lite, runs every golden patch and
// compares score, decision with python/snn_int.py. Reports cycles per patch.
`timescale 1ns/1ps
module tb_snn_axi;
    parameter LANES = 16;
    reg clk = 0, rstn = 0;
    always #2 clk = ~clk;                       // 250 MHz
    reg  [17:0] awaddr, araddr; reg awvalid=0, wvalid=0, bready=1, arvalid=0, rready=1;
    reg  [31:0] wdata;
    wire awready, wready, bvalid, arready, rvalid, irq; wire [31:0] rdata; wire [1:0] bresp, rresp;
    snn_axi #(.LANES(LANES)) dut(.s_axi_aclk(clk), .s_axi_aresetn(rstn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'd0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'd0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready), .irq_done(irq));

    task axi_write(input [17:0] a, input [31:0] d); begin
        @(posedge clk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1;
        @(posedge clk); while (!(awready && wready)) @(posedge clk);
        awvalid <= 0; wvalid <= 0;
        while (!bvalid) @(posedge clk);
        @(posedge clk);
    end endtask
    task axi_read(input [17:0] a, output [31:0] d); begin
        @(posedge clk); araddr <= a; arvalid <= 1;
        @(posedge clk); while (!arready) @(posedge clk);
        arvalid <= 0;
        while (!rvalid) @(posedge clk);
        d = rdata; @(posedge clk);
    end endtask

    reg [31:0] wv [0:40000];
    reg [31:0] pv [0:40000];
    integer nw, np, i, j, p, n, pos, errs, total_cyc, maxcyc, total_ev;
    reg [31:0] r, acc, dec, cyc;
    initial begin
        $readmemh("vectors/wvec.mem", wv);
        $readmemh("vectors/patches.mem", pv);
        if (!$value$plusargs("NW=%d", nw)) nw = 16769;
        if (!$value$plusargs("NP=%d", np)) np = 1511;
        repeat (5) @(posedge clk); rstn = 1; repeat (2) @(posedge clk);
        axi_read(18'h20, r); $display("ID = %h", r);
        axi_read(18'h24, r); $display("PARAMS = %h", r);
        for (i = 0; i < nw; i = i + 1) axi_write(wv[2*i][17:0], wv[2*i+1]);
        $display("weights loaded (%0d writes)", nw);
        errs = 0; pos = 0; total_cyc = 0; maxcyc = 0; total_ev = 0;
        for (p = 0; p < np; p = p + 1) begin
            n = pv[pos]; pos = pos + 1;
            axi_write(18'h00, 32'h2);                         // clear FIFO
            for (j = 0; j < n; j = j + 1) axi_write(18'h0C, pv[pos + j]);
            pos = pos + n;
            axi_write(18'h00, 32'h1);                         // start
            r = 0; while (r[1:0] != 2'b10) axi_read(18'h00, r);
            axi_read(18'h10, acc); axi_read(18'h14, dec); axi_read(18'h1C, cyc);
            if (acc !== pv[pos] || dec[0] !== pv[pos+1][0]) begin
                errs = errs + 1;
                if (errs < 10) $display("MISMATCH patch %0d n=%0d: acc %0d expected %0d, dec %0d expected %0d",
                                        p, n, $signed(acc), $signed(pv[pos]), dec[0], pv[pos+1][0]);
            end
            pos = pos + 2; total_cyc = total_cyc + cyc; total_ev = total_ev + n;
            if (cyc > maxcyc) maxcyc = cyc;
        end
        $display("RESULT patches=%0d mismatches=%0d mean_cycles=%0d max_cycles=%0d mean_events=%0d",
                 np, errs, total_cyc / np, maxcyc, total_ev / np);
        if (errs == 0) $display("PASS"); else $display("FAIL");
        $finish;
    end
endmodule
