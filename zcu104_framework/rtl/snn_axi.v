// snn_axi.v -- AXI4-Lite slave wrapper around snn_core (connect to the ZynqMP M_AXI_HPM0_FPD/LPD)
//
// Register map (byte offsets):
//   0x000 CTRL     W: bit0 start, bit1 clear event FIFO      R: bit0 busy, bit1 done
//   0x004 STATUS   R: [13:0] FIFO count, bit31 FIFO full
//   0x00C EVENT    W: [12:0] = {bin[3:0], addr[8:0]}  push one event (addr = pol*256 + ys*16 + xs)
//   0x010 ACC      R: signed score of the last patch
//   0x014 DECISION R: bit0 = (ACC >= ACC_TH)
//   0x018 ACC_TH   RW: signed decision threshold
//   0x01C CYCLES   R: clock cycles taken by the last patch
//   0x020 ID       R: 0x534E4E01 ("SNN" v1)
//   0x024 PARAMS   R: {P_LANES, P_T, P_NH}
//   0x10000-0x1FFFF  W1 (write only): word (a*NH/4 + j/4) holds W[a][j..j+3], byte 0 = neuron j
//   0x20000 + 4j   bias B[j]   (int16, write only)
//   0x20400 + 4j   threshold TH[j] (int16, write only)
//   0x20800 + 4j   output weight WD[j] (int8, write only)
`timescale 1ns/1ps
module snn_axi #(
    parameter C_S_AXI_ADDR_WIDTH = 18,
    parameter LANES = 16,
    parameter NH = 128,
    parameter T = 10
)(
    input  wire                          s_axi_aclk,
    input  wire                          s_axi_aresetn,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  wire [2:0]                    s_axi_awprot,
    input  wire                          s_axi_awvalid,
    output reg                           s_axi_awready,
    input  wire [31:0]                   s_axi_wdata,
    input  wire [3:0]                    s_axi_wstrb,
    input  wire                          s_axi_wvalid,
    output reg                           s_axi_wready,
    output wire [1:0]                    s_axi_bresp,
    output reg                           s_axi_bvalid,
    input  wire                          s_axi_bready,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  wire [2:0]                    s_axi_arprot,
    input  wire                          s_axi_arvalid,
    output reg                           s_axi_arready,
    output reg  [31:0]                   s_axi_rdata,
    output wire [1:0]                    s_axi_rresp,
    output reg                           s_axi_rvalid,
    input  wire                          s_axi_rready,
    output wire                          irq_done            // level: high when a result is ready
);
    wire clk = s_axi_aclk, rstn = s_axi_aresetn;
    localparam [7:0] P_LANES = LANES, P_T = T;
    localparam [15:0] P_NH = NH;
    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    // ---------- write channel: take AW and W together ----------
    reg        wr_fire;
    reg [C_S_AXI_ADDR_WIDTH-1:0] wa;
    reg [31:0] wdat;
    always @(posedge clk) begin
        if (!rstn) begin
            s_axi_awready <= 0; s_axi_wready <= 0; s_axi_bvalid <= 0; wr_fire <= 0; wa <= 0; wdat <= 0;
        end else begin
            wr_fire <= 0;
            if (!s_axi_awready && !s_axi_bvalid && s_axi_awvalid && s_axi_wvalid) begin
                s_axi_awready <= 1; s_axi_wready <= 1;
                wa <= s_axi_awaddr; wdat <= s_axi_wdata; wr_fire <= 1;
            end else begin
                s_axi_awready <= 0; s_axi_wready <= 0;
            end
            if (wr_fire) s_axi_bvalid <= 1;
            else if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 0;
        end
    end

    // ---------- decode ----------
    wire is_w1  = wr_fire && (wa[17:16] == 2'b01);
    wire is_par = wr_fire && (wa[17:16] == 2'b10);
    wire [1:0] par_sel = wa[11:10];                       // 0: bias, 1: threshold, 2: output weight
    reg  [31:0] acc_th;
    wire start      = wr_fire && (wa[17:0] == 18'h0) && wdat[0];
    wire fifo_clear = wr_fire && (wa[17:0] == 18'h0) && wdat[1];
    wire ev_we      = wr_fire && (wa[17:0] == 18'h0C);
    always @(posedge clk) begin
        if (!rstn) acc_th <= 0;
        else if (wr_fire && wa[17:0] == 18'h18) acc_th <= wdat;
    end

    wire        cfg_we  = is_w1 || (is_par && par_sel != 2'd3);
    wire [1:0]  cfg_sel = is_w1 ? 2'd0 : (par_sel + 2'd1);
    wire [13:0] cfg_idx = is_w1 ? wa[15:2] : {6'd0, wa[9:2]};

    wire        busy, done, decision, fifo_full;
    wire [13:0] fifo_count;
    wire [31:0] acc, cycles;
    snn_core #(.NH(NH), .T(T), .LANES(LANES)) core (
        .clk(clk), .rstn(rstn),
        .cfg_we(cfg_we), .cfg_sel(cfg_sel), .cfg_idx(cfg_idx), .cfg_wdata(wdat),
        .ev_we(ev_we), .ev_wdata(wdat[12:0]), .fifo_clear(fifo_clear), .fifo_count(fifo_count), .fifo_full(fifo_full),
        .start(start), .acc_th(acc_th), .busy(busy), .done(done), .acc_out(acc), .decision(decision), .cycles(cycles));
    assign irq_done = done;

    // ---------- read channel ----------
    always @(posedge clk) begin
        if (!rstn) begin
            s_axi_arready <= 0; s_axi_rvalid <= 0; s_axi_rdata <= 0;
        end else begin
            if (!s_axi_arready && !s_axi_rvalid && s_axi_arvalid) begin
                s_axi_arready <= 1;
                s_axi_rvalid  <= 1;
                case (s_axi_araddr[17:0])
                    18'h000: s_axi_rdata <= {30'd0, done, busy};
                    18'h004: s_axi_rdata <= {fifo_full, 17'd0, fifo_count};
                    18'h010: s_axi_rdata <= acc;
                    18'h014: s_axi_rdata <= {31'd0, decision};
                    18'h018: s_axi_rdata <= acc_th;
                    18'h01C: s_axi_rdata <= cycles;
                    18'h020: s_axi_rdata <= 32'h534E4E01;
                    18'h024: s_axi_rdata <= {P_LANES, P_T, P_NH};
                    default: s_axi_rdata <= 32'hDEADBEEF;
                endcase
            end else begin
                s_axi_arready <= 0;
                if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 0;
            end
        end
    end
endmodule
