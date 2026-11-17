// snn_core.v -- event-driven FC-only LIF SNN classifier (512 -> 128 LIF -> 1 score)
//
// Bit-exact with python/snn_int.py:
//   for k in 0..T-1:
//     m_j = ((m_j * LEAK_NUM) >>> LEAK_SHIFT) + B_j                  (LEAK phase)
//     m_j += W[addr][j]  for every event (bin == k, addr) in the FIFO  (ACCUM phase)
//     s_j = m_j > TH_j ; m_j -= s_j ? TH_j : 0 ; acc += s_j ? WD_j : 0 (FIRE phase)
//   decision = acc >= acc_th
//
// LANES neurons are processed per cycle; G = NH/LANES groups.
// Cycles per patch ~= T*(2*G + 3) + n_events*G + 2.
// Events must be pushed in non-decreasing bin order (the driver sorts them).
// An event whose bin is lower than the current bin is still consumed in the current bin.
`timescale 1ns/1ps
module snn_core #(
    parameter NIN        = 512,
    parameter NH         = 128,
    parameter T          = 10,
    parameter LANES      = 16,
    parameter MEMW       = 20,     // membrane width (observed max |m| = 10.5k; +-524k allowed)
    parameter ACCW       = 20,     // score accumulator width (observed |acc| < 37k)
    parameter FIFO_AW    = 13,     // 8192-entry event FIFO (>= 5120 possible voxels)
    parameter LEAK_NUM   = 205,
    parameter LEAK_SHIFT = 8
)(
    input  wire                 clk,
    input  wire                 rstn,
    // parameter writes (only while idle)
    input  wire                 cfg_we,
    input  wire [1:0]           cfg_sel,     // 0: W1 word, 1: bias, 2: threshold, 3: output weight
    input  wire [13:0]          cfg_idx,     // W1: word index (addr*NH/4 + j/4); others: neuron j
    input  wire [31:0]          cfg_wdata,
    // event FIFO
    input  wire                 ev_we,
    input  wire [12:0]          ev_wdata,    // {bin[3:0], addr[8:0]}
    input  wire                 fifo_clear,
    output wire [FIFO_AW:0]     fifo_count,
    output wire                 fifo_full,
    // control / result
    input  wire                 start,
    input  wire signed [31:0]   acc_th,
    output reg                  busy,
    output reg                  done,
    output reg  signed [31:0]   acc_out,
    output reg                  decision,
    output reg  [31:0]          cycles
);
    localparam G   = NH / LANES;
    localparam GW  = (G > 1) ? $clog2(G) : 1;
    localparam W1D = NIN * G;
    localparam W1AW = $clog2(W1D);
    localparam signed [10:0] LEAKC = LEAK_NUM;

    // ---------------- parameter memories ----------------
    reg [LANES*8-1:0]  w1   [0:W1D-1];     // BRAM, byte write enables
    (* ram_style = "distributed" *) reg [LANES*16-1:0] bias [0:G-1];
    (* ram_style = "distributed" *) reg [LANES*16-1:0] thr  [0:G-1];
    (* ram_style = "distributed" *) reg [LANES*8-1:0]  wd   [0:G-1];
    reg [LANES*MEMW-1:0] mem [0:G-1];      // membranes

    // ---- control state: declared before first use (portable style).
    localparam S_IDLE = 3'd0, S_CLR = 3'd1, S_LEAK = 3'd2, S_ACC = 3'd3, S_DRAIN = 3'd4, S_FIRE = 3'd5, S_DONE = 3'd6, S_FLUSH = 3'd7;
    reg [2:0]    st;
    reg [GW-1:0] g;
    reg [3:0]    k;
    wire         fire_v = (st == S_FIRE);

    // W1 write: word w covers neurons j0..j0+3 of input addr a
    wire [13:0] w_a  = cfg_idx / (NH/4);
    wire [13:0] w_j0 = (cfg_idx % (NH/4)) * 4;
    wire [W1AW-1:0] w_row = w_a * G + w_j0 / LANES;
    wire [13:0] w_lane0 = w_j0 % LANES;
    wire [13:0] n_row  = cfg_idx / LANES;
    wire [13:0] n_lane = cfg_idx % LANES;

    // byte-enable write: LANES/4 copies of the 32-bit word, enable the 4 target bytes
    wire [LANES*8-1:0] w_din = {(LANES/4){cfg_wdata}};
    wire [LANES-1:0]   w_be  = ({{(LANES-4){1'b0}}, 4'b1111} << w_lane0);
    wire               w_we  = cfg_we && (cfg_sel == 2'd0);
    integer i;
    always @(posedge clk) begin
        for (i = 0; i < LANES; i = i + 1)
            if (w_we && w_be[i]) w1[w_row][i*8 +: 8] <= w_din[i*8 +: 8];
    end
    always @(posedge clk) begin
        if (cfg_we && cfg_sel == 2'd1) bias[n_row][n_lane*16 +: 16] <= cfg_wdata[15:0];
        if (cfg_we && cfg_sel == 2'd2) thr [n_row][n_lane*16 +: 16] <= cfg_wdata[15:0];
        if (cfg_we && cfg_sel == 2'd3) wd  [n_row][n_lane*8  +: 8 ] <= cfg_wdata[7:0];
    end

    // ---------------- event FIFO (first-word fall-through) ----------------
    (* ram_style = "block" *) reg [12:0] fifo [0:(1<<FIFO_AW)-1];
    reg [FIFO_AW:0] wp, rp;
    reg pop;
    wire fifo_empty = (wp == rp);
    assign fifo_count = wp - rp;
    assign fifo_full  = (fifo_count == (1 << FIFO_AW));
    // registered look-ahead read so the FIFO maps to block RAM: head_q always holds fifo[rp]
    // (events are written while the core is idle, so there is no read-during-write case).
    reg  [12:0] head_q;
    wire [FIFO_AW:0] rp_next = rp + 1'b1;
    always @(posedge clk) head_q <= fifo[(pop && !fifo_empty) ? rp_next[FIFO_AW-1:0] : rp[FIFO_AW-1:0]];
    wire [12:0] head = head_q;
    wire [3:0]  head_bin  = head[12:9];
    wire [8:0]  head_addr = head[8:0];
    always @(posedge clk) begin
        if (!rstn || fifo_clear) begin
            wp <= 0; rp <= 0;
        end else begin
            if (ev_we && !fifo_full) begin
                fifo[wp[FIFO_AW-1:0]] <= ev_wdata;
                wp <= wp + 1'b1;
            end
            if (pop && !fifo_empty) rp <= rp + 1'b1;
        end
    end

    // ---------------- datapath helpers ----------------
    // leak + bias for one group
    function [LANES*MEMW-1:0] f_leak;
        input [LANES*MEMW-1:0] m;
        input [LANES*16-1:0]   b;
        integer l;
        reg signed [MEMW-1:0]   ml;
        reg signed [MEMW+9:0]   prod;
        reg signed [MEMW-1:0]   r;
        begin
            for (l = 0; l < LANES; l = l + 1) begin
                ml   = m[l*MEMW +: MEMW];
                prod = ml * LEAKC;
                r    = prod >>> LEAK_SHIFT;
                f_leak[l*MEMW +: MEMW] = r + {{(MEMW-16){b[l*16+15]}}, b[l*16 +: 16]};
            end
        end
    endfunction
    // add one weight row
    function [LANES*MEMW-1:0] f_acc;
        input [LANES*MEMW-1:0] m;
        input [LANES*8-1:0]    w;
        integer l;
        reg signed [MEMW-1:0] ml;
        begin
            for (l = 0; l < LANES; l = l + 1) begin
                ml = m[l*MEMW +: MEMW];
                f_acc[l*MEMW +: MEMW] = ml + {{(MEMW-8){w[l*8+7]}}, w[l*8 +: 8]};
            end
        end
    endfunction

    // fire: combinational for the current group
    // One generate block per lane (a loop with shared blocking temporaries in an always @* was
    // synthesised by Vivado 2018.2 as lane 0 only: lanes 1..LANES-1 were trimmed away).
    wire [LANES*MEMW-1:0] fire_m;
    wire [LANES*8-1:0]    fire_c;          // per-lane score contribution (WD if the neuron fired, else 0)
    wire [LANES*MEMW-1:0] mem_g = mem[g];
    wire [LANES*16-1:0]   thr_g = thr[g];
    wire [LANES*8-1:0]    wd_g  = wd[g];
    genvar gl;
    generate for (gl = 0; gl < LANES; gl = gl + 1) begin : FIRE
        wire signed [MEMW-1:0] mf    = mem_g[gl*MEMW +: MEMW];
        wire signed [MEMW-1:0] tf    = {{(MEMW-16){thr_g[gl*16+15]}}, thr_g[gl*16 +: 16]};
        wire                   fires = (mf > tf);
        assign fire_m[gl*MEMW +: MEMW] = fires ? (mf - tf) : mf;
        assign fire_c[gl*8 +: 8]       = fires ? wd_g[gl*8 +: 8] : 8'd0;
    end endgenerate

    // score pipeline: lane contributions -> partial sums of 4 lanes -> total -> acc   (3 stages)
    localparam NP = LANES / 4;
    reg [LANES*8-1:0]  fc_q;   reg fc_v;
    reg [NP*11-1:0]    ps_q;   reg ps_v;
    reg signed [14:0]  sm_q;   reg sm_v;
    integer p1, q1;
    reg signed [10:0] t4; reg signed [14:0] ttl;
    always @(posedge clk) begin
        if (!rstn) begin fc_v <= 0; ps_v <= 0; sm_v <= 0; fc_q <= 0; ps_q <= 0; sm_q <= 0; end
        else begin
            fc_q <= fire_c;   fc_v <= fire_v;
            for (p1 = 0; p1 < NP; p1 = p1 + 1) begin
                t4 = 0;
                for (q1 = 0; q1 < 4; q1 = q1 + 1) t4 = t4 + $signed(fc_q[(p1*4+q1)*8 +: 8]);
                ps_q[p1*11 +: 11] <= t4;
            end
            ps_v <= fc_v;
            ttl = 0;
            for (p1 = 0; p1 < NP; p1 = p1 + 1) ttl = ttl + $signed(ps_q[p1*11 +: 11]);
            sm_q <= ttl; sm_v <= ps_v;
        end
    end

    // ---------------- control FSM ----------------
    // pipelined leak: result registered, written back one cycle later (different group, no hazard)
    reg [LANES*MEMW-1:0] lk_q; reg lk_v; reg [GW-1:0] lk_g;
    reg [1:0] flush_cnt;
    reg signed [ACCW-1:0] acc;
    // W1 read pipeline
    reg [W1AW-1:0] rd_addr;
    reg [LANES*8-1:0] rd_data;
    reg rd_v, rd_v_d;
    reg [GW-1:0] g_d, g_dd;
    always @(posedge clk) begin
        rd_data <= w1[rd_addr];
    end

    wire issue = (st == S_ACC) && !fifo_empty && (head_bin <= k);
    always @* begin
        pop = issue && (g == G-1);
    end

    always @(posedge clk) begin
        if (!rstn) begin
            st <= S_IDLE; busy <= 0; done <= 0; acc_out <= 0; decision <= 0; cycles <= 0;
            g <= 0; k <= 0; acc <= 0; rd_v <= 0; rd_v_d <= 0; g_d <= 0; g_dd <= 0; rd_addr <= 0; lk_v <= 0; lk_g <= 0; lk_q <= 0; flush_cnt <= 0;
        end else begin
            // read pipeline: issue (cycle 0) -> rd_addr registered -> rd_data registered -> accumulate
            rd_v   <= issue;
            g_d    <= g;
            rd_v_d <= rd_v;
            g_dd   <= g_d;
            if (issue) rd_addr <= head_addr * G + g;
            if (rd_v_d) mem[g_dd] <= f_acc(mem[g_dd], rd_data);
            if (busy) cycles <= cycles + 1;
            lk_v <= 0;
            if (lk_v) mem[lk_g] <= lk_q;
            if (sm_v) acc <= acc + sm_q;

            case (st)
            S_IDLE: if (start) begin
                busy <= 1; done <= 0; cycles <= 0; acc <= 0; k <= 0; g <= 0; st <= S_CLR;
            end
            S_CLR: begin                       // zero membranes
                mem[g] <= 0;
                if (g == G-1) begin g <= 0; st <= S_LEAK; end else g <= g + 1'b1;
            end
            S_LEAK: begin
                lk_q <= f_leak(mem[g], bias[g]); lk_v <= 1; lk_g <= g;
                if (g == G-1) begin g <= 0; st <= S_ACC; end else g <= g + 1'b1;
            end
            S_ACC: begin
                if (issue) begin
                    g <= (g == G-1) ? 0 : g + 1'b1;
                end else begin
                    g <= 0; st <= S_DRAIN;
                end
            end
            S_DRAIN: begin                     // wait for in-flight weight rows
                if (!rd_v && !rd_v_d) st <= S_FIRE;
            end
            S_FIRE: begin
                mem[g] <= fire_m;
                if (g == G-1) begin
                    g <= 0;
                    if (k == T-1) begin st <= S_FLUSH; flush_cnt <= 0; end
                    else begin k <= k + 1'b1; st <= S_LEAK; end
                end else g <= g + 1'b1;
            end
            S_FLUSH: begin                     // let the 3-stage score pipeline drain into acc
                flush_cnt <= flush_cnt + 1'b1;
                if (flush_cnt == 2'd3) st <= S_DONE;
            end
            S_DONE: begin
                acc_out  <= {{(32-ACCW){acc[ACCW-1]}}, acc};
                decision <= ($signed({{(32-ACCW){acc[ACCW-1]}}, acc}) >= acc_th);
                busy <= 0; done <= 1; st <= S_IDLE;
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
