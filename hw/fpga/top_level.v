// ============================================================================
// JAIDE v40 Top-Level FPGA Module
// ============================================================================
// Description: Top-level integration module for JAIDE hardware accelerators
//              Includes AXI-Lite slave interface for CPU communication
// Target:      iCE40-HX8K FPGA
// Clock:       100 MHz
// ============================================================================

module top_level (
    // Clock and Reset
    input wire clk,
    input wire rst_n,
    
    // AXI-Lite Slave Interface (32-bit)
    // Write Address Channel
    input  wire [15:0] axi_awaddr,
    input  wire        axi_awvalid,
    output wire        axi_awready,
    input  wire [2:0]  axi_awprot,
    
    // Write Data Channel
    input  wire [31:0] axi_wdata,
    input  wire [3:0]  axi_wstrb,
    input  wire        axi_wvalid,
    output wire        axi_wready,
    
    // Write Response Channel
    output wire [1:0]  axi_bresp,
    output wire        axi_bvalid,
    input  wire        axi_bready,
    
    // Read Address Channel
    input  wire [15:0] axi_araddr,
    input  wire        axi_arvalid,
    output wire        axi_arready,
    input  wire [2:0]  axi_arprot,
    
    // Read Data Channel
    output wire [31:0] axi_rdata,
    output wire [1:0]  axi_rresp,
    output wire        axi_rvalid,
    input  wire        axi_rready,
    
    // Memory Interface
    output wire [31:0] mem_addr,
    output wire [15:0] mem_wdata,
    input  wire [15:0] mem_rdata,
    output wire        mem_we,
    output wire        mem_oe,
    output wire        mem_ce,
    input  wire        mem_ready,
    
    // Status and Debug
    output wire [7:0]  led_status,
    output wire        led_error,
    output wire        irq_out
);

    // ========================================================================
    // Internal Signals
    // ========================================================================
    
    wire reset;
    assign reset = !rst_n;
    
    // AXI-Lite register interface
    reg [31:0] control_reg;
    reg [31:0] status_reg;
    reg [31:0] config_reg;
    reg [31:0] result_reg;
    
    // SSI Search signals
    wire [63:0] ssi_search_key;
    wire [31:0] ssi_root_addr;
    wire        ssi_start;
    wire [31:0] ssi_result_addr;
    wire        ssi_found;
    wire [7:0]  ssi_depth;
    wire        ssi_done;
    
    // Ranker signals
    wire [63:0] ranker_query_hash;
    wire [63:0] ranker_segment_id;
    wire [63:0] ranker_segment_pos;
    wire [31:0] ranker_base_score;
    wire        ranker_valid;
    wire [31:0] ranker_final_score;
    wire [15:0] ranker_rank;
    wire        ranker_done;
    
    // Memory arbiter signals
    wire [31:0] arbiter_mem_addr;
    wire [15:0] arbiter_mem_wdata;
    wire        arbiter_mem_we;
    wire        arbiter_mem_req;
    wire        arbiter_grant;
    
    // Client request signals (4 clients: SSI, Ranker, CPU, Reserved)
    wire [3:0]  client_req;
    wire [3:0]  client_grant;
    
    // ========================================================================
    // AXI-Lite Slave State Machine
    // ========================================================================
    
    localparam ADDR_CONTROL   = 16'h0000;
    localparam ADDR_STATUS    = 16'h0004;
    localparam ADDR_CONFIG    = 16'h0008;
    localparam ADDR_RESULT    = 16'h000C;
    localparam ADDR_SSI_KEY_L = 16'h0010;
    localparam ADDR_SSI_KEY_H = 16'h0014;
    localparam ADDR_SSI_ROOT  = 16'h0018;
    localparam ADDR_SSI_RES   = 16'h001C;
    localparam ADDR_RNK_HASH_L= 16'h0020;
    localparam ADDR_RNK_HASH_H= 16'h0024;
    localparam ADDR_RNK_SEG_L = 16'h0028;
    localparam ADDR_RNK_SEG_H = 16'h002C;
    localparam ADDR_RNK_POS_L = 16'h0030;
    localparam ADDR_RNK_POS_H = 16'h0034;
    localparam ADDR_RNK_SCORE = 16'h0038;
    localparam ADDR_RNK_RES   = 16'h003C;
    
    reg [1:0] axi_wr_state;
    reg [1:0] axi_rd_state;
    
    localparam AXI_IDLE  = 2'b00;
    localparam AXI_ADDR  = 2'b01;
    localparam AXI_DATA  = 2'b10;
    localparam AXI_RESP  = 2'b11;
    
    reg [15:0] wr_addr_reg;
    reg [15:0] rd_addr_reg;
    reg [31:0] rd_data_reg;
    
    reg axi_awready_reg;
    reg axi_wready_reg;
    reg axi_bvalid_reg;
    reg axi_arready_reg;
    reg axi_rvalid_reg;
    reg [1:0] axi_bresp_reg;
    reg [1:0] axi_rresp_reg;
    
    assign axi_awready = axi_awready_reg;
    assign axi_wready  = axi_wready_reg;
    assign axi_bvalid  = axi_bvalid_reg;
    assign axi_bresp   = axi_bresp_reg;
    assign axi_arready = axi_arready_reg;
    assign axi_rvalid  = axi_rvalid_reg;
    assign axi_rdata   = rd_data_reg;
    assign axi_rresp   = axi_rresp_reg;
    
    // SSI search registers
    reg [63:0] ssi_key_reg;
    reg [31:0] ssi_root_reg;
    reg [31:0] ssi_result_reg;
    
    // Ranker registers
    reg [63:0] ranker_hash_reg;
    reg [63:0] ranker_seg_reg;
    reg [63:0] ranker_pos_reg;
    reg [31:0] ranker_score_reg;
    reg [31:0] ranker_result_reg;
    
    // Write State Machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            axi_wr_state <= AXI_IDLE;
            axi_awready_reg <= 1'b0;
            axi_wready_reg <= 1'b0;
            axi_bvalid_reg <= 1'b0;
            axi_bresp_reg <= 2'b00;
            wr_addr_reg <= 16'h0;
            control_reg <= 32'h0;
            config_reg <= 32'h0;
            ssi_key_reg <= 64'h0;
            ssi_root_reg <= 32'h0;
            ranker_hash_reg <= 64'h0;
            ranker_seg_reg <= 64'h0;
            ranker_pos_reg <= 64'h0;
            ranker_score_reg <= 32'h0;
        end else begin
            case (axi_wr_state)
                AXI_IDLE: begin
                    axi_bvalid_reg <= 1'b0;
                    if (axi_awvalid && axi_wvalid) begin
                        axi_awready_reg <= 1'b1;
                        axi_wready_reg <= 1'b1;
                        wr_addr_reg <= axi_awaddr;
                        axi_wr_state <= AXI_DATA;
                    end
                end
                
                AXI_DATA: begin
                    axi_awready_reg <= 1'b0;
                    axi_wready_reg <= 1'b0;
                    
                    case (wr_addr_reg)
                        ADDR_CONTROL: control_reg <= axi_wdata;
                        ADDR_CONFIG: config_reg <= axi_wdata;
                        ADDR_SSI_KEY_L: ssi_key_reg[31:0] <= axi_wdata;
                        ADDR_SSI_KEY_H: ssi_key_reg[63:32] <= axi_wdata;
                        ADDR_SSI_ROOT: ssi_root_reg <= axi_wdata;
                        ADDR_RNK_HASH_L: ranker_hash_reg[31:0] <= axi_wdata;
                        ADDR_RNK_HASH_H: ranker_hash_reg[63:32] <= axi_wdata;
                        ADDR_RNK_SEG_L: ranker_seg_reg[31:0] <= axi_wdata;
                        ADDR_RNK_SEG_H: ranker_seg_reg[63:32] <= axi_wdata;
                        ADDR_RNK_POS_L: ranker_pos_reg[31:0] <= axi_wdata;
                        ADDR_RNK_POS_H: ranker_pos_reg[63:32] <= axi_wdata;
                        ADDR_RNK_SCORE: ranker_score_reg <= axi_wdata;
                    endcase
                    
                    axi_bresp_reg <= 2'b00;
                    axi_bvalid_reg <= 1'b1;
                    axi_wr_state <= AXI_RESP;
                end
                
                AXI_RESP: begin
                    if (axi_bready) begin
                        axi_bvalid_reg <= 1'b0;
                        axi_wr_state <= AXI_IDLE;
                    end
                end
                
                default: axi_wr_state <= AXI_IDLE;
            endcase
        end
    end
    
    // Read State Machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            axi_rd_state <= AXI_IDLE;
            axi_arready_reg <= 1'b0;
            axi_rvalid_reg <= 1'b0;
            axi_rresp_reg <= 2'b00;
            rd_addr_reg <= 16'h0;
            rd_data_reg <= 32'h0;
        end else begin
            case (axi_rd_state)
                AXI_IDLE: begin
                    axi_rvalid_reg <= 1'b0;
                    if (axi_arvalid) begin
                        axi_arready_reg <= 1'b1;
                        rd_addr_reg <= axi_araddr;
                        axi_rd_state <= AXI_DATA;
                    end
                end
                
                AXI_DATA: begin
                    axi_arready_reg <= 1'b0;
                    
                    case (rd_addr_reg)
                        ADDR_CONTROL: rd_data_reg <= control_reg;
                        ADDR_STATUS: rd_data_reg <= status_reg;
                        ADDR_CONFIG: rd_data_reg <= config_reg;
                        ADDR_RESULT: rd_data_reg <= result_reg;
                        ADDR_SSI_KEY_L: rd_data_reg <= ssi_key_reg[31:0];
                        ADDR_SSI_KEY_H: rd_data_reg <= ssi_key_reg[63:32];
                        ADDR_SSI_ROOT: rd_data_reg <= ssi_root_reg;
                        ADDR_SSI_RES: rd_data_reg <= ssi_result_reg;
                        ADDR_RNK_HASH_L: rd_data_reg <= ranker_hash_reg[31:0];
                        ADDR_RNK_HASH_H: rd_data_reg <= ranker_hash_reg[63:32];
                        ADDR_RNK_SEG_L: rd_data_reg <= ranker_seg_reg[31:0];
                        ADDR_RNK_SEG_H: rd_data_reg <= ranker_seg_reg[63:32];
                        ADDR_RNK_POS_L: rd_data_reg <= ranker_pos_reg[31:0];
                        ADDR_RNK_POS_H: rd_data_reg <= ranker_pos_reg[63:32];
                        ADDR_RNK_SCORE: rd_data_reg <= ranker_score_reg;
                        ADDR_RNK_RES: rd_data_reg <= ranker_result_reg;
                        default: rd_data_reg <= 32'hDEADBEEF;
                    endcase
                    
                    axi_rresp_reg <= 2'b00;
                    axi_rvalid_reg <= 1'b1;
                    axi_rd_state <= AXI_RESP;
                end
                
                AXI_RESP: begin
                    if (axi_rready) begin
                        axi_rvalid_reg <= 1'b0;
                        axi_rd_state <= AXI_IDLE;
                    end
                end
                
                default: axi_rd_state <= AXI_IDLE;
            endcase
        end
    end
    
    // ========================================================================
    // Control Logic
    // ========================================================================
    
    assign ssi_search_key = ssi_key_reg;
    assign ssi_root_addr = ssi_root_reg;
    assign ssi_start = control_reg[0];
    
    assign ranker_query_hash = ranker_hash_reg;
    assign ranker_segment_id = ranker_seg_reg;
    assign ranker_segment_pos = ranker_pos_reg;
    assign ranker_base_score = ranker_score_reg;
    assign ranker_valid = control_reg[1];
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            ssi_result_reg <= 32'h0;
            ranker_result_reg <= 32'h0;
            status_reg <= 32'h0;
        end else begin
            if (ssi_done) begin
                ssi_result_reg <= ssi_result_addr;
                status_reg[0] <= ssi_found;
                status_reg[15:8] <= ssi_depth;
            end
            
            if (ranker_done) begin
                ranker_result_reg <= ranker_final_score;
                status_reg[1] <= 1'b1;
                status_reg[31:16] <= ranker_rank;
            end
        end
    end
    
    // ========================================================================
    // Module Instantiations
    // ========================================================================
    
    // Note: These are placeholders for Clash-generated Verilog modules
    // The actual instantiation will happen after Clash compilation
    
    // SSI Search Accelerator
    // SSISearch_topEntity ssi_search (
    //     .clk(clk),
    //     .rst(reset),
    //     .enable(1'b1),
    //     .searchRequest_key(ssi_search_key),
    //     .searchRequest_root(ssi_root_addr),
    //     .searchRequest_valid(ssi_start),
    //     .nodeData(mem_rdata),
    //     .nodeValid(mem_ready),
    //     .memAddr(/* connect to arbiter */),
    //     .resultAddr(ssi_result_addr),
    //     .resultFound(ssi_found),
    //     .resultDepth(ssi_depth),
    //     .resultValid(ssi_done)
    // );
    
    // Ranker Core
    // RankerCore_topEntity ranker (
    //     .clk(clk),
    //     .rst(reset),
    //     .enable(1'b1),
    //     .queryHash(ranker_query_hash),
    //     .segmentID(ranker_segment_id),
    //     .segmentPos(ranker_segment_pos),
    //     .baseScore(ranker_base_score),
    //     .inputValid(ranker_valid),
    //     .finalScore(ranker_final_score),
    //     .rank(ranker_rank),
    //     .outputValid(ranker_done)
    // );
    
    // Memory Arbiter
    // MemoryArbiter_topEntity mem_arbiter (
    //     .clk(clk),
    //     .rst(reset),
    //     .enable(1'b1),
    //     .client0_req(client_req[0]),
    //     .client1_req(client_req[1]),
    //     .client2_req(client_req[2]),
    //     .client3_req(client_req[3]),
    //     .client0_grant(client_grant[0]),
    //     .client1_grant(client_grant[1]),
    //     .client2_grant(client_grant[2]),
    //     .client3_grant(client_grant[3]),
    //     .memAddr(arbiter_mem_addr),
    //     .memWData(arbiter_mem_wdata),
    //     .memWE(arbiter_mem_we),
    //     .memReq(arbiter_mem_req)
    // );
    
    // ========================================================================
    // Memory Interface Assignment
    // ========================================================================
    
    assign mem_addr = arbiter_mem_addr;
    assign mem_wdata = arbiter_mem_wdata;
    assign mem_we = arbiter_mem_we;
    assign mem_oe = !arbiter_mem_we && arbiter_mem_req;
    assign mem_ce = arbiter_mem_req;
    
    // ========================================================================
    // Status and Debug
    // ========================================================================
    
    assign led_status = {
        ssi_done,
        ranker_done,
        arbiter_mem_req,
        mem_ready,
        client_grant[3:0]
    };
    
    assign led_error = (status_reg[0] == 1'b0) && ssi_done;
    
    assign irq_out = ssi_done || ranker_done;

endmodule
