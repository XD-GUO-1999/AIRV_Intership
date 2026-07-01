// Copyright 2021 Thales DIS design services SAS
//
// Licensed under the Solderpad Hardware Licence, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
// You may obtain a copy of the License at https://solderpad.org/licenses/
//
// Original Author: Guillaume Chauvon (guillaume.chauvon@thalesgroup.com)
// Example coprocessor adds rs1,rs2(,rs3) together and gives back the result to the CPU via the CoreV-X-Interface.
// Coprocessor delays the sending of the result depending on result least significant bits.

module cvxif_example_coprocessor
  import cvxif_pkg::*;
  import cvxif_instr_pkg::*;
(
    input  logic        clk_i,        // Clock
    input  logic        rst_ni,       // Asynchronous reset active low
    input  cvxif_req_t  cvxif_req_i,
    output cvxif_resp_t cvxif_resp_o
);

  //Compressed interface
  logic               x_compressed_valid_i;
  logic               x_compressed_ready_o;
  x_compressed_req_t  x_compressed_req_i;
  x_compressed_resp_t x_compressed_resp_o;
  //Issue interface
  logic               x_issue_valid_i;
  logic               x_issue_ready_o;
  x_issue_req_t       x_issue_req_i;
  x_issue_resp_t      x_issue_resp_o;
  //Commit interface
  logic               x_commit_valid_i;
  x_commit_t          x_commit_i;
  //Memory interface
  logic               x_mem_valid_o;
  logic               x_mem_ready_i;
  x_mem_req_t         x_mem_req_o;
  x_mem_resp_t        x_mem_resp_i;
  //Memory result interface
  logic               x_mem_result_valid_i;
  x_mem_result_t      x_mem_result_i;
  //Result interface
  logic               x_result_valid_o;
  logic               x_result_ready_i;
  x_result_t          x_result_o;

  assign x_compressed_valid_i            = cvxif_req_i.x_compressed_valid;
  assign x_compressed_req_i              = cvxif_req_i.x_compressed_req;
  assign x_issue_valid_i                 = cvxif_req_i.x_issue_valid;
  assign x_issue_req_i                   = cvxif_req_i.x_issue_req;
  assign x_commit_valid_i                = cvxif_req_i.x_commit_valid;
  assign x_commit_i                      = cvxif_req_i.x_commit;
  assign x_mem_ready_i                   = cvxif_req_i.x_mem_ready;
  assign x_mem_resp_i                    = cvxif_req_i.x_mem_resp;
  assign x_mem_result_valid_i            = cvxif_req_i.x_mem_result_valid;
  assign x_mem_result_i                  = cvxif_req_i.x_mem_result;
  assign x_result_ready_i                = cvxif_req_i.x_result_ready;

  assign cvxif_resp_o.x_compressed_ready = x_compressed_ready_o;
  assign cvxif_resp_o.x_compressed_resp  = x_compressed_resp_o;
  assign cvxif_resp_o.x_issue_ready      = x_issue_ready_o;
  assign cvxif_resp_o.x_issue_resp       = x_issue_resp_o;
  assign cvxif_resp_o.x_mem_valid        = x_mem_valid_o;
  assign cvxif_resp_o.x_mem_req          = x_mem_req_o;
  assign cvxif_resp_o.x_result_valid     = x_result_valid_o;
  assign cvxif_resp_o.x_result           = x_result_o;

  //Compressed interface
  assign x_compressed_ready_o            = '0;
  assign x_compressed_resp_o.instr       = '0;
  assign x_compressed_resp_o.accept      = '0;

  instr_decoder #(
      .NbInstr   (cvxif_instr_pkg::NbInstr),
      .CoproInstr(cvxif_instr_pkg::CoproInstr)
  ) instr_decoder_i (
      .clk_i         (clk_i),
      .x_issue_req_i (x_issue_req_i),
      .x_issue_resp_o(x_issue_resp_o)
  );

  typedef struct packed {
    x_issue_req_t  req;
    x_issue_resp_t resp;
  } x_issue_t;

  logic fifo_full, fifo_empty;
  logic x_issue_ready_q;
  logic instr_push, instr_pop;
  x_issue_t req_i;
  x_issue_t req_o;



  assign instr_push = x_issue_resp_o.accept ? 1 : 0;
  assign instr_pop = (x_commit_i.x_commit_kill && x_commit_valid_i) || (x_result_valid_o && x_result_ready_i);
  assign x_issue_ready_q = ~fifo_full; // if something is in the fifo, the instruction is being processed
                                       // so we can't receive anything else
  assign req_i.req = x_issue_req_i;
  assign req_i.resp = x_issue_resp_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin : regs
    if (!rst_ni) begin
      x_issue_ready_o <= 1;
    end else begin
      x_issue_ready_o <= x_issue_ready_q;
    end
  end

  fifo_v3 #(
      .FALL_THROUGH(1),         //data_o ready and pop in the same cycle
      .DATA_WIDTH  (64),
      .DEPTH       (8),
      .dtype       (x_issue_t)
  ) fifo_commit_i (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .flush_i   (1'b0),
      .testmode_i(1'b0),
      .full_o    (fifo_full),
      .empty_o   (fifo_empty),
      .usage_o   (),
      .data_i    (req_i),
      .push_i    (instr_push),
      .data_o    (req_o),
      .pop_i     (instr_pop)
  );

  logic [3:0] c;
  counter #(
      .WIDTH(4)
  ) counter_i (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .clear_i   (~x_commit_i.x_commit_kill && x_commit_valid_i),
      .en_i      (1'b1),
      .load_i    (),
      .down_i    (),
      .d_i       (),
      .q_o       (c),
      .overflow_o()
  );

// Modification: define the buffer parameters and key control signals.
  localparam int unsigned  INPUT_BUF_WORDS = 100;

  logic [31:0] input_buffer [0:INPUT_BUF_WORDS-1];

  logic [4:0] wr_block_cnt_q;
  logic [4:0] rd_block_cnt_q;
  logic [4:0] active_blocks_q;

  logic [4:0] buf_active_blocks;
  logic [4:0] wr_block_sel;
  logic [6:0] wr_base;
  logic [6:0] rd_base;

  assign buf_active_blocks = req_o.req.instr[11:7] + 5'd1;

  assign wr_base = {wr_block_sel, 2'b00};
  assign rd_base = {rd_block_cnt_q, 2'b00};


  assign is_buf4_ex     = (req_o.req.instr[6:0] == 7'b0101011);
  assign is_mac16buf_ex = (req_o.req.instr[6:0] == 7'b0001011);

  // Modification: handle the FIFO state.
  assign x_result_valid_o = ~fifo_empty && ~x_commit_i.x_commit_kill;

  assign wr_block_sel = (is_buf4_ex && (buf_active_blocks != active_blocks_q)) ? 5'd0 : wr_block_cnt_q;
  // Buffer write state machine (response to buf4)
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_blocks_q <= 5'd1;
      wr_block_cnt_q <= 5'd0;
      rd_block_cnt_q <= 5'd0;

      for (int i = 0; i < INPUT_BUF_WORDS; i++) begin
        input_buffer[i] <= '0;
      end
    end else if (x_result_valid_o && x_result_ready_i) begin
      if(is_buf4_ex) begin
        active_blocks_q <= buf_active_blocks;

        input_buffer[wr_base + 0] <= req_o.req.rs[0]; 
        input_buffer[wr_base + 1] <= req_o.req.rs[1]; 
        input_buffer[wr_base + 2] <= req_o.req.rs[3]; 
        input_buffer[wr_base + 3] <= req_o.req.rs[4]; 

        if (wr_block_sel == (buf_active_blocks - 5'd1)) begin
          wr_block_cnt_q <= 5'd0;
        end else begin
          wr_block_cnt_q <= wr_block_sel + 5'd1;
        end

        if (buf_active_blocks != active_blocks_q) begin
          rd_block_cnt_q <= 5'd0; // Reset read block counter if active blocks change
        end
      end else if (is_mac16buf_ex) begin
        if (rd_block_cnt_q == (active_blocks_q - 5'd1)) begin
          rd_block_cnt_q <= 5'd0;
        end else begin
          rd_block_cnt_q <= rd_block_cnt_q + 5'd1;
        end
      end
    end
  end

  // MAC16 parallel multiply-accumulate logic (response to mac16buf)
  logic signed [31:0] mac_result;
  logic signed [15:0] p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
  logic signed [31:0] input1, input2, input3, input4, weight1, weight2, weight3, weight4, acc;

  always_comb begin
    if (is_mac16buf_ex) begin
      weight1 = $signed(req_o.req.rs[0]);
      weight2 = $signed(req_o.req.rs[1]); 
      acc = $signed(req_o.req.rs[2]); 
      weight3 = $signed(req_o.req.rs[3]); 
      weight4 = $signed(req_o.req.rs[4]); 

      input1 = $signed(input_buffer[rd_base + 0]);
      input2 = $signed(input_buffer[rd_base + 1]);
      input3 = $signed(input_buffer[rd_base + 2]);
      input4 = $signed(input_buffer[rd_base + 3]);

      p0 = $signed({1'b0, input1[7:0]}) * $signed(weight1[7:0]);
      p1 = $signed({1'b0, input1[15:8]}) * $signed(weight1[15:8]);
      p2 = $signed({1'b0, input1[23:16]}) * $signed(weight1[23:16]);
      p3 = $signed({1'b0, input1[31:24]}) * $signed(weight1[31:24]);
      p4 = $signed({1'b0, input2[7:0]}) * $signed(weight2[7:0]);
      p5 = $signed({1'b0, input2[15:8]}) * $signed(weight2[15:8]);
      p6 = $signed({1'b0, input2[23:16]}) * $signed(weight2[23:16]);
      p7 = $signed({1'b0, input2[31:24]}) * $signed(weight2[31:24]);
      p8 = $signed({1'b0, input3[7:0]}) * $signed(weight3[7:0]);
      p9 = $signed({1'b0, input3[15:8]}) * $signed(weight3[15:8]);
      p10 = $signed({1'b0, input3[23:16]}) * $signed(weight3[23:16]);
      p11 = $signed({1'b0, input3[31:24]}) * $signed(weight3[31:24]);
      p12 = $signed({1'b0, input4[7:0]}) * $signed(weight4[7:0]);
      p13 = $signed({1'b0, input4[15:8]}) * $signed(weight4[15:8]);
      p14 = $signed({1'b0, input4[23:16]}) * $signed(weight4[23:16]);
      p15 = $signed({1'b0, input4[31:24]}) * $signed(weight4[31:24]);

      mac_result = acc + 32'(p0)+ 32'(p1)+ 32'(p2)+ 32'(p3) + 32'(p4) + 32'(p5) + 32'(p6) + 32'(p7) + 32'(p8) 
                    + 32'(p9) + 32'(p10) + 32'(p11) + 32'(p12) + 32'(p13) + 32'(p14) + 32'(p15); // we add the result togeter

    end else begin
      mac_result = '0;
    end
  end


  always_comb begin
    x_result_o.data    = mac_result; //modification: send the result of the MAC16 operation to the CPU
    x_result_o.id      = req_o.req.id;
    x_result_o.rd      = req_o.req.instr[11:7];
    x_result_o.we      = req_o.resp.writeback & x_result_valid_o & is_mac16buf_ex; 
    x_result_o.exc     = 0;
    x_result_o.exccode = 0;
  end

endmodule
