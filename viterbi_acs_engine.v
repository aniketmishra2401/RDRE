`timescale 1ns / 1ps

module viterbi_acs_engine #(
    parameter MAX_WAVES = 8,
    parameter DATA_WIDTH = 32,
    parameter STATE_WIDTH = 4
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   frame_valid,
    input  wire signed [DATA_WIDTH-1:0] log_A [0:MAX_WAVES-1][0:MAX_WAVES-1],
    input  wire signed [DATA_WIDTH-1:0] log_B [0:MAX_WAVES-1],
    output reg  [STATE_WIDTH-1:0] best_state,
    output reg                    state_valid
);

    reg signed [DATA_WIDTH-1:0] v_scores [0:MAX_WAVES-1];
    reg signed [DATA_WIDTH-1:0] v_scores_next [0:MAX_WAVES-1];
    
    integer j, k;
    reg signed [DATA_WIDTH-1:0] max_val;
    reg [STATE_WIDTH-1:0]       best_prev;
    reg signed [DATA_WIDTH-1:0] candidate_prob;

    localparam signed [DATA_WIDTH-1:0] INITIAL_SCORE = -32'sd532;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < MAX_WAVES; j = j + 1) begin
                v_scores[j] <= INITIAL_SCORE;
            end
            best_state  <= 4'd1;
            state_valid <= 1'b0;
        end else if (frame_valid) begin
            for (j = 0; j < MAX_WAVES; j = j + 1) begin
                max_val = -32'sd2147483648;
                best_prev = 0;
                
                for (k = 0; k < MAX_WAVES; k = k + 1) begin
                    candidate_prob = v_scores[k] + log_A[k][j];
                    if (candidate_prob > max_val) begin
                        max_val = candidate_prob;
                        best_prev = k[STATE_WIDTH-1:0];
                    end
                end
                
                v_scores_next[j] = max_val + log_B[j];
            end

            max_val = -32'sd2147483648;
            for (j = 0; j < MAX_WAVES; j = j + 1) begin
                v_scores[j] <= v_scores_next[j];
                if (v_scores_next[j] > max_val) begin
                    max_val = v_scores_next[j];
                    best_state <= j[STATE_WIDTH-1:0] + 4'd1;
                end
            end
            
            state_valid <= 1'b1;
        end else begin
            state_valid <= 1'b0;
        end
    end

endmodule