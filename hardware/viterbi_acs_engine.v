// File: viterbi_acs_engine.sv
// IEEE 1800-2012 SystemVerilog Compliant
// Sequential Time-Multiplexed Viterbi ACS Engine with Metric Normalization

module viterbi_acs_engine #(
    parameter int NUM_STATES     = 8,        // 8 Signed States (IDLE, W1+, W1-, W2+, W2-, W3+, W3-, REV)
    parameter int METRIC_WIDTH   = 32,       // 32-bit metric precision
    parameter int TRANS_COST_W   = 16        // 16-bit transition cost precision
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         frame_start,
    
    // Emission penalty costs from observation generator [0..NUM_STATES-1]
    input  logic signed [METRIC_WIDTH-1:0] emission_cost [NUM_STATES],
    
    // Transition cost matrix A_cost[prev_state][next_state]
    input  logic signed [TRANS_COST_W-1:0] trans_cost [NUM_STATES][NUM_STATES],
    
    output logic                         frame_done,
    output logic [2:0]                   best_state,
    output logic signed [METRIC_WIDTH-1:0] state_metrics [NUM_STATES]
);

    // Internal FSM States
    typedef enum logic [2:0] {
        ST_IDLE      = 3'b000,
        ST_COMPUTE   = 3'b001,
        ST_FIND_MIN  = 3'b010,
        ST_NORMALIZE = 3'b011,
        ST_DONE      = 3'b100
    } fsm_state_e;

    fsm_state_e current_state, next_state;

    // Loop Counters for Time-Multiplexing
    logic [2:0] curr_s;  // Target next state (0..7)
    logic [2:0] prev_s;  // Candidate previous state (0..7)

    // Intermediate Accumulator Registers
    logic signed [METRIC_WIDTH-1:0] v_prev [NUM_STATES];
    logic signed [METRIC_WIDTH-1:0] v_next [NUM_STATES];
    logic signed [METRIC_WIDTH-1:0] min_candidate;
    logic signed [METRIC_WIDTH-1:0] v_min_frame;
    logic [2:0]                     best_state_reg;

    // -------------------------------------------------------------------------
    // 1. FSM State Sequential Transition
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= ST_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    // -------------------------------------------------------------------------
    // 2. FSM Next-State & Loop Logic
    // -------------------------------------------------------------------------
    always_comb begin
        next_state = current_state;
        case (current_state)
            ST_IDLE: begin
                if (frame_start) next_state = ST_COMPUTE;
            end
            ST_COMPUTE: begin
                if (curr_s == NUM_STATES - 1 && prev_s == NUM_STATES - 1)
                    next_state = ST_FIND_MIN;
            end
            ST_FIND_MIN: begin
                next_state = ST_NORMALIZE;
            end
            ST_NORMALIZE: begin
                next_state = ST_DONE;
            end
            ST_DONE: begin
                next_state = ST_IDLE;
            end
            default: next_state = ST_IDLE;
        endcase
    end

    // -------------------------------------------------------------------------
    // 3. Time-Multiplexed ACS Computation & Overflow Prevention Normalization
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            curr_s         <= '0;
            prev_s         <= '0;
            frame_done     <= 1'b0;
            best_state_reg <= '0;
            v_min_frame    <= '0;
            min_candidate  <= '1; // Set to max positive
            
            for (int i = 0; i < NUM_STATES; i++) begin
                v_prev[i] <= '0;
                v_next[i] <= '0;
            end
        end else begin
            case (current_state)
                ST_IDLE: begin
                    frame_done <= 1'b0;
                    curr_s     <= '0;
                    prev_s     <= '0;
                    if (frame_start) begin
                        min_candidate <= '1; // Initialize candidate comparison
                    end
                end

                // Issue 14 Fix: Compute 1 ACS candidate per clock cycle (64 total cycles)
                ST_COMPUTE: begin
                    automatic logic signed [METRIC_WIDTH-1:0] cand_path;
                    cand_path = v_prev[prev_s] + $signed(trans_cost[prev_s][curr_s]) + emission_cost[curr_s];

                    // Find minimum cost path into curr_s state
                    if (prev_s == 3'd0 || cand_path < min_candidate) begin
                        min_candidate <= cand_path;
                    end

                    // Inner/Outer loop iteration over states
                    if (prev_s == NUM_STATES - 1) begin
                        v_next[curr_s] <= (cand_path < min_candidate) ? cand_path : min_candidate;
                        prev_s         <= '0;
                        min_candidate  <= '1;
                        if (curr_s == NUM_STATES - 1) begin
                            curr_s <= '0;
                        end else begin
                            curr_s <= curr_s + 1'b1;
                        end
                    end else begin
                        prev_s <= prev_s + 1'b1;
                    end
                end

                // Issue 15 Fix (Step 1): Identify minimum metric across all states
                ST_FIND_MIN: begin
                    automatic logic signed [METRIC_WIDTH-1:0] min_val = v_next[0];
                    automatic logic [2:0] min_idx = 3'd0;
                    
                    for (int i = 1; i < NUM_STATES; i++) begin
                        if (v_next[i] < min_val) begin
                            min_val = v_next[i];
                            min_idx = i[2:0];
                        end
                    end
                    v_min_frame    <= min_val;
                    best_state_reg <= min_idx;
                end

                // Issue 15 Fix (Step 2): Subtract minimum from all metric accumulators
                ST_NORMALIZE: begin
                    for (int i = 0; i < NUM_STATES; i++) begin
                        v_prev[i] <= v_next[i] - v_min_frame;
                    end
                end

                ST_DONE: begin
                    frame_done <= 1'b1;
                end
            endcase
        end
    end

    // Output assignments
    assign best_state    = best_state_reg;
    assign state_metrics = v_prev;

endmodule