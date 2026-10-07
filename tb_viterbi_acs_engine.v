`timescale 1ns / 1ps

module tb_viterbi_acs_engine;

    parameter MAX_WAVES = 8;
    parameter DATA_WIDTH = 32;
    parameter STATE_WIDTH = 4;
    parameter NUM_FRAMES = 20001;

    reg clk;
    reg rst_n;
    reg frame_valid;
    reg signed [DATA_WIDTH-1:0] log_A [0:MAX_WAVES-1][0:MAX_WAVES-1];
    reg signed [DATA_WIDTH-1:0] log_B [0:MAX_WAVES-1];
    
    wire [STATE_WIDTH-1:0] best_state;
    wire state_valid;

    // Test Vector Memories
    reg signed [DATA_WIDTH-1:0] mem_A [0:63];
    reg signed [DATA_WIDTH-1:0] mem_B [0:(MAX_WAVES*NUM_FRAMES)-1];
    reg [15:0] mem_G [0:NUM_FRAMES-1];

    integer i, j, frame_idx, errors;

    // Instantiate Unit Under Test (UUT)
    viterbi_acs_engine #(
        .MAX_WAVES(MAX_WAVES),
        .DATA_WIDTH(DATA_WIDTH),
        .STATE_WIDTH(STATE_WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .frame_valid(frame_valid),
        .log_A(log_A),
        .log_B(log_B),
        .best_state(best_state),
        .state_valid(state_valid)
    );

    // Clock Generator (100 MHz -> 10ns period)
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        frame_valid = 0;
        errors = 0;

        // Load Hex Test Vectors into Memory
        $readmemh("test_vectors/transition_matrix_A.hex", mem_A);
        $readmemh("test_vectors/emission_matrix_B.hex", mem_B);
        $readmemh("test_vectors/golden_states.hex", mem_G);

        // Map flat transition memory to 2D array
        for (i = 0; i < MAX_WAVES; i = i + 1) begin
            for (j = 0; j < MAX_WAVES; j = j + 1) begin
                log_A[i][j] = mem_A[i*MAX_WAVES + j];
            end
        end

        #20 rst_n = 1;
        #10;

        // Cycle through all exported frames
        for (frame_idx = 0; frame_idx < NUM_FRAMES; frame_idx = frame_idx + 1) begin
            // Load current frame emissions
            for (i = 0; i < MAX_WAVES; i = i + 1) begin
                log_B[i] = mem_B[frame_idx*MAX_WAVES + i];
            end

            frame_valid = 1;
            #10; // Clock tick
            frame_valid = 0;

            // Assert matching hardware state against golden state
            if (best_state !== mem_G[frame_idx][STATE_WIDTH-1:0]) begin
                $display("[ERROR] Frame %0d: Expected State %0d, Got State %0d", 
                         frame_idx, mem_G[frame_idx], best_state);
                errors = errors + 1;
            end
        end

        if (errors == 0) begin
            $display("\n==================================================");
            $display(" VERILOG CO-SIMULATION PASSED (0 Errors / %0d Frames)", NUM_FRAMES);
            $display("==================================================\n");
        end else begin
            $display("\n[FAILED] Test finished with %0d errors.", errors);
        end

        $finish;
    end

endmodule