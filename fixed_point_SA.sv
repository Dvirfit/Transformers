`include "double_adder.sv"
`include "double_multiplier.sv"
// Unified Self-Attention Module and Testbench with Scaling and Softmax
// Self-Attention Module
module self_attention #(
    parameter DATA_WIDTH = 32,
    parameter Qint = 5,
    parameter Qfrac = 10,
    parameter SEQ_LEN = 6,
    parameter EMBED_DIM = 8,
    parameter DK = 24,
    parameter DV = 24,
    parameter SCALE_FACTOR = 16'h00D1
)(
    input  logic [DATA_WIDTH-1:0] x [0:SEQ_LEN-1][0:EMBED_DIM-1],
    input  logic [DATA_WIDTH-1:0] W_query [0:EMBED_DIM-1][0:DK-1],
    input  logic [DATA_WIDTH-1:0] W_key   [0:EMBED_DIM-1][0:DK-1],
    input  logic [DATA_WIDTH-1:0] W_value [0:EMBED_DIM-1][0:DV-1],
    output logic [DATA_WIDTH-1:0] output_data [0:SEQ_LEN-1][0:DV-1]

);

//add a clock

/*
==========================================================================
Floating point addition handler
==========================================================================
*/
    // Declare signals (example widths for IEEE 754 double)
    logic [63:0] adder_input_a, adder_input_b, adder_output_z;
    logic adder_input_a_stb, adder_input_b_stb;
    logic adder_output_z_ack, adder_output_z_stb;
    logic adder_input_a_ack, adder_input_b_ack;
    logic clk, adder_rst;

    // Instantiate 32-bit floating point adder
    double_adder u_double_adder (
        .input_a       (adder_input_a),
        .input_b       (adder_input_b),
        .input_a_stb   (adder_input_a_stb),
        .input_b_stb   (adder_input_b_stb),
        .output_z_ack  (adder_output_z_ack),
        .clk           (clk),
        .rst           (adder_rst),
        .output_z      (adder_output_z),
        .output_z_stb  (adder_output_z_stb),
        .input_a_ack   (adder_input_a_ack),
        .input_b_ack   (adder_input_b_ack)
    );

    task automatic perform_addition(
    input  logic [63:0] a,
    input  logic [63:0] b,
    output logic [63:0] result
    );
        // Drive inputs
        adder_input_a     = a;
        adder_input_b     = b;
        adder_input_a_stb = 1;
        adder_input_b_stb = 1;

        // Wait until adder accepts the inputs
        wait (adder_input_a_ack && adder_input_b_ack);
        @(posedge clk);
        adder_input_a_stb = 0;
        adder_input_b_stb = 0;

        // Wait for result to be ready
        wait (adder_output_z_stb);
        result = adder_output_z;

        // Acknowledge result
        adder_output_z_ack = 1;
        @(posedge clk);
        adder_output_z_ack = 0;
    endtask


/*
==========================================================================
Floating point multiplication handler
==========================================================================
*/
    // Declare signals (example widths for IEEE 754 single-precision)
    logic [63:0] multiplier_input_a, multiplier_input_b, multiplier_output_z;
    logic multiplier_input_a_stb, multiplier_input_b_stb;
    logic multiplier_output_z_ack, multiplier_output_z_stb;
    logic multiplier_input_a_ack, multiplier_input_b_ack;
    logic multiplier_rst;

    // Instantiate 32-bit floating point multiplier
    double_multiplier u_double_multiplier (
        .input_a       (multiplier_input_a),
        .input_b       (multiplier_input_b),
        .input_a_stb   (multiplier_input_a_stb),
        .input_b_stb   (multiplier_input_b_stb),
        .output_z_ack  (multiplier_output_z_ack),
        .clk           (clk),
        .rst           (multiplier_rst),
        .output_z      (multiplier_output_z),
        .output_z_stb  (multiplier_output_z_stb),
        .input_a_ack   (multiplier_input_a_ack),
        .input_b_ack   (multiplier_input_b_ack)
    );

    task automatic perform_multiplication(
        input  logic [63:0] a,
        input  logic [63:0] b,
        output logic [63:0] result
    );
        // Drive inputs
        multiplier_input_a     = a;
        multiplier_input_b     = b;
        multiplier_input_a_stb = 1;
        multiplier_input_b_stb = 1;

        // Wait until multiplier accepts the inputs
        wait (multiplier_input_a_ack && multiplier_input_b_ack);
        @(posedge clk);
        multiplier_input_a_stb = 0;
        multiplier_input_b_stb = 0;

        // Wait for result to be ready
        wait (multiplier_output_z_stb);
        result = multiplier_output_z;

        // Acknowledge result
        multiplier_output_z_ack = 1;
        @(posedge clk);
        multiplier_output_z_ack = 0;
    endtask







    

    // Intermediate arrays
    logic [DATA_WIDTH-1:0] Q [0:SEQ_LEN-1][0:DK-1];         // Query matrix
    logic [DATA_WIDTH-1:0] K [0:SEQ_LEN-1][0:DK-1];         // Key matrix
    logic [DATA_WIDTH-1:0] V [0:SEQ_LEN-1][0:DV-1];         // Value matrix
    logic [DATA_WIDTH-1:0] scores [0:SEQ_LEN-1][0:SEQ_LEN-1];         // Attention scores (wider for accumulation)
    logic [DATA_WIDTH-1:0] attention [0:SEQ_LEN-1][0:SEQ_LEN-1]; // Attention weights

    //multiply and truncate 2 Qi.f fixed point representation signed numbers as described in project book
    function automatic logic signed [DATA_WIDTH-1:0] mul_and_trunc (
        input logic signed [DATA_WIDTH-1:0] val1,
        input logic signed [DATA_WIDTH-1:0] val2
    );
        logic signed [2 * (Qint + Qfrac + 1) - 1:0] product;
        logic signed [(Qint + Qfrac):0] trunc;
        begin
            product = val1 * val2;
            trunc = {product[(((Qint + Qfrac + 1) * 2) - 1)], product[(((2 * Qfrac) + Qint) - 1):Qfrac]};
            return trunc;
        end
    endfunction

    // Exponential approximation function using LUT in Q5.10 format and linear interpolation (e^x is convex)
    function automatic logic signed [DATA_WIDTH-1:0] exp_approx(input logic signed [DATA_WIDTH-1:0] val);
        // LUT for x = -5 to 4, in Q5.10 format (scaled by 1024)
        logic [DATA_WIDTH-1:0] lut [0:9] = '{
            16'sh0007, // e^-5 ≈ 0.0067 * 1024 ≈ 7
            16'sh0013, // e^-4 ≈ 0.0183 * 1024 ≈ 19
            16'sh0033, // e^-3 ≈ 0.0498 * 1024 ≈ 51
            16'sh008B, // e^-2 ≈ 0.1353 * 1024 ≈ 139
            16'sh0178, // e^-1 ≈ 0.3679 * 1024 ≈ 376
            16'sh0400, // e^0 = 1.0 * 1024 = 1024
            16'sh0AE0, // e^1 ≈ 2.718 * 1024 ≈ 2784
            16'sh1D8A, // e^2 ≈ 7.389 * 1024 ≈ 7562
            16'sh5058, // e^3 ≈ 20.086 * 1024 ≈ 20568
            16'shDA39  // e^4 ≈ 54.598 * 1024 ≈ 55889
        };
        logic signed [DATA_WIDTH-1:0] int_part = {val[15:10], 10'b0}; // Extract integer part (Q5.10)
        logic signed [DATA_WIDTH-1:0] frac = {{6{val[15]}}, val[9:0]}; // Fractional part (lower 10 bits)
        if (int_part < -5) begin
            return 16'h0000; // Approximate e^x ≈ 0 for x < -5
        end else if (int_part >= 4) begin
            return lut[9]; // Cap at e^4 for x > 4
        end else begin
            int index = {{16{int_part[15]}}, int_part} + 5; // Map -5 to 0, ..., 4 to 9
            logic signed [DATA_WIDTH-1:0] lut_val = lut[index];
            logic signed [DATA_WIDTH-1:0] lut_next = lut[index + 1];
            logic signed [DATA_WIDTH-1:0] delta = lut_next - lut_val; // Difference for interpolation
            logic signed [DATA_WIDTH-1:0] increment = mul_and_trunc(delta, frac);
            logic signed [DATA_WIDTH-1:0] approx = lut_val + increment; // Interpolated value
            return approx; // Zero-extend to 32 bits
        end
    endfunction

    // Self-attention computation
    always_comb begin
        // Declare loop variables outside the for loops to avoid Verilator errors
        int i, j, k;
        logic [DATA_WIDTH-1:0] temp_Q, temp_K, temp_V, temp_attention, sum_exp;
        logic [DATA_WIDTH-1:0] exp_vals [0:SEQ_LEN-1];

        // Step 1: Compute Q, K, V matrices
        for (i = 0; i < SEQ_LEN; i++) begin
            for (k = 0; k < DK; k++) begin
                Q[i][k] = 0;
                K[i][k] = 0;
                for (j = 0; j < EMBED_DIM; j++) begin
                    temp_Q = mul_and_trunc(x[i][j], W_query[j][k]);
                    Q[i][k] = Q[i][k] + temp_Q;
                    temp_K = mul_and_trunc(x[i][j], W_key[j][k]);
                    K[i][k] = K[i][k] + temp_K;
                end
            end
            for (k = 0; k < DV; k++) begin
                V[i][k] = 0;
                for (j = 0; j < EMBED_DIM; j++) begin
                    temp_V = mul_and_trunc(x[i][j], W_value[j][k]);
                    V[i][k] = V[i][k] + temp_V;
                end
            end
        end

        // Step 2: Compute scaled attention scores: Q*K'*(1/sqrt(DK)) with additional scaling
        for (i = 0; i < SEQ_LEN; i++) begin
            for (j = 0; j < SEQ_LEN; j++) begin
                scores[i][j] = 0;
                for (k = 0; k < DK; k++) begin
                    scores[i][j] = scores[i][j] + mul_and_trunc(Q[i][k], K[j][k]); //switched indices for K transpose
                end
                scores[i][j] = scores[i][j] >>> 8; // Additional scaling by 1/256
            end
        end

        // Step 3: Apply softmax to get attention weights: sigma(Q*K'/sqrt(DK))
        for (i = 0; i < SEQ_LEN; i++) begin
            sum_exp = 0;
            for (j = 0; j < SEQ_LEN; j++) begin
                exp_vals[j] = exp_approx(scores[i][j]);
                sum_exp = sum_exp + exp_vals[j];
            end
            for (j = 0; j < SEQ_LEN; j++) begin
                temp_attention = (exp_vals[j] << 10) / (sum_exp != 0 ? sum_exp : 1);
                attention[i][j] = temp_attention[15:0]; //repair scaling
            end
        end

        // Step 4: Compute output using attention weights: sigma(QK'/sqrt(DK))*V
        for (i = 0; i < SEQ_LEN; i++) begin
            for (k = 0; k < DV; k++) begin
                output_data[i][k] = 0;
                for (j = 0; j < SEQ_LEN; j++) begin
                    output_data[i][k] = output_data[i][k] + mul_and_trunc(attention[i][j], V[j][k]);
                end           
            end
        end
    end
endmodule

// Testbench Module
module tb_self_attention;
    // Parameters
    parameter DATA_WIDTH = 16;
    parameter SEQ_LEN = 6;
    parameter EMBED_DIM = 8;  //was 16
    parameter DK = 24;
    parameter DV = 24;
    parameter SCALE_FACTOR = 16'h00D1; // Q5.10 format: 1 / sqrt(24) ≈ 0.2041 * 1024 ≈ 209

    // Signals
    logic [DATA_WIDTH-1:0] x [0:SEQ_LEN-1][0:EMBED_DIM-1];
    logic [DATA_WIDTH-1:0] W_query [0:EMBED_DIM-1][0:DK-1];
    logic [DATA_WIDTH-1:0] W_key   [0:EMBED_DIM-1][0:DK-1];
    logic [DATA_WIDTH-1:0] W_value [0:EMBED_DIM-1][0:DV-1];
    logic [DATA_WIDTH-1:0] output_data [0:SEQ_LEN-1][0:DV-1];

    // Instantiate the DUT
    self_attention #(
    .DATA_WIDTH(16),
    .SEQ_LEN(6),
    .EMBED_DIM(8),
    .DK(24),
    .DV(24),
    .SCALE_FACTOR(16'h00D1)
    ) dut (
    .x(x),  // input data
    .W_query(W_query),
    .W_key(W_key),
    .W_value(W_value),
    .output_data(output_data)
    );

    task print_matrix(input logic [DATA_WIDTH-1:0] matrix [0:EMBED_DIM-1][0:DK-1], input string name);
    $display("%s:", name);
    $write("[");
    for (int i = 0; i < EMBED_DIM; i++) begin
        if (i > 0) $write(",\n ");
        else $write("\n ");
        $write("[");
        for (int j = 0; j < DK; j++) begin
            $write("%04h", matrix[i][j]);
            if (j < DK-1) $write(", ");
        end
        $write("]");
    end
    $write("\n]");
    $display("");
    endtask

    task print_value_matrix(input logic [DATA_WIDTH-1:0] matrix [0:EMBED_DIM-1][0:DV-1], input string name);
    $display("%s:", name);
    $write("[");
    for (int i = 0; i < EMBED_DIM; i++) begin
        if (i > 0) $write(",\n ");
        else $write("\n ");
        $write("[");
        for (int j = 0; j < DV; j++) begin
            $write("%04h", matrix[i][j]);
            if (j < DV-1) $write(", ");
        end
        $write("]");
    end
    $write("\n]");
    $display("");
    endtask

    task print_output_data(input logic [DATA_WIDTH-1:0] matrix [0:SEQ_LEN-1][0:DV-1], input string name);
    $display("%s:", name);
    $write("[");
    for (int i = 0; i < SEQ_LEN; i++) begin
        if (i > 0) $write(",\n ");
        else $write("\n ");
        $write("[");
        for (int j = 0; j < DV; j++) begin
            $write("%04h", matrix[i][j]);
            if (j < DV-1) $write(", ");
        end
        $write("]");
    end
    $write("\n]");
    $display("");
    endtask

    // Test stimulus
    initial begin
        // Initialize input with specific 6x9 matrix in Q5.10 format
        x = '{
        '{16'sh01FD, 16'shFF72, 16'sh0297, 16'sh0618, 16'shFF10, 16'shFF10, 16'sh0651, 16'sh0312},
        '{16'shFE1F, 16'sh022C, 16'shFE25, 16'shFE23, 16'sh00F8, 16'shF859, 16'shF91A, 16'shFDC0},
        '{16'shFBF3, 16'sh0142, 16'shFC5E, 16'shFA5A, 16'sh05DD, 16'shFF19, 16'sh0045, 16'shFA4D},
        '{16'shFDD3, 16'sh0072, 16'shFB65, 16'sh0181, 16'shFD99, 16'shFED5, 16'shFD98, 16'sh0769},
        '{16'shFFF2, 16'shFBC5, 16'sh034A, 16'shFB1E, 16'sh00D6, 16'shF829, 16'shFAB0, 16'sh00CA},
        '{16'sh02F4, 16'sh00AF, 16'shFF8A, 16'shFECC, 16'shFA16, 16'shFD1F, 16'shFE28, 16'sh043A}
        };

          W_query = '{
        '{16'sh0120, 16'sh022C, 16'sh0090, 16'sh0335, 16'sh004C, 16'sh03F3, 16'sh0317, 16'sh00CB, 16'sh0006, 16'sh0343, 16'sh02D4, 16'sh02EB, 16'sh0316, 16'sh004C, 16'sh016F, 16'sh0077, 16'sh0374, 16'sh027E, 16'sh0153, 16'sh0041, 16'sh013E, 16'sh014D, 16'sh02EB, 16'sh028D},
        '{16'sh038D, 16'sh01E4, 16'sh007A, 16'sh02DA, 16'sh030B, 16'sh023F, 16'sh0315, 16'sh01FA, 16'sh0217, 16'sh01B6, 16'sh001A, 16'sh006E, 16'sh0020, 16'sh028C, 16'sh0142, 16'sh0209, 16'sh03A1, 16'sh00FF, 16'sh01A4, 16'sh0306, 16'sh00EA, 16'sh004F, 16'sh0129, 16'sh00A5},
        '{16'sh03B8, 16'sh033C, 16'sh0289, 16'sh037C, 16'sh0337, 16'sh00BF, 16'sh0392, 16'sh0228, 16'sh033B, 16'sh0396, 16'sh0146, 16'sh0071, 16'sh00E9, 16'sh01B5, 16'sh0346, 16'sh0371, 16'sh0007, 16'sh020B, 16'sh01AB, 16'sh00E3, 16'sh007B, 16'sh015A, 16'sh03C6, 16'sh014B},
        '{16'sh0213, 16'sh02D0, 16'sh0174, 16'sh03E3, 16'sh03DA, 16'sh0102, 16'sh01FD, 16'sh0134, 16'sh0124, 16'sh0026, 16'sh0270, 16'sh0203, 16'sh0035, 16'sh011D, 16'sh03A2, 16'sh00F5, 16'sh0094, 16'sh01F5, 16'sh03F1, 16'sh00F8, 16'sh02B0, 16'sh030C, 16'sh00F3, 16'sh02EA},
        '{16'sh0179, 16'sh0287, 16'sh0289, 16'sh0225, 16'sh005C, 16'sh0357, 16'sh0148, 16'sh00BF, 16'sh002A, 16'sh025D, 16'sh02B6, 16'sh0011, 16'sh020C, 16'sh00E8, 16'sh0295, 16'sh00B3, 16'sh02C4, 16'sh018C, 16'sh03BF, 16'sh008D, 16'sh015D, 16'sh0074, 16'sh03B3, 16'sh0382},
        '{16'sh0108, 16'sh02A4, 16'sh0345, 16'sh0239, 16'sh021E, 16'sh00F8, 16'sh005F, 16'sh0397, 16'sh039A, 16'sh0288, 16'sh015B, 16'sh0166, 16'sh02E7, 16'sh0397, 16'sh038C, 16'sh031F, 16'sh0291, 16'sh0056, 16'sh00A6, 16'sh0398, 16'sh026D, 16'sh0009, 16'sh0068, 16'sh02A7},
        '{16'sh0005, 16'sh00A5, 16'sh0232, 16'sh02C5, 16'sh029C, 16'sh00E6, 16'sh02D9, 16'sh00F3, 16'sh014D, 16'sh02FC, 16'sh0299, 16'sh0366, 16'sh02A1, 16'sh0246, 16'sh0060, 16'sh0179, 16'sh0110, 16'sh00FA, 16'sh03E4, 16'sh0193, 16'sh0391, 16'sh0286, 16'sh032E, 16'sh0203},
        '{16'sh024F, 16'sh01F8, 16'sh00C8, 16'sh02E4, 16'sh0120, 16'sh0019, 16'sh0295, 16'sh00B5, 16'sh03C3, 16'sh03D1, 16'sh03A9, 16'sh017B, 16'sh0010, 16'sh03B7, 16'sh01B6, 16'sh03DE, 16'sh03DB, 16'sh0369, 16'sh012E, 16'sh018A, 16'sh0368, 16'sh0145, 16'sh00AE, 16'sh023A}
        };

        W_key = '{
        '{16'sh03BF, 16'sh02C9, 16'sh0248, 16'sh0064, 16'sh0276, 16'sh03F6, 16'sh008F, 16'sh0213, 16'sh0382, 16'sh02F7, 16'sh02CA, 16'sh02CF, 16'sh0170, 16'sh012D, 16'sh033D, 16'sh033E, 16'sh0378, 16'sh03A7, 16'sh020C, 16'sh0202, 16'sh0331, 16'sh029A, 16'sh02CF, 16'sh032F},
        '{16'sh038F, 16'sh015A, 16'sh0181, 16'sh0060, 16'sh0250, 16'sh0025, 16'sh01DD, 16'sh022C, 16'sh0125, 16'sh025D, 16'sh001F, 16'sh0026, 16'sh034A, 16'sh0171, 16'sh0082, 16'sh0217, 16'sh0314, 16'sh00DD, 16'sh027E, 16'sh0057, 16'sh0035, 16'sh0220, 16'sh022A, 16'sh028D},
        '{16'sh02E8, 16'sh03E7, 16'sh0211, 16'sh014B, 16'sh032E, 16'sh0115, 16'sh01C2, 16'sh0050, 16'sh001A, 16'sh03DA, 16'sh0358, 16'sh02C9, 16'sh01A3, 16'sh00B1, 16'sh00A0, 16'sh0100, 16'sh0232, 16'sh02DC, 16'sh02A4, 16'sh011F, 16'sh03D2, 16'sh02F4, 16'sh0238, 16'sh0272},
        '{16'sh01AE, 16'sh00FE, 16'sh016D, 16'sh0308, 16'sh000F, 16'sh0077, 16'sh002F, 16'sh002A, 16'sh036C, 16'sh02D1, 16'sh01E6, 16'sh0064, 16'sh01F7, 16'sh01E5, 16'sh00B1, 16'sh01BC, 16'sh0198, 16'sh0277, 16'sh028A, 16'sh002E, 16'sh0180, 16'sh0281, 16'sh0203, 16'sh036D},
        '{16'sh02A3, 16'sh00A7, 16'sh0048, 16'sh0292, 16'sh001B, 16'sh0258, 16'sh03C3, 16'sh024D, 16'sh018D, 16'sh0293, 16'sh01D5, 16'sh022F, 16'sh03C4, 16'sh018B, 16'sh03D8, 16'sh039F, 16'sh00C8, 16'sh0047, 16'sh0067, 16'sh0013, 16'sh0061, 16'sh02BB, 16'sh0049, 16'sh0147},
        '{16'sh0361, 16'sh0018, 16'sh0342, 16'sh0121, 16'sh0079, 16'sh02C9, 16'sh0284, 16'sh0383, 16'sh02F1, 16'sh0337, 16'sh0121, 16'sh00B6, 16'sh0301, 16'sh033A, 16'sh03F6, 16'sh01A7, 16'sh017D, 16'sh031B, 16'sh015D, 16'sh03B9, 16'sh036F, 16'sh01B7, 16'sh0301, 16'sh0305},
        '{16'sh006A, 16'sh039C, 16'sh0205, 16'sh034E, 16'sh0148, 16'sh0395, 16'sh018F, 16'sh000B, 16'sh039F, 16'sh005D, 16'sh0147, 16'sh03CD, 16'sh03CD, 16'sh024B, 16'sh0287, 16'sh01CB, 16'sh012C, 16'sh0151, 16'sh02B1, 16'sh0302, 16'sh032B, 16'sh0329, 16'sh005D, 16'sh01FA},
        '{16'sh003B, 16'sh0233, 16'sh01C4, 16'sh038D, 16'sh0167, 16'sh0078, 16'sh0092, 16'sh030C, 16'sh0279, 16'sh0068, 16'sh0056, 16'sh02CE, 16'sh004B, 16'sh034A, 16'sh02D3, 16'sh0053, 16'sh0057, 16'sh03F2, 16'sh017F, 16'sh017C, 16'sh0340, 16'sh03CA, 16'sh03F2, 16'sh0303}
        };

        W_value = '{
        '{16'sh01A8, 16'sh013E, 16'shFFC8, 16'sh0082, 16'shFFEF, 16'sh0071, 16'sh00B9, 16'shFD95, 16'sh0193, 16'sh0247, 16'sh0180, 16'shFD81, 16'sh01DE, 16'shFE21, 16'sh0010, 16'shFFE8, 16'sh00FB, 16'sh00F1, 16'shFCAB, 16'shFEAE, 16'shFDFA, 16'shFDFF, 16'shFE63, 16'shFFA1},
        '{16'sh0035, 16'sh02A6, 16'shFD55, 16'sh023C, 16'shFCC4, 16'sh0247, 16'sh01EB, 16'shFC8F, 16'shFFDA, 16'sh0151, 16'shFE1B, 16'shFDD7, 16'shFD82, 16'sh0024, 16'sh010E, 16'shFE5D, 16'shFF95, 16'shFD2F, 16'sh039E, 16'shFFBE, 16'sh00E9, 16'sh00BD, 16'sh000D, 16'sh0304},
        '{16'sh010D, 16'sh03C6, 16'shFE21, 16'shFF8F, 16'shFD89, 16'shFF20, 16'sh0028, 16'shFE7E, 16'sh0264, 16'shFD8A, 16'sh0222, 16'sh00BD, 16'shFF91, 16'shFD82, 16'sh01AF, 16'sh0049, 16'shFE5A, 16'sh015E, 16'shFEE8, 16'sh002C, 16'sh0303, 16'sh00A0, 16'sh01E5, 16'shFFF5},
        '{16'shFFD8, 16'sh03F4, 16'shFD8D, 16'sh01DD, 16'shFE84, 16'sh02FE, 16'sh0320, 16'shFEF4, 16'sh026E, 16'shFEBB, 16'shFDFB, 16'shFDAB, 16'sh002A, 16'sh0191, 16'sh0142, 16'sh0228, 16'sh0063, 16'sh0060, 16'sh0106, 16'sh013A, 16'sh0021, 16'sh0013, 16'sh034C, 16'shFEED},
        '{16'sh02FF, 16'shFE1E, 16'sh0145, 16'sh020D, 16'shFE0A, 16'sh00B4, 16'sh0338, 16'sh00A0, 16'sh03AD, 16'sh0336, 16'shFFB1, 16'shFF81, 16'sh032F, 16'shFE46, 16'sh01A3, 16'shFED6, 16'sh0175, 16'shFF88, 16'shFD47, 16'sh03F5, 16'shFE79, 16'shFE29, 16'sh0122, 16'sh012B},
        '{16'sh03B1, 16'shFCC2, 16'shFFE9, 16'sh01B7, 16'sh00DC, 16'shFF61, 16'shFEEB, 16'shFD81, 16'shFFDF, 16'sh023B, 16'shFF90, 16'shFD01, 16'shFE40, 16'shFC69, 16'sh01B2, 16'shFF54, 16'sh0099, 16'shFDA3, 16'sh0228, 16'shFF31, 16'sh02DB, 16'sh00EA, 16'sh01DD, 16'sh03DE},
        '{16'sh0132, 16'shFEC5, 16'shFD71, 16'shFDFB, 16'sh0155, 16'sh038A, 16'sh0273, 16'shFFEF, 16'sh01CB, 16'shFDE3, 16'sh00B3, 16'shFFA1, 16'sh0164, 16'shFF7E, 16'sh0079, 16'shFF2D, 16'sh00CA, 16'shFE34, 16'sh0223, 16'shFF07, 16'shFF1F, 16'shFEF4, 16'shFDF7, 16'shFFDF},
        '{16'sh0308, 16'shFCA5, 16'shFD96, 16'shFF1B, 16'shFE88, 16'shFF7A, 16'shFF4F, 16'sh00C1, 16'sh0373, 16'shFED2, 16'sh0346, 16'sh006E, 16'sh0151, 16'sh0178, 16'shFE5E, 16'shFDFF, 16'shFD89, 16'shFF25, 16'sh00CE, 16'sh0254, 16'shFE53, 16'shFF6C, 16'shFED6, 16'sh007D}
        };

        // Display weight matrices
        $display("Initial Weight Matrices:");
        print_matrix(dut.W_query, "W_query");
        print_matrix(dut.W_key, "W_key");
        print_value_matrix(dut.W_value, "W_value"); // Note: DV = DK, so size matches

        #10; // Wait for computation

        // Print output_data
        print_output_data(output_data, "Output Data");

        $finish;
    end
endmodule










/*wider range LUT
// Exponential approximation function using LUT with interpolation in Q5.10 format
// LUT covers x = -5 to 4, sufficient for typical self-attention scores
// Q5.10 inputs can range from -32 to 31, but extreme values are handled as follows:
// - x < -5: returns 0 (e^x is negligible)
// - x > 4: caps at e^4 (prevents overflow in 16-bit output)
function automatic logic [31:0] exp_approx(input logic [31:0] val);
    // LUT for x = -5 to 4, in Q5.10 format (scaled by 1024)
    logic [15:0] lut [0:9] = '{
        16'h0007, // e^-5 ≈ 0.0067 * 1024 ≈ 7
        16'h0013, // e^-4 ≈ 0.0183 * 1024 ≈ 19
        16'h0033, // e^-3 ≈ 0.0498 * 1024 ≈ 51
        16'h008B, // e^-2 ≈ 0.1353 * 1024 ≈ 139
        16'h0178, // e^-1 ≈ 0.3679 * 1024 ≈ 376
        16'h0400, // e^0 = 1.0 * 1024 = 1024
        16'h0AE0, // e^1 ≈ 2.718 * 1024 ≈ 2784
        16'h1D8A, // e^2 ≈ 7.389 * 1024 ≈ 7562
        16'h5058, // e^3 ≈ 20.086 * 1024 ≈ 20568
        16'hDA39  // e^4 ≈ 54.598 * 1024 ≈ 55889
    };
    int int_part = $signed(val) >>> 10; // Extract integer part (Q5.10)
    logic [9:0] frac = val[9:0]; // Fractional part (lower 10 bits)
    if (int_part < -5) begin
        return 32'h00000000; // e^x ≈ 0 for x < -5
    end else if (int_part > 4) begin
        return {16'h0000, lut[9]}; // Cap at e^4 for x > 4
    end else begin
        int index = int_part + 5; // Map -5 to 0, ..., 4 to 9
        if (index == 9) begin
            return {16'h0000, lut[9]}; // Use e^4 directly for x = 4
        end else begin
            logic [15:0] lut_val = lut[index];
            logic [15:0] lut_next = lut[index + 1];
            logic [15:0] delta = lut_next - lut_val; // Difference for interpolation
            logic [25:0] product = delta * frac; // 16-bit * 10-bit = 26-bit
            logic [15:0] increment = product >> 10; // Scale by 1/1024
            logic [15:0] approx = lut_val + increment; // Interpolated value
            return {16'h0000, approx}; // Zero-extend to 32 bits
        end
    end
endfunction
*/



