`timescale 1ns/1ps
module adas_key_control_tb;

    reg clk;
    reg rst_n;
    reg [5:0] key_n;
    wire [1:0] scene_mode;
    wire [1:0] direction_request;
    wire drive_request;
    wire force_request;
    wire control_active;
    wire [7:0] key_sequence;
    wire [5:0] key_press_pulse;
    wire [7:0] key_state_byte;
    integer errors;

    adas_key_control #(
        .DEBOUNCE_COUNT(4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .key_n(key_n),
        .scene_mode(scene_mode),
        .direction_request(direction_request),
        .drive_request(drive_request),
        .force_request(force_request),
        .control_active(control_active),
        .key_sequence(key_sequence),
        .key_press_pulse(key_press_pulse),
        .key_state_byte(key_state_byte)
    );

    always #10 clk = ~clk;

    task press_key;
        input integer index;
        begin
            @(negedge clk);
            key_n[index] = 1'b0;
            repeat (8) @(posedge clk);
            @(negedge clk);
            key_n[index] = 1'b1;
            repeat (8) @(posedge clk);
            @(negedge clk);
        end
    endtask

    task expect_state;
        input [7:0] expected;
        input [7:0] expected_sequence;
        begin
            if (key_state_byte !== expected || key_sequence !== expected_sequence) begin
                $display("ERROR time=%0t state=%02h expected=%02h sequence=%02h expected=%02h",
                         $time, key_state_byte, expected, key_sequence, expected_sequence);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        key_n = 6'b111111;
        errors = 0;
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);
        @(negedge clk);

        // Default: traffic-light scene, straight, stop, no force.
        expect_state(8'h00, 8'd0);
        if (control_active !== 1'b0) errors = errors + 1;

        press_key(0); // scene -> lane-change scene
        expect_state(8'h04, 8'd1);
        if (control_active !== 1'b1) errors = errors + 1;

        press_key(1); // left
        expect_state(8'h05, 8'd2);

        press_key(4); // request movement
        expect_state(8'h15, 8'd3);

        press_key(5); // force request
        expect_state(8'h35, 8'd4);

        press_key(2); // straight; force is cleared
        expect_state(8'h14, 8'd5);

        press_key(3); // right
        expect_state(8'h16, 8'd6);

        press_key(4); // stop; force remains cleared
        expect_state(8'h06, 8'd7);

        // Short bounce shorter than the debounce threshold must be ignored.
        @(negedge clk);
        key_n[1] = 1'b0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        key_n[1] = 1'b1;
        repeat (8) @(posedge clk);
        @(negedge clk);
        expect_state(8'h06, 8'd7);

        if (errors == 0)
            $display("ALL_KEY_TESTS_PASSED");
        else
            $display("KEY_TESTS_FAILED errors=%0d", errors);
        $finish;
    end

endmodule
