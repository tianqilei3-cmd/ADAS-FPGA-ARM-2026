`timescale 1ns/1ps
//=============================================================================
// Project : FPGA ADAS Road Perception System
// Module  : adas_key_control
// Version : 1.0
// Function: Six active-low physical keys: synchronize, debounce and maintain
//           an idempotent ADAS control-state byte for UART query command 0x03.
//=============================================================================
module adas_key_control #(
    parameter integer DEBOUNCE_COUNT = 1000000  // 20 ms at 50 MHz
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [5:0] key_n,
    input wire remote_valid,
    input wire [7:0] remote_state,

    output reg  [1:0] scene_mode,
    output reg  [1:0] direction_request,
    output reg        drive_request,
    output reg        force_request,
    output reg        control_active,
    output reg  [7:0] key_sequence,
    output reg  [5:0] key_press_pulse,
    output wire [7:0] key_state_byte
);

    // key_n bit assignment:
    // [0] scene, [1] left, [2] straight, [3] right,
    // [4] drive/stop toggle, [5] force toggle.
    // All physical keys are active low.

    localparam [1:0] DIR_STRAIGHT = 2'd0;
    localparam [1:0] DIR_LEFT     = 2'd1;
    localparam [1:0] DIR_RIGHT    = 2'd2;

    reg remote_mode;
    reg [5:0] key_sync_0;
    reg [5:0] key_sync_1;
    reg [5:0] key_stable;
    reg [5:0] key_stable_d;
    reg [19:0] debounce_counter [0:5];

    genvar i;

    // Two-stage synchronizer for asynchronous mechanical-key inputs.
    always @(posedge clk) begin
        if (!rst_n) begin
            key_sync_0 <= 6'b111111;
            key_sync_1 <= 6'b111111;
        end
        else begin
            key_sync_0 <= key_n;
            key_sync_1 <= key_sync_0;
        end
    end

    // A new electrical level is accepted only after it remains unchanged for
    // DEBOUNCE_COUNT clock cycles.
    generate
        for (i = 0; i < 6; i = i + 1) begin : g_key_debounce
            always @(posedge clk) begin
                if (!rst_n) begin
                    key_stable[i]        <= 1'b1;
                    debounce_counter[i] <= 20'd0;
                end
                else if (key_sync_1[i] == key_stable[i]) begin
                    debounce_counter[i] <= 20'd0;
                end
                else if (debounce_counter[i] == DEBOUNCE_COUNT - 1) begin
                    key_stable[i]        <= key_sync_1[i];
                    debounce_counter[i] <= 20'd0;
                end
                else begin
                    debounce_counter[i] <= debounce_counter[i] + 20'd1;
                end
            end
        end
    endgenerate

    // Generate one clk-wide pulse on each debounced press (high-to-low).
    always @(posedge clk) begin
        if (!rst_n) begin
            key_stable_d    <= 6'b111111;
            key_press_pulse <= 6'b000000;
        end
        else begin
            key_stable_d    <= key_stable;
            key_press_pulse <= key_stable_d & ~key_stable;
        end
    end

    // Persistent control state. A queried snapshot is therefore not lost when
    // the physical key has already been released.
    always @(posedge clk) begin
        if (!rst_n) begin
            scene_mode       <= 2'd0;
            direction_request<= DIR_STRAIGHT;
            drive_request    <= 1'b0;
            force_request    <= 1'b0;
            control_active   <= 1'b0;
            key_sequence     <= 8'd0;
            remote_mode<=0;
        end
        else begin
            if(remote_valid && !(|key_press_pulse))begin
                control_active<=1;key_sequence<=key_sequence+8'd1;
                if(remote_state==8'hff)begin
                    scene_mode<=1;direction_request<=0;drive_request<=0;force_request<=0;remote_mode<=0;
                end else begin
                    scene_mode<=remote_state[3:2];direction_request<=remote_state[1:0];
                    drive_request<=remote_state[4];force_request<=remote_state[5];remote_mode<=1;
                end
            end else begin
            if (|key_press_pulse) begin
                remote_mode<=0;
                control_active <= 1'b1;
                key_sequence <= key_sequence + 8'd1;
            end
            if (key_press_pulse[0]) begin
                scene_mode    <= scene_mode + 2'd1;
                force_request <= 1'b0;
            end

            if (key_press_pulse[1]) begin
                direction_request <= DIR_LEFT;
                force_request     <= 1'b0;
            end
            else if (key_press_pulse[2]) begin
                direction_request <= DIR_STRAIGHT;
                force_request     <= 1'b0;
            end
            else if (key_press_pulse[3]) begin
                direction_request <= DIR_RIGHT;
                force_request     <= 1'b0;
            end

            if (key_press_pulse[4]) begin
                drive_request <= ~drive_request;
                if (drive_request)
                    force_request <= 1'b0;
            end

            // Force request is accepted only while a drive request is active.
            if (key_press_pulse[5] && drive_request && !(|key_press_pulse[4:0]))
                force_request <= ~force_request;
            end
        end
    end

    // UART 0x03 response payload:
    // [7] reserved=0, [6] explicitly labelled remote test driver
    // [5] force request
    // [4] drive request (1=request movement, 0=stop)
    // [3:2] scene (0=traffic light, 1=lane change,
    //              2=navigation, 3=road risk)
    // [1:0] direction (0=straight, 1=left, 2=right, 3=reserved)
    assign key_state_byte = {1'b0, remote_mode, force_request, drive_request,
                             scene_mode, direction_request};

endmodule
