`timescale 1ns/1ps
//=============================================================================
// Project : FPGA ADAS Road Perception System
// Module  : adas_decision
// Version : 2.0
// Function: 融合ARM检测、灯色/导航、车辆状态和FPGA车道类型，生成驾驶动作与告警。
// Priority: 通信超时 > 行人 > 路障 > 红灯 > 非法实线变道 > 正常导航。
//=============================================================================
module adas_decision #(
    parameter [31:0] STATUS_TIMEOUT_CYCLES = 32'd25000000
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       target_valid,
    input  wire [2:0] target_type,
    input  wire       status_valid,
    input  wire [7:0] status_valid_flags,
    input  wire [7:0] status_object_flags,
    input  wire [2:0] status_light_state,
    input  wire [2:0] status_navigation,
    input  wire [7:0] status_control_flags,
    input  wire [1:0] lane_type,
    output reg        stop_enable,
    output reg        alarm_enable,
    output reg  [2:0] action_code,
    output reg  [2:0] alarm_code,
    output reg  [1:0] decision_code,
    output reg        link_timeout
);
    localparam [2:0] ACTION_UNKNOWN=3'd0, ACTION_STOP=3'd1;
    localparam [2:0] ACTION_STRAIGHT=3'd2, ACTION_LEFT=3'd3, ACTION_RIGHT=3'd4;
    localparam [2:0] ACTION_CHANGE_LEFT=3'd5, ACTION_CHANGE_RIGHT=3'd6;
    localparam [2:0] ALARM_NONE=3'd0, ALARM_COLLISION=3'd1;
    localparam [2:0] ALARM_RED_VIOLATION=3'd2, ALARM_SOLID_CHANGE=3'd3;
    localparam [2:0] ALARM_LINK_TIMEOUT=3'd4;
    localparam [1:0] DECISION_NONE=2'd0, DECISION_OBSTACLE=2'd1;
    localparam [1:0] DECISION_RED=2'd2, DECISION_PERSON=2'd3;
    localparam [2:0] LIGHT_RED=3'd1, LIGHT_GREEN=3'd3;
    localparam [1:0] LANE_DASHED=2'd1, LANE_SOLID=2'd2;

    reg [31:0] timeout_counter;
    reg seen_v2_status;
    wire person=status_object_flags[1];
    wire obstacle=status_object_flags[3];
    wire light_valid=status_valid_flags[1];
    wire navigation_valid=status_valid_flags[2];
    wire motion_valid=status_valid_flags[3];
    wire vehicle_moving=motion_valid && status_control_flags[0];
    wire force_move=status_control_flags[1];
    wire force_lane_change=status_control_flags[2];
    wire change_right=status_control_flags[3];

    always @(posedge clk) begin
        if (!rst_n) begin
            timeout_counter<=0; seen_v2_status<=0; link_timeout<=0;
        end else if (status_valid) begin
            timeout_counter<=0; seen_v2_status<=1; link_timeout<=0;
        end else if (seen_v2_status && !link_timeout) begin
            if (timeout_counter>=STATUS_TIMEOUT_CYCLES-1'b1) begin
                timeout_counter<=timeout_counter; link_timeout<=1;
            end else timeout_counter<=timeout_counter+1'b1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            stop_enable<=0; alarm_enable<=0; action_code<=ACTION_UNKNOWN;
            alarm_code<=ALARM_NONE; decision_code<=DECISION_NONE;
        end else if (link_timeout && !status_valid) begin
            stop_enable<=1; alarm_enable<=1; action_code<=ACTION_STOP;
            alarm_code<=ALARM_LINK_TIMEOUT; decision_code<=DECISION_NONE;
        end else if (status_valid) begin
            stop_enable<=0; alarm_enable<=0; action_code<=ACTION_UNKNOWN;
            alarm_code<=ALARM_NONE; decision_code<=DECISION_NONE;
            if (person) begin
                stop_enable<=1; alarm_enable<=1; action_code<=ACTION_STOP;
                alarm_code<=ALARM_COLLISION; decision_code<=DECISION_PERSON;
            end else if (obstacle) begin
                stop_enable<=1; alarm_enable<=1; action_code<=ACTION_STOP;
                alarm_code<=ALARM_COLLISION; decision_code<=DECISION_OBSTACLE;
            end else if (light_valid && status_light_state==LIGHT_RED) begin
                stop_enable<=1; action_code<=ACTION_STOP; decision_code<=DECISION_RED;
                if (vehicle_moving || force_move) begin
                    alarm_enable<=1; alarm_code<=ALARM_RED_VIOLATION;
                end
            end else if (force_lane_change && lane_type==LANE_SOLID) begin
                alarm_enable<=1; action_code<=ACTION_STRAIGHT;
                alarm_code<=ALARM_SOLID_CHANGE;
            end else if (force_lane_change && lane_type==LANE_DASHED) begin
                action_code<=change_right ? ACTION_CHANGE_RIGHT : ACTION_CHANGE_LEFT;
            end else if (light_valid && status_light_state==LIGHT_GREEN) begin
                if (navigation_valid && status_navigation==3'd1) action_code<=ACTION_LEFT;
                else if (navigation_valid && status_navigation==3'd3) action_code<=ACTION_RIGHT;
                else action_code<=ACTION_STRAIGHT;
            end else if (navigation_valid) begin
                if (status_navigation==3'd1) action_code<=ACTION_LEFT;
                else if (status_navigation==3'd3) action_code<=ACTION_RIGHT;
                else if (status_navigation==3'd2) action_code<=ACTION_STRAIGHT;
            end
        end else if (target_valid && !seen_v2_status) begin
            // 旧协议只用于早期串口/LED回退。
            alarm_code<=ALARM_NONE;
            if (target_type[2]) begin
                stop_enable<=1; alarm_enable<=1; action_code<=ACTION_STOP;
                alarm_code<=ALARM_COLLISION; decision_code<=DECISION_PERSON;
            end else if (target_type[1]) begin
                stop_enable<=1; alarm_enable<=0; action_code<=ACTION_STOP;
                decision_code<=DECISION_RED;
            end else if (target_type[0]) begin
                stop_enable<=1; alarm_enable<=1; action_code<=ACTION_STOP;
                alarm_code<=ALARM_COLLISION; decision_code<=DECISION_OBSTACLE;
            end else begin
                stop_enable<=0; alarm_enable<=0; action_code<=ACTION_UNKNOWN;
                decision_code<=DECISION_NONE;
            end
        end
    end
endmodule
