`timescale 1ns/1ps

//=============================================================================
// Copyright (c) 2026 ADAS Project Team
// Project : FPGA ADAS Road Perception System
// Module  : lane_roi_mask
// Version : 1.0
// Author  : ADAS Project Team
// Function: 生成道路梯形感兴趣区域。视频数据原样透传，o_roi_valid仅标记
//           需要送入车道检测的像素，避免破坏最终显示画面。
//=============================================================================
module lane_roi_mask #(
    parameter integer H_ACTIVE            = 1920,
    parameter integer V_ACTIVE            = 1080,
    parameter integer ROI_TOP             = 540,
    parameter integer ROI_TOP_HALF_WIDTH  = 160,
    parameter integer ROI_SLOPE_SHIFT     = 1
)(
    input  wire        pix_clk,
    input  wire        rst_n,            // 同步低有效复位
    input  wire        i_vsync,
    input  wire        i_hsync,
    input  wire        i_de,
    input  wire [7:0]  i_r,
    input  wire [7:0]  i_g,
    input  wire [7:0]  i_b,

    output reg         o_vsync,
    output reg         o_hsync,
    output reg         o_de,
    output reg  [7:0]  o_r,
    output reg  [7:0]  o_g,
    output reg  [7:0]  o_b,
    output reg         o_roi_valid,
    output reg  [11:0] o_x,
    output reg  [11:0] o_y
);

    localparam [11:0] CENTER_X = H_ACTIVE / 2;

    reg [11:0] x_count;
    reg [11:0] y_count;
    reg        de_d;
    reg        vsync_d;

    wire       frame_start;
    wire [11:0] roi_delta_y;
    wire [11:0] roi_half_width;
    wire [11:0] roi_left;
    wire [11:0] roi_right;
    wire        roi_now;

    assign frame_start = i_vsync && !vsync_d;
    assign roi_delta_y = (y_count >= ROI_TOP) ? (y_count - ROI_TOP) : 12'd0;
    assign roi_half_width = ROI_TOP_HALF_WIDTH +
                            (roi_delta_y >> ROI_SLOPE_SHIFT);
    assign roi_left  = (CENTER_X > roi_half_width) ?
                       (CENTER_X - roi_half_width) : 12'd0;
    assign roi_right = ((CENTER_X + roi_half_width) < H_ACTIVE) ?
                       (CENTER_X + roi_half_width) : H_ACTIVE - 1;
    assign roi_now = i_de && (y_count >= ROI_TOP) &&
                     (y_count < V_ACTIVE) &&
                     (x_count >= roi_left) && (x_count <= roi_right);

    // 根据DE生成有效区像素坐标。
    always @(posedge pix_clk) begin
        if (!rst_n) begin
            x_count <= 12'd0;
            y_count <= 12'd0;
            de_d     <= 1'b0;
            vsync_d  <= 1'b0;
        end
        else begin
            de_d    <= i_de;
            vsync_d <= i_vsync;

            if (frame_start) begin
                x_count <= 12'd0;
                y_count <= 12'd0;
            end
            else begin
                if (i_de)
                    x_count <= x_count + 1'b1;
                else
                    x_count <= 12'd0;

                if (de_d && !i_de) begin
                    if (y_count < V_ACTIVE - 1)
                        y_count <= y_count + 1'b1;
                    else
                        y_count <= 12'd0;
                end
            end
        end
    end

    // 视频和ROI标志统一延迟一拍输出。
    always @(posedge pix_clk) begin
        if (!rst_n) begin
            o_vsync    <= 1'b0;
            o_hsync    <= 1'b0;
            o_de       <= 1'b0;
            o_r        <= 8'd0;
            o_g        <= 8'd0;
            o_b        <= 8'd0;
            o_roi_valid <= 1'b0;
            o_x        <= 12'd0;
            o_y        <= 12'd0;
        end
        else begin
            o_vsync    <= i_vsync;
            o_hsync    <= i_hsync;
            o_de       <= i_de;
            o_r        <= i_r;
            o_g        <= i_g;
            o_b        <= i_b;
            o_roi_valid <= roi_now;
            o_x        <= x_count;
            o_y        <= y_count;
        end
    end

endmodule

