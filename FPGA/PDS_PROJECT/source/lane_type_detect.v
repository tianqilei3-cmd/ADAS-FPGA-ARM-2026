`timescale 1ns/1ps

//=============================================================================
// Copyright (c) 2026 ADAS Project Team
// Project : FPGA ADAS Road Perception System
// Module  : lane_type_detect
// Version : 1.0
// Author  : ADAS Project Team
// Function: 在ROI内提取白/黄色车道像素，按每帧有效行覆盖率和断续变化
//           判断实线/虚线，并根据左右像素缺失给出偏离预警。
//=============================================================================
module lane_type_detect #(
    parameter integer H_ACTIVE          = 1920,
    parameter integer WHITE_MIN         = 190,
    parameter integer WHITE_SPREAD_MAX  = 45,
    parameter integer YELLOW_R_MIN      = 170,
    parameter integer YELLOW_G_MIN      = 140,
    parameter integer YELLOW_B_MAX      = 150,
    parameter integer ROW_PIXEL_MIN     = 4,
    parameter integer SOLID_ROW_MIN     = 320,
    parameter integer DASHED_ROW_MIN    = 70,
    parameter integer MIN_TRANSITIONS   = 4,
    parameter integer SIDE_PIXEL_MIN    = 800
)(
    input  wire        pix_clk,
    input  wire        rst_n,                // 同步低有效复位
    input  wire        i_vsync,
    input  wire        i_de,
    input  wire        i_roi_valid,
    input  wire [11:0] i_x,
    input  wire [11:0] i_y,
    input  wire [7:0]  i_r,
    input  wire [7:0]  i_g,
    input  wire [7:0]  i_b,

    output reg  [1:0]  lane_type,             // 00未知 01虚线 10实线
    output reg         lane_departure_warning,
    output reg  [1:0]  departure_direction,   // 00正常 01左侧缺失 10右侧缺失
    output wire        lane_pixel_debug
);

    localparam [1:0] LANE_UNKNOWN = 2'b00;
    localparam [1:0] LANE_DASHED  = 2'b01;
    localparam [1:0] LANE_SOLID   = 2'b10;

    reg        vsync_d;
    reg        de_d;
    reg [15:0] row_lane_pixels;
    reg        row_lane_hit;
    reg        previous_row_hit;
    reg [15:0] valid_lane_rows;
    reg [15:0] row_transitions;
    reg [31:0] left_lane_pixels;
    reg [31:0] right_lane_pixels;

    wire       frame_start;
    wire       line_end;
    wire [7:0] rgb_max;
    wire [7:0] rgb_min;
    wire       white_pixel;
    wire       yellow_pixel;
    wire       lane_pixel;

    assign frame_start = i_vsync && !vsync_d;
    assign line_end    = de_d && !i_de;

    assign rgb_max = (i_r >= i_g) ? ((i_r >= i_b) ? i_r : i_b) :
                                    ((i_g >= i_b) ? i_g : i_b);
    assign rgb_min = (i_r <= i_g) ? ((i_r <= i_b) ? i_r : i_b) :
                                    ((i_g <= i_b) ? i_g : i_b);

    assign white_pixel = (i_r >= WHITE_MIN) && (i_g >= WHITE_MIN) &&
                         (i_b >= WHITE_MIN) &&
                         ((rgb_max - rgb_min) <= WHITE_SPREAD_MAX);
    assign yellow_pixel = (i_r >= YELLOW_R_MIN) &&
                          (i_g >= YELLOW_G_MIN) &&
                          (i_b <= YELLOW_B_MAX) && (i_r >= i_g);
    assign lane_pixel = i_de && i_roi_valid && (white_pixel || yellow_pixel);
    assign lane_pixel_debug = lane_pixel;

    always @(posedge pix_clk) begin
        if (!rst_n) begin
            vsync_d               <= 1'b0;
            de_d                  <= 1'b0;
            row_lane_pixels       <= 16'd0;
            row_lane_hit          <= 1'b0;
            previous_row_hit      <= 1'b0;
            valid_lane_rows       <= 16'd0;
            row_transitions       <= 16'd0;
            left_lane_pixels      <= 32'd0;
            right_lane_pixels     <= 32'd0;
            lane_type             <= LANE_UNKNOWN;
            lane_departure_warning <= 1'b0;
            departure_direction  <= 2'b00;
        end
        else begin
            vsync_d <= i_vsync;
            de_d    <= i_de;

            if (frame_start) begin
                // 使用上一帧统计值产生稳定的帧级输出。
                if (valid_lane_rows >= SOLID_ROW_MIN)
                    lane_type <= LANE_SOLID;
                else if ((valid_lane_rows >= DASHED_ROW_MIN) &&
                         (row_transitions >= MIN_TRANSITIONS))
                    lane_type <= LANE_DASHED;
                else
                    lane_type <= LANE_UNKNOWN;

                if ((left_lane_pixels < SIDE_PIXEL_MIN) &&
                    (right_lane_pixels >= SIDE_PIXEL_MIN)) begin
                    lane_departure_warning <= 1'b1;
                    departure_direction <= 2'b01;
                end
                else if ((right_lane_pixels < SIDE_PIXEL_MIN) &&
                         (left_lane_pixels >= SIDE_PIXEL_MIN)) begin
                    lane_departure_warning <= 1'b1;
                    departure_direction <= 2'b10;
                end
                else begin
                    lane_departure_warning <= 1'b0;
                    departure_direction <= 2'b00;
                end

                row_lane_pixels   <= 16'd0;
                row_lane_hit      <= 1'b0;
                previous_row_hit  <= 1'b0;
                valid_lane_rows   <= 16'd0;
                row_transitions   <= 16'd0;
                left_lane_pixels  <= 32'd0;
                right_lane_pixels <= 32'd0;
            end
            else begin
                if (lane_pixel) begin
                    row_lane_pixels <= row_lane_pixels + 1'b1;
                    if (row_lane_pixels >= ROW_PIXEL_MIN - 1)
                        row_lane_hit <= 1'b1;

                    if (i_x < H_ACTIVE / 2)
                        left_lane_pixels <= left_lane_pixels + 1'b1;
                    else
                        right_lane_pixels <= right_lane_pixels + 1'b1;
                end

                if (line_end) begin
                    if (row_lane_hit)
                        valid_lane_rows <= valid_lane_rows + 1'b1;
                    if (row_lane_hit != previous_row_hit)
                        row_transitions <= row_transitions + 1'b1;

                    previous_row_hit <= row_lane_hit;
                    row_lane_pixels  <= 16'd0;
                    row_lane_hit     <= 1'b0;
                end
            end
        end
    end

endmodule

