`timescale 1ns/1ps

//=============================================================================
// Copyright (c) 2026 ADAS Project Team
// Project : FPGA ADAS Road Perception System
// Module  : night_enhance
// Version : 1.0
// Author  : ADAS Project Team
// Function: 可关闭的单拍夜间亮度/对比度增强，通道结果自动饱和到8位。
//=============================================================================
module night_enhance #(
    parameter [7:0] BRIGHTNESS_OFFSET = 8'd20
)(
    input  wire       pix_clk,
    input  wire       rst_n,          // 同步低有效复位
    input  wire       enhance_enable,
    input  wire       i_vsync,
    input  wire       i_hsync,
    input  wire       i_de,
    input  wire [7:0] i_r,
    input  wire [7:0] i_g,
    input  wire [7:0] i_b,

    output reg        o_vsync,
    output reg        o_hsync,
    output reg        o_de,
    output reg  [7:0] o_r,
    output reg  [7:0] o_g,
    output reg  [7:0] o_b
);

    function [7:0] enhance_channel;
        input [7:0] channel_in;
        reg [9:0] expanded;
        begin
            // 1.5倍增益加偏置：channel + channel/2 + offset。
            expanded = {2'b0, channel_in} + {3'b0, channel_in[7:1]} +
                       BRIGHTNESS_OFFSET;
            if (expanded > 10'd255)
                enhance_channel = 8'hFF;
            else
                enhance_channel = expanded[7:0];
        end
    endfunction

    always @(posedge pix_clk) begin
        if (!rst_n) begin
            o_vsync <= 1'b0;
            o_hsync <= 1'b0;
            o_de    <= 1'b0;
            o_r     <= 8'd0;
            o_g     <= 8'd0;
            o_b     <= 8'd0;
        end
        else begin
            o_vsync <= i_vsync;
            o_hsync <= i_hsync;
            o_de    <= i_de;
            if (enhance_enable) begin
                o_r <= enhance_channel(i_r);
                o_g <= enhance_channel(i_g);
                o_b <= enhance_channel(i_b);
            end
            else begin
                o_r <= i_r;
                o_g <= i_g;
                o_b <= i_b;
            end
        end
    end

endmodule

