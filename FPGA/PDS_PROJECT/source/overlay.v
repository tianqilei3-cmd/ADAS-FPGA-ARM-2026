`timescale 1ns/1ps

//=============================================================================
// Copyright (c) 2026 ADAS Project Team
// Project : FPGA ADAS Road Perception System
// Module  : overlay
// Version : 2.0
// Author  : ADAS Project Team
// Function: 叠加STOP、报警、动作箭头以及ARM回传的最多4个检测框。
//=============================================================================
module overlay #(
    parameter integer H_ACTIVE = 1920,
    parameter integer V_ACTIVE = 1080,
    parameter integer TEXT_X   = 32,
    parameter integer TEXT_Y   = 18
)(
    input  wire        pix_clk,
    input  wire        rst_n,          // 同步低有效复位
    input  wire        overlay_enable,
    input  wire        i_vsync,
    input  wire        i_hsync,
    input  wire        i_de,
    input  wire [11:0] i_x,
    input  wire [11:0] i_y,
    input  wire [7:0]  i_r,
    input  wire [7:0]  i_g,
    input  wire [7:0]  i_b,
    input  wire        stop_enable,
    input  wire        alarm_enable,
    input  wire [1:0]  decision_code,
    input  wire [2:0]  action_code,
    input  wire [2:0]  light_state,
    input  wire        lane_departure_warning,
    input  wire [3:0]  boxes_valid,
    input  wire [3:0]  boxes_model_id,
    input  wire [31:0] boxes_class_id,
    input  wire [63:0] boxes_x1,
    input  wire [63:0] boxes_y1,
    input  wire [63:0] boxes_x2,
    input  wire [63:0] boxes_y2,

    output reg         o_vsync,
    output reg         o_hsync,
    output reg         o_de,
    output reg  [7:0]  o_r,
    output reg  [7:0]  o_g,
    output reg  [7:0]  o_b
);

    reg  [1:0] char_select;
    reg  [2:0] glyph_column;
    reg  [4:0] glyph_bits;
    reg        text_pixel;
    reg  [11:0] character_x;
    wire [2:0] glyph_row_index;
    wire       in_text_height;
    wire       in_alarm_border;
    wire       in_lane_warning_bar;
    wire       in_status_block;
    wire       in_box0, in_box1, in_box2, in_box3;
    wire       in_any_box;
    reg  [1:0] box_index;
    reg  [7:0] box_r, box_g, box_b;
    wire [11:0] arrow_cx = H_ACTIVE/2;
    wire in_straight_arrow = (action_code==3'd2) &&
        ((((i_x>=arrow_cx-3)&&(i_x<=arrow_cx+3)&&(i_y>=80)&&(i_y<=150))) ||
         ((i_y>=80)&&(i_y<=105)&&
          ((i_x+i_y>=arrow_cx+77)&&(i_x+i_y<=arrow_cx+83) ||
           (i_x-i_y>=arrow_cx-83)&&(i_x-i_y<=arrow_cx-77))));
    wire in_left_arrow = ((action_code==3'd3)||(action_code==3'd5)) &&
        (((i_y>=112)&&(i_y<=118)&&(i_x>=arrow_cx-70)&&(i_x<=arrow_cx+20)) ||
         ((i_x>=arrow_cx-70)&&(i_x<=arrow_cx-45) &&
          ((i_x+i_y>=arrow_cx+42)&&(i_x+i_y<=arrow_cx+48) ||
           (i_y-i_x>=118-arrow_cx)&&(i_y-i_x<=124-arrow_cx))));
    wire in_right_arrow = ((action_code==3'd4)||(action_code==3'd6)) &&
        (((i_y>=112)&&(i_y<=118)&&(i_x>=arrow_cx-20)&&(i_x<=arrow_cx+70)) ||
         ((i_x>=arrow_cx+45)&&(i_x<=arrow_cx+70) &&
          ((i_x+i_y>=arrow_cx+162)&&(i_x+i_y<=arrow_cx+168) ||
           (i_x-i_y>=arrow_cx-73)&&(i_x-i_y<=arrow_cx-67))));
    wire in_action_arrow = i_de && (in_straight_arrow || in_left_arrow || in_right_arrow);

    assign glyph_row_index = (i_y - TEXT_Y) >> 2;
    assign in_text_height = (i_y >= TEXT_Y) && (i_y < TEXT_Y + 28);
    assign in_alarm_border = i_de && alarm_enable &&
                             ((i_x < 12'd4) || (i_x >= H_ACTIVE - 4) ||
                              (i_y < 12'd4) || (i_y >= V_ACTIVE - 4));
    assign in_lane_warning_bar = i_de && lane_departure_warning &&
                                 (i_y >= V_ACTIVE - 32);
    assign in_status_block = i_de && (i_x >= H_ACTIVE - 80) &&
                             (i_x < H_ACTIVE - 16) &&
                             (i_y >= 16) && (i_y < 80);

    assign in_box0 = boxes_valid[0] && i_de && i_x>=boxes_x1[15:0] && i_x<=boxes_x2[15:0] &&
        i_y>=boxes_y1[15:0] && i_y<=boxes_y2[15:0] &&
        ((i_x<boxes_x1[15:0]+3)||(i_x+3>boxes_x2[15:0])||(i_y<boxes_y1[15:0]+3)||(i_y+3>boxes_y2[15:0]));
    assign in_box1 = boxes_valid[1] && i_de && i_x>=boxes_x1[31:16] && i_x<=boxes_x2[31:16] &&
        i_y>=boxes_y1[31:16] && i_y<=boxes_y2[31:16] &&
        ((i_x<boxes_x1[31:16]+3)||(i_x+3>boxes_x2[31:16])||(i_y<boxes_y1[31:16]+3)||(i_y+3>boxes_y2[31:16]));
    assign in_box2 = boxes_valid[2] && i_de && i_x>=boxes_x1[47:32] && i_x<=boxes_x2[47:32] &&
        i_y>=boxes_y1[47:32] && i_y<=boxes_y2[47:32] &&
        ((i_x<boxes_x1[47:32]+3)||(i_x+3>boxes_x2[47:32])||(i_y<boxes_y1[47:32]+3)||(i_y+3>boxes_y2[47:32]));
    assign in_box3 = boxes_valid[3] && i_de && i_x>=boxes_x1[63:48] && i_x<=boxes_x2[63:48] &&
        i_y>=boxes_y1[63:48] && i_y<=boxes_y2[63:48] &&
        ((i_x<boxes_x1[63:48]+3)||(i_x+3>boxes_x2[63:48])||(i_y<boxes_y1[63:48]+3)||(i_y+3>boxes_y2[63:48]));
    assign in_any_box = in_box0 || in_box1 || in_box2 || in_box3;

    always @(*) begin
        if (in_box0) box_index=2'd0;
        else if (in_box1) box_index=2'd1;
        else if (in_box2) box_index=2'd2;
        else box_index=2'd3;
        box_r=8'h00; box_g=8'hFF; box_b=8'hFF;
        if (boxes_model_id[box_index]) begin
            if (boxes_class_id[box_index*8 +: 8]==8'd0) begin box_r=8'hFF; box_g=0; box_b=0; end
            else if (boxes_class_id[box_index*8 +: 8]==8'd2) begin box_r=0; box_g=8'hFF; box_b=0; end
            else begin box_r=8'hFF; box_g=8'hD0; box_b=0; end
        end else if (boxes_class_id[box_index*8 +: 8]==8'd3) begin
            box_r=8'hD0; box_g=0; box_b=8'hFF;
        end
    end

    function [4:0] stop_glyph_row;
        input [1:0] character;
        input [2:0] row_index;
        begin
            case (character)
                2'd0: begin // S
                    case (row_index)
                        3'd0: stop_glyph_row = 5'b11111;
                        3'd1: stop_glyph_row = 5'b10000;
                        3'd2: stop_glyph_row = 5'b10000;
                        3'd3: stop_glyph_row = 5'b11111;
                        3'd4: stop_glyph_row = 5'b00001;
                        3'd5: stop_glyph_row = 5'b00001;
                        default: stop_glyph_row = 5'b11111;
                    endcase
                end
                2'd1: begin // T
                    case (row_index)
                        3'd0: stop_glyph_row = 5'b11111;
                        3'd1, 3'd2, 3'd3, 3'd4, 3'd5:
                              stop_glyph_row = 5'b00100;
                        default: stop_glyph_row = 5'b00100;
                    endcase
                end
                2'd2: begin // O
                    case (row_index)
                        3'd0: stop_glyph_row = 5'b11111;
                        3'd1, 3'd2, 3'd3, 3'd4, 3'd5:
                              stop_glyph_row = 5'b10001;
                        default: stop_glyph_row = 5'b11111;
                    endcase
                end
                default: begin // P
                    case (row_index)
                        3'd0: stop_glyph_row = 5'b11110;
                        3'd1: stop_glyph_row = 5'b10001;
                        3'd2: stop_glyph_row = 5'b10001;
                        3'd3: stop_glyph_row = 5'b11110;
                        3'd4, 3'd5: stop_glyph_row = 5'b10000;
                        default: stop_glyph_row = 5'b10000;
                    endcase
                end
            endcase
        end
    endfunction

    // 5x7字体放大4倍，每个字符占24像素宽，其中20像素为字形。
    always @(*) begin
        char_select = 2'd0;
        character_x = 12'd0;
        glyph_column = 3'd0;
        glyph_bits = 5'b00000;
        text_pixel = 1'b0;

        if (in_text_height) begin
            if ((i_x >= TEXT_X) && (i_x < TEXT_X + 20)) begin
                char_select = 2'd0;
                character_x = i_x - TEXT_X;
                text_pixel = 1'b1;
            end
            else if ((i_x >= TEXT_X + 24) && (i_x < TEXT_X + 44)) begin
                char_select = 2'd1;
                character_x = i_x - (TEXT_X + 24);
                text_pixel = 1'b1;
            end
            else if ((i_x >= TEXT_X + 48) && (i_x < TEXT_X + 68)) begin
                char_select = 2'd2;
                character_x = i_x - (TEXT_X + 48);
                text_pixel = 1'b1;
            end
            else if ((i_x >= TEXT_X + 72) && (i_x < TEXT_X + 92)) begin
                char_select = 2'd3;
                character_x = i_x - (TEXT_X + 72);
                text_pixel = 1'b1;
            end

            glyph_column = character_x[4:2];
            glyph_bits = stop_glyph_row(char_select, glyph_row_index);
            if (text_pixel && (glyph_column < 5))
                text_pixel = glyph_bits[4 - glyph_column];
            else
                text_pixel = 1'b0;
        end
    end

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

            if (!overlay_enable || !i_de) begin
                o_r <= i_r;
                o_g <= i_g;
                o_b <= i_b;
            end
            else if (stop_enable && text_pixel) begin
                // STOP白色文字。
                o_r <= 8'hFF;
                o_g <= 8'hFF;
                o_b <= 8'hFF;
            end
            else if (stop_enable && (i_y < 12'd64)) begin
                // 停车状态红色顶部条。
                o_r <= 8'hD0;
                o_g <= 8'h00;
                o_b <= 8'h00;
            end
            else if (in_alarm_border) begin
                // 危险报警黄色边框。
                o_r <= 8'hFF;
                o_g <= 8'hD0;
                o_b <= 8'h00;
            end
            else if (in_any_box) begin
                o_r <= box_r;
                o_g <= box_g;
                o_b <= box_b;
            end
            else if (in_action_arrow) begin
                o_r <= 8'h00;
                o_g <= 8'hFF;
                o_b <= 8'h40;
            end
            else if (in_lane_warning_bar) begin
                // 车道偏离橙色底部条。
                o_r <= 8'hFF;
                o_g <= 8'h60;
                o_b <= 8'h00;
            end
            else if (in_status_block && (decision_code != 2'b00)) begin
                // 右上角场景块：路障黄、红灯红、行人紫。
                case (decision_code)
                    2'b01: begin o_r <= 8'hFF; o_g <= 8'hC0; o_b <= 8'h00; end
                    2'b10: begin o_r <= 8'hFF; o_g <= 8'h00; o_b <= 8'h00; end
                    default: begin o_r <= 8'hD0; o_g <= 8'h00; o_b <= 8'hFF; end
                endcase
            end
            else if (in_status_block && (light_state == 3'd3)) begin
                o_r <= 8'h00; o_g <= 8'hFF; o_b <= 8'h00;
            end
            else begin
                o_r <= i_r;
                o_g <= i_g;
                o_b <= i_b;
            end
        end
    end

endmodule

