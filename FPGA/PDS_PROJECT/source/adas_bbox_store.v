`timescale 1ns/1ps
//=============================================================================
// Module  : adas_bbox_store
// Function: 按camera_id/frame_id接收最多4个ARM检测框；新状态帧到来时清除旧框。
//           扁平向量中slot0位于最低位，便于Verilog-2001顶层连接。
//=============================================================================
module adas_bbox_store(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        status_valid,
    input  wire [1:0]  status_camera_id,
    input  wire [15:0] status_frame_id,
    input  wire        bbox_valid,
    input  wire [1:0]  bbox_camera_id,
    input  wire [15:0] bbox_frame_id,
    input  wire [1:0]  bbox_slot,
    input  wire        bbox_model_id,
    input  wire [7:0]  bbox_class_id,
    input  wire [7:0]  bbox_confidence,
    input  wire [15:0] bbox_x1,
    input  wire [15:0] bbox_y1,
    input  wire [15:0] bbox_x2,
    input  wire [15:0] bbox_y2,
    output reg  [3:0]  boxes_valid,
    output reg  [3:0]  boxes_model_id,
    output reg  [31:0] boxes_class_id,
    output reg  [31:0] boxes_confidence,
    output reg  [63:0] boxes_x1,
    output reg  [63:0] boxes_y1,
    output reg  [63:0] boxes_x2,
    output reg  [63:0] boxes_y2,
    output reg         update_toggle
);
    reg [1:0] current_camera_id;
    reg [15:0] current_frame_id;
    always @(posedge clk) begin
        if (!rst_n) begin
            current_camera_id<=0; current_frame_id<=0; boxes_valid<=0;
            boxes_model_id<=0; boxes_class_id<=0; boxes_confidence<=0;
            boxes_x1<=0; boxes_y1<=0; boxes_x2<=0; boxes_y2<=0; update_toggle<=0;
        end else if (status_valid) begin
            current_camera_id<=status_camera_id; current_frame_id<=status_frame_id;
            boxes_valid<=0; update_toggle<=~update_toggle;
        end else if (bbox_valid && bbox_camera_id==current_camera_id &&
                     bbox_frame_id==current_frame_id) begin
            update_toggle<=~update_toggle;
            case (bbox_slot)
                2'd0: begin boxes_valid[0]<=1; boxes_model_id[0]<=bbox_model_id;
                    boxes_class_id[7:0]<=bbox_class_id; boxes_confidence[7:0]<=bbox_confidence;
                    boxes_x1[15:0]<=bbox_x1; boxes_y1[15:0]<=bbox_y1;
                    boxes_x2[15:0]<=bbox_x2; boxes_y2[15:0]<=bbox_y2; end
                2'd1: begin boxes_valid[1]<=1; boxes_model_id[1]<=bbox_model_id;
                    boxes_class_id[15:8]<=bbox_class_id; boxes_confidence[15:8]<=bbox_confidence;
                    boxes_x1[31:16]<=bbox_x1; boxes_y1[31:16]<=bbox_y1;
                    boxes_x2[31:16]<=bbox_x2; boxes_y2[31:16]<=bbox_y2; end
                2'd2: begin boxes_valid[2]<=1; boxes_model_id[2]<=bbox_model_id;
                    boxes_class_id[23:16]<=bbox_class_id; boxes_confidence[23:16]<=bbox_confidence;
                    boxes_x1[47:32]<=bbox_x1; boxes_y1[47:32]<=bbox_y1;
                    boxes_x2[47:32]<=bbox_x2; boxes_y2[47:32]<=bbox_y2; end
                default: begin boxes_valid[3]<=1; boxes_model_id[3]<=bbox_model_id;
                    boxes_class_id[31:24]<=bbox_class_id; boxes_confidence[31:24]<=bbox_confidence;
                    boxes_x1[63:48]<=bbox_x1; boxes_y1[63:48]<=bbox_y1;
                    boxes_x2[63:48]<=bbox_x2; boxes_y2[63:48]<=bbox_y2; end
            endcase
        end
    end
endmodule
