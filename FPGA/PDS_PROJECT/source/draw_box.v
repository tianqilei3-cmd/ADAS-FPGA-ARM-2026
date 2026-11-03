module draw_box #(
    parameter LINE_MAX = 1920,
    parameter THICK    = 2
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_vs,
    input  wire       i_hs,
    input  wire       i_de,
    input  wire [7:0] i_r,
    input  wire [7:0] i_g,
    input  wire [7:0] i_b,
    input  wire       box_valid,
    input  wire [11:0] box_x0,
    input  wire [11:0] box_y0,
    input  wire [11:0] box_x1,
    input  wire [11:0] box_y1,
    output reg        o_vs,
    output reg        o_hs,
    output reg        o_de,
    output reg [7:0]  o_r,
    output reg [7:0]  o_g,
    output reg [7:0]  o_b
);
    reg [11:0] x_cnt, y_cnt;
    reg vs_d, hs_d;
    reg [11:0] bx0, by0, bx1, by1;
    reg        bvalid;

    // 同步 & 计数
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_vs<=0; o_hs<=0; vs_d<=0; hs_d<=0; x_cnt<=0; y_cnt<=0;
        end else begin
            vs_d <= i_vs;
            hs_d <= i_hs;
            o_vs <= i_vs;
            o_hs <= i_hs;
            if(i_vs && !vs_d) begin
                x_cnt<=0; y_cnt<=0;
            end else if(i_hs && !hs_d) begin
                x_cnt<=0; y_cnt<=y_cnt+1'b1;
            end else if(i_de) begin
                x_cnt<=x_cnt+1'b1;
            end
        end
    end

    // 帧起始锁存框
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            bx0<=0; by0<=0; bx1<=0; by1<=0; bvalid<=0;
        end else if(i_vs && !vs_d) begin
            if(box_valid) begin
                bx0<=box_x0; by0<=box_y0; bx1<=box_x1; by1<=box_y1; bvalid<=1'b1;
            end else bvalid<=1'b0;
        end
    end

    wire on_left   = bvalid && (x_cnt>=bx0) && (x_cnt<bx0+THICK) && (y_cnt>=by0) && (y_cnt<=by1);
    wire on_right  = bvalid && (x_cnt<=bx1) && (x_cnt>bx1-THICK) && (y_cnt>=by0) && (y_cnt<=by1);
    wire on_top    = bvalid && (y_cnt>=by0) && (y_cnt<by0+THICK) && (x_cnt>=bx0) && (x_cnt<=bx1);
    wire on_bottom = bvalid && (y_cnt<=by1) && (y_cnt>by1-THICK) && (x_cnt>=bx0) && (x_cnt<=bx1);
    wire on_box    = on_left | on_right | on_top | on_bottom;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_de<=0; o_r<=0; o_g<=0; o_b<=0;
        end else begin
            o_de <= i_de;
            if(i_de) begin
                if(on_box) begin
                    o_r <= 8'hFF; o_g <= 8'h00; o_b <= 8'h00; // 红框
                end else begin
                    o_r <= i_r; o_g <= i_g; o_b <= i_b;
                end
            end else begin
                o_r<=0; o_g<=0; o_b<=0;
            end
        end
    end
endmodule