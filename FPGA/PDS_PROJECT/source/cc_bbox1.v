module cc_bbox1 #(
    parameter integer ROI_X0 = 192,   // 10% of 1920
    parameter integer ROI_X1 = 1728,  // 90%
    parameter integer ROI_Y0 = 108,   // 10% of 1080
    parameter integer ROI_Y1 = 972    // 90%
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        i_vs,
    input  wire        i_hs,
    input  wire        i_de,
    input  wire        i_bin,

    output reg         o_vs,
    output reg         o_valid,
    output reg  [11:0] o_x0,
    output reg  [11:0] o_y0,
   
);

    reg vs_d, hs_d;

    //wire hs_edge = (hs_d ^ i_hs);
    wire hs_fall = (hs_d) & ~i_hs;   // 1->0
    // 只认一个 vs 边沿：上升沿（如果你需要下降沿，把这行换成 vs_fall）
    wire vs_rise = (~vs_d) & i_vs;
    // wire vs_fall = vs_d & ~i_vs;

    reg [11:0] x, y;

    reg        hit_any;
    reg [11:0] x0_acc, y0_acc, x1_acc, y1_acc;
    wire in_roi = (x >= ROI_X0) && (x <= ROI_X1) && (y >= ROI_Y0) && (y <= ROI_Y1);
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            vs_d <= 1'b0;
            hs_d <= 1'b0;
            o_vs <= 1'b0;

            x <= 12'd0;
            y <= 12'd0;

            hit_any <= 1'b0;
            x0_acc  <= 12'hFFF;
            y0_acc  <= 12'hFFF;
            x1_acc  <= 12'd0;
            y1_acc  <= 12'd0;

            o_valid <= 1'b0;
            o_x0 <= 12'd0; o_y0 <= 12'd0; o_x1 <= 12'd0; o_y1 <= 12'd0;
        end else begin
            vs_d <= i_vs;
            hs_d <= i_hs;
            o_vs <= i_vs;

            // 行边界：换行
            if(hs_fall) begin
                x <= 12'd0;
                y <= y + 12'd1;
            end else if(i_de) begin
                x <= x + 12'd1;
            end

            // 帧边界：锁存上一帧并清零累计
            if(vs_rise) begin
                o_valid <= hit_any;
                o_x0    <= hit_any ? x0_acc : 12'd0;
                o_y0    <= hit_any ? y0_acc : 12'd0;
                o_x1    <= hit_any ? x1_acc : 12'd0;
                o_y1    <= hit_any ? y1_acc : 12'd0;

                // reset for new frame
                x <= 12'd0;
                y <= 12'd0;

                hit_any <= 1'b0;
                x0_acc  <= 12'hFFF;
                y0_acc  <= 12'hFFF;
                x1_acc  <= 12'd0;
                y1_acc  <= 12'd0;
            end else begin
                // bbox 更新：只在有效区统计
            if(i_de && i_bin && in_roi) begin
                    hit_any <= 1'b1;
                    if(x < x0_acc) x0_acc <= x;
                    if(y < y0_acc) y0_acc <= y;
                    if(x > x1_acc) x1_acc <= x;
                    if(y > y1_acc) y1_acc <= y;
                end
            end
        end
    end
endmodule