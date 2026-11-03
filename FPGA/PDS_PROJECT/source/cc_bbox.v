module cc_bbox #(
    parameter integer ROI_X0   = 0,
    parameter integer ROI_X1   = 1919,
    parameter integer ROI_Y0   = 0,
    parameter integer ROI_Y1   = 1079,
    parameter integer ROW_TH   = 150,
    parameter integer GAP_MAX  = 10,
    parameter integer MIN_BAND_H = 15
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        i_vs,
    input  wire        i_hs,
    input  wire        i_de,
    input  wire        i_bin,

    output reg         o_vs,
    output reg         o_hs,
    output reg         o_de,
    output reg         o_valid,
    output reg  [11:0] o_x0,
    output reg  [11:0] o_y0,
    output reg  [11:0] o_x1,
    output reg  [11:0] o_y1
);

    // ========== 内部坐标计数 ==========
    reg [11:0] x, y;
    reg vs_d, hs_d, de_d;

    // 边界检测
    wire de_fall = de_d & ~i_de;    // DE 下降沿 = 行结束
    wire vs_fall = vs_d & ~i_vs;    // VS 下降沿 = 帧结束

    // ========== BBox 累积 ==========
    reg        hit_any;
    reg [11:0] x0_acc, y0_acc, x1_acc, y1_acc;

    // ========== 行投影统计 ==========
    reg [11:0] row_cnt;
    reg        band_hit;
    reg [11:0] band_y0, band_y1;
    reg [7:0]  gap_cnt;
    reg [11:0] band_h;

    // ========== 组合逻辑 ==========
    wire in_roi = (x >= ROI_X0) && (x <= ROI_X1) && (y >= ROI_Y0) && (y <= ROI_Y1);
    wire in_band = band_hit && (y >= band_y0) && (y <= band_y1);

    // ========== 主时序逻辑 ==========
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            // 初始化所有寄存器
            vs_d <= 1'b0;
            hs_d <= 1'b0;
            de_d <= 1'b0;
            o_vs <= 1'b0;
            o_hs <= 1'b0;
            o_de <= 1'b0;

            x <= 12'd0;
            y <= 12'd0;

            hit_any <= 1'b0;
            x0_acc  <= 12'hFFF;
            y0_acc  <= 12'hFFF;
            x1_acc  <= 12'd0;
            y1_acc  <= 12'd0;

            o_valid <= 1'b0;
            o_x0    <= 12'd0;
            o_y0    <= 12'd0;
            o_x1    <= 12'd0;
            o_y1    <= 12'd0;

            row_cnt  <= 12'd0;
            band_hit <= 1'b0;
            band_y0  <= 12'hFFF;
            band_y1  <= 12'd0;
            gap_cnt  <= 8'd0;
            band_h   <= 12'd0;

        end else begin
            // ========== 延迟同步信号 ==========
            vs_d <= i_vs;
            hs_d <= i_hs;
            de_d <= i_de;

            // ========== ✅ 同步信号直通（无延迟，与坐标同步） ==========
            o_vs <= i_vs;
            o_hs <= i_hs;
            o_de <= i_de;

            // ========== 帧边界：VS 下降沿时锁存结果 & 重置 ==========
            if(vs_fall) begin
                // 锁存当前帧的检测结果
                if(hit_any && band_h >= MIN_BAND_H) begin
                    o_valid <= 1'b1;
                    o_x0    <= x0_acc;
                    o_y0    <= y0_acc;
                    o_x1    <= x1_acc;
                    o_y1    <= y1_acc;
                end else begin
                    o_valid <= 1'b0;
                end

                // 重置所有累积
                x       <= 12'd0;
                y       <= 12'd0;
                hit_any <= 1'b0;
                x0_acc  <= 12'hFFF;
                y0_acc  <= 12'hFFF;
                x1_acc  <= 12'd0;
                y1_acc  <= 12'd0;

                row_cnt  <= 12'd0;
                band_hit <= 1'b0;
                band_y0  <= 12'hFFF;
                band_y1  <= 12'd0;
                gap_cnt  <= 8'd0;
                band_h   <= 12'd0;

            end else begin
                // ========== 坐标计数 ==========
                if(i_de) begin
                    x <= x + 12'd1;
                end

                // DE 下降沿时，y 递增
                if(de_fall) begin
                    x <= 12'd0;
                    y <= y + 12'd1;
                end

                // ========== 行投影统计 ==========
                if(i_de && i_bin && in_roi) begin
                    row_cnt <= row_cnt + 12'd1;
                end

                // ========== 行末处理：根据行投影更新带 ==========
                if(de_fall) begin
                    if(row_cnt >= ROW_TH) begin
                        // 本行像素数超过阈值：命中
                        if(!band_hit) begin
                            band_hit <= 1'b1;
                            band_y0  <= y;
                            band_y1  <= y;
                            band_h   <= 12'd1;
                        end else begin
                            if(y < band_y0) band_y0 <= y;
                            if(y > band_y1) band_y1 <= y;
                            band_h <= band_y1 - band_y0 + 12'd1;
                        end
                        gap_cnt <= 8'd0;
                    end else if(band_hit) begin
                        if(gap_cnt < GAP_MAX) begin
                            gap_cnt <= gap_cnt + 8'd1;
                        end
                    end

                    row_cnt <= 12'd0;
                end

                // ========== 在带内累积 BBox ==========
                if(i_de && i_bin && in_roi && in_band) begin
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