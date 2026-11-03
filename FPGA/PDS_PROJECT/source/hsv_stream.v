module hsv_stream(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_vs,
    input  wire       i_hs,
    input  wire       i_de,
    input  wire [7:0] i_r,
    input  wire [7:0] i_g,
    input  wire [7:0] i_b,
    output reg        o_vs,
    output reg        o_hs,
    output reg        o_de,
    output reg  [7:0] o_h,
    output reg  [7:0] o_s,
    output reg  [7:0] o_v
);
    // 1) 同步信号打一拍
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_vs <= 1'b0;
            o_hs <= 1'b0;
            o_de <= 1'b0;
        end else begin
            o_vs <= i_vs;
            o_hs <= i_hs;
            o_de <= i_de;
        end
    end

    // 2) 计算 max/min 和 diff
    wire [7:0] max1 = (i_r > i_g) ? i_r : i_g;
    wire [7:0] maxc = (max1 > i_b) ? max1 : i_b;
    wire [7:0] min1 = (i_r < i_g) ? i_r : i_g;
    wire [7:0] minc = (min1 < i_b) ? min1 : i_b;
    wire [7:0] diff = maxc - minc;

    // 3) V：亮度
    wire [7:0] v_tmp = maxc;

    // 4) S：用 diff 近似“颜色强度”（不做 diff/max 的标准化，避免除法）
    // 注意：这样 o_s 的意义变了，后面 s_min 建议用 20~60 量级试
    wire [7:0] s_tmp = diff;

    // 5) H：粗略分色编码（只要能分蓝/绿/黄方向即可）
    // - B 最大：蓝/青区域
    // - G 最大：绿区域
    // - R 最大：红/黄区域
    reg [7:0] h_tmp;
    always @(*) begin
        if(diff == 0) begin
            h_tmp = 8'd0;
        end else if(maxc == i_b) begin
            h_tmp = 8'd140; // 蓝
        end else if(maxc == i_g) begin
            h_tmp = 8'd90;  // 绿
        end else begin
            h_tmp = 8'd30;  // 红/黄
        end
    end

    // 6) 输出寄存
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_h <= 8'd0;
            o_s <= 8'd0;
            o_v <= 8'd0;
        end else begin
            o_h <= h_tmp;
            o_s <= s_tmp;
            o_v <= v_tmp;
        end
    end
endmodule