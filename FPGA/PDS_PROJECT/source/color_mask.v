module color_mask(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_vs,
    input  wire       i_hs,
    input  wire       i_de,
    input  wire [7:0] i_h,
    input  wire [7:0] i_s,
    input  wire [7:0] i_v,
    input  wire [7:0] blue_h_min,
    input  wire [7:0] blue_h_max,
    input  wire [7:0] green_h_min,
    input  wire [7:0] green_h_max,
    input  wire [7:0] s_min,
    input  wire [7:0] v_min,
    output reg        o_vs,
    output reg        o_hs,
    output reg        o_de,
    output reg        o_mask
);

    // 你现在 hsv_stream 的 H 是“粗编码”（30/90/140）
    // 因此这里直接用范围粗判定即可，更稳更容易出框
    wire hit_blue  = (i_h >= 8'd120) && (i_h <= 8'd179); // 覆盖 140
    wire hit_green = (i_h >= 8'd70)  && (i_h <= 8'd120); // 覆盖 90
    wire hit_redyl = (i_h >= 8'd0)   && (i_h <= 8'd60);  // 覆盖 30（黄/红方向）

    // S=diff(max-min)，建议 s_min 从 20~40 起步；v_min 按亮度定
    wire hit_sv    = (i_s >= s_min) && (i_v >= v_min);

    // 多色都允许（最快先把框跑出来）
    wire mask_now  = i_de && hit_sv && (hit_blue || hit_green || hit_redyl);

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_vs<=0; o_hs<=0; o_de<=0; o_mask<=0;
        end else begin
            o_vs   <= i_vs;
            o_hs   <= i_hs;
            o_de   <= i_de;
            o_mask <= mask_now;
        end
    end
endmodule