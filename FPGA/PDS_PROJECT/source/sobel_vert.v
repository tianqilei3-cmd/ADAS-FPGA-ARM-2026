module sobel_vert(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_vs,
    input  wire       i_hs,
    input  wire       i_de,
    input  wire [7:0] i_gray,
    output reg        o_vs,
    output reg        o_hs,
    output reg        o_de,
    output reg [9:0]  o_grad
);
    // =========================================================
    // 行缓冲：显式指定用块 RAM，深度 2048（可覆盖 1920 像素）
    // =========================================================
    (* ram_style = "block" *) reg [7:0] line1 [0:2047];
    (* ram_style = "block" *) reg [7:0] line2 [0:2047];

    reg [7:0] line1_q, line2_q;   // 同步读出的数据
    reg [11:0] col;               // 列计数

    // 列计数
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) col <= 0;
        else if(i_de)      col <= col + 1'b1;
        else if(i_hs)      col <= 0;
    end

    // 同步读（1 拍延迟）
    always @(posedge clk) begin
        line1_q <= line1[col];
        line2_q <= line2[col];
    end

    // 写入：先用上一拍读出的 line1_q 更新 line2，再写新像素到 line1
    // 依赖 RAM 的 read-before-write（大多数 FPGA BRAM 默认支持）
    always @(posedge clk) begin
        if(i_de) begin
            line2[col] <= line1_q;
            line1[col] <= i_gray;
        end
    end

    // 3x3 窗口寄存（使用同步读出的 line1_q/line2_q）
    reg [7:0] p00,p01,p02,p10,p11,p12,p20,p21,p22;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            p00<=0;p01<=0;p02<=0;
            p10<=0;p11<=0;p12<=0;
            p20<=0;p21<=0;p22<=0;
        end else begin
            // 横向移位
            p00 <= p01;      p01 <= p02;      p02 <= line2_q;
            p10 <= p11;      p11 <= p12;      p12 <= line1_q;
            p20 <= p21;      p21 <= p22;      p22 <= i_gray;
        end
    end

    // 同步信号延迟 6 拍（匹配窗口/BRAM 延迟）
    reg [5:0] vs_d, hs_d, de_d;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            vs_d <= 0; hs_d <= 0; de_d <= 0;
        end else begin
            vs_d <= {vs_d[4:0], i_vs};
            hs_d <= {hs_d[4:0], i_hs};
            de_d <= {de_d[4:0], i_de};
        end
    end

    // Sobel 垂直分量
    wire signed [10:0] gx = -$signed({3'b0,p00}) - ($signed({3'b0,p10})<<<1) - $signed({3'b0,p20})
                            +  $signed({3'b0,p02}) + ($signed({3'b0,p12})<<<1) + $signed({3'b0,p22});
    wire [10:0] gx_abs = gx[10] ? (~gx + 1'b1) : gx;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_vs   <= 1'b0;
            o_hs   <= 1'b0;
            o_de   <= 1'b0;
            o_grad <= 10'd0;
        end else begin
            o_vs   <= vs_d[5];
            o_hs   <= hs_d[5];
            o_de   <= de_d[5];
            o_grad <= gx_abs[9:0];
        end
    end
endmodule