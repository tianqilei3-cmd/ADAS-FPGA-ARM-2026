module morph_close_3x3(
    input  wire clk,
    input  wire rst_n,
    input  wire i_vs,
    input  wire i_hs,
    input  wire i_de,
    input  wire i_bin,
    output reg  o_vs,
    output reg  o_hs,
    output reg  o_de,
    output reg  o_bin
);

    // ==========================================
    // 第一级：膨胀（Dilation）
    // ==========================================
    
    // 行缓冲
    reg [7:0] dil_line1 [0:2047];
    reg [7:0] dil_line2 [0:2047];
    
    reg [11:0] col;
    wire [7:0] dil_line1_q, dil_line2_q;
    
    // 列计数
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) col <= 0;
        else if(i_de) col <= col + 1'b1;
        else if(i_hs) col <= 0;
    end
    
    // 异步读
    assign dil_line1_q = dil_line1[col];
    assign dil_line2_q = dil_line2[col];
    
    // 写入新数据
    always @(posedge clk) begin
        if(i_de) begin
            dil_line2[col] <= dil_line1_q;
            dil_line1[col] <= {7'b0, i_bin};  // 转成 8bit 方便处理
        end
    end
    
    // 3x3 窗口（膨胀）
    reg [7:0] p00,p01,p02,p10,p11,p12,p20,p21,p22;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            {p00,p01,p02,p10,p11,p12,p20,p21,p22} <= 72'b0;
        end else begin
            p00<=p01; p01<=p02; p02<=dil_line2_q;
            p10<=p11; p11<=p12; p12<=dil_line1_q;
            p20<=p21; p21<=p22; p22<={7'b0, i_bin};
        end
    end
    
    // 膨胀：任意一个为 1 则输出 1
    wire dil = p00[0]|p01[0]|p02[0]|p10[0]|p11[0]|p12[0]|p20[0]|p21[0]|p22[0];
    
    // 同步延迟（膨胀阶段）
    reg [5:0] vs_dil, hs_dil, de_dil;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            vs_dil <= 0; hs_dil <= 0; de_dil <= 0;
        end else begin
            vs_dil <= {vs_dil[4:0], i_vs};
            hs_dil <= {hs_dil[4:0], i_hs};
            de_dil <= {de_dil[4:0], i_de};
        end
    end
    
    // ==========================================
    // 第二级：腐蚀（Erosion）
    // ==========================================
    
    reg [7:0] ero_line1 [0:2047];
    reg [7:0] ero_line2 [0:2047];
    reg [11:0] col2;
    wire [7:0] ero_line1_q, ero_line2_q;
    
    // 列计数
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) col2 <= 0;
        else if(de_dil[0]) col2 <= col2 + 1'b1;
        else if(hs_dil[0]) col2 <= 0;
    end
    
    // 异步读
    assign ero_line1_q = ero_line1[col2];
    assign ero_line2_q = ero_line2[col2];
    
    // 写入膨胀的结果
    always @(posedge clk) begin
        if(de_dil[0]) begin
            ero_line2[col2] <= ero_line1_q;
            ero_line1[col2] <= {7'b0, dil};
        end
    end
    
    // 3x3 窗口（腐蚀）
    reg [7:0] q00,q01,q02,q10,q11,q12,q20,q21,q22;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            {q00,q01,q02,q10,q11,q12,q20,q21,q22} <= 72'b0;
        end else begin
            q00<=q01; q01<=q02; q02<=ero_line2_q;
            q10<=q11; q11<=q12; q12<=ero_line1_q;
            q20<=q21; q21<=q22; q22<={7'b0, dil};
        end
    end
    
    // 腐蚀：全部为 1 才输出 1
    wire erode = q00[0]&q01[0]&q02[0]&q10[0]&q11[0]&q12[0]&q20[0]&q21[0]&q22[0];
    
    // 同步延迟（腐蚀阶段）
    reg [5:0] vs_ero, hs_ero, de_ero;
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            vs_ero <= 0; hs_ero <= 0; de_ero <= 0;
        end else begin
            vs_ero <= {vs_ero[4:0], vs_dil[5]};
            hs_ero <= {hs_ero[4:0], hs_dil[5]};
            de_ero <= {de_ero[4:0], de_dil[5]};
        end
    end
    
    // ==========================================
    // 输出
    // ==========================================
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_vs <= 0; o_hs <= 0; o_de <= 0; o_bin <= 0;
        end else begin
            o_vs  <= vs_ero[5];
            o_hs  <= hs_ero[5];
            o_de  <= de_ero[5];
            o_bin <= erode;
        end
    end

endmodule