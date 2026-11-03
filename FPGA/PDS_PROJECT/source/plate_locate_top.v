module plate_locate_top #(
    parameter H_ACTIVE = 1920,
    parameter V_ACTIVE = 1080
)(
    input  wire         pix_clk,
    input  wire         rst_n,
    input  wire         i_vs,
    input  wire         i_hs,
    input  wire         i_de,
    input  wire [7:0]   i_r,
    input  wire [7:0]   i_g,
    input  wire [7:0]   i_b,
    input  wire [7:0]   i_s_min,
    input  wire [7:0]   i_v_min,
    input  wire [7:0]   blue_h_min,
    input  wire [7:0]   blue_h_max,
    input  wire [7:0]   green_h_min,
    input  wire [7:0]   green_h_max,
    input  wire [7:0]   edge_th,
    
    output wire         o_mask_color,
    output wire         o_edge_close,
    output wire         o_box_valid,
    output wire [11:0]  o_x0, o_y0, o_x1, o_y1,
    
    output wire         o_vs,
    output wire         o_hs,
    output wire         o_de,
    output wire [7:0]   o_r,
    output wire [7:0]   o_g,
    output wire [7:0]   o_b,
    output wire [11:0]  o_x,
    output wire [11:0]  o_y,
    
    // ✅ 新增：cc_bbox 调试输出
    output wire [11:0]  o_dbg_cc_x,
    output wire [11:0]  o_dbg_cc_y,
    output wire [11:0]  o_dbg_cc_row_cnt,
    output wire         o_dbg_cc_band_hit,
    output wire         o_dbg_cc_hit_any
);

    // ==========================================
    // 1. HSV 颜色空间转换
    // ==========================================
    wire [7:0] h, s, v;
    wire hsv_vs, hsv_hs, hsv_de;
    
    hsv_stream u_hsv (
        .clk   (pix_clk),
        .rst_n (rst_n),
        .i_vs  (i_vs),
        .i_hs  (i_hs),
        .i_de  (i_de),
        .i_r   (i_r),
        .i_g   (i_g),
        .i_b   (i_b),
        .o_vs  (hsv_vs),
        .o_hs  (hsv_hs),
        .o_de  (hsv_de),
        .o_h   (h),
        .o_s   (s),
        .o_v   (v)
    );

    // ==========================================
    // 2. 局部 Gamma 映射（增强对比度）
    // ==========================================
    wire gamma_vs, gamma_hs, gamma_de;
    wire [7:0] v_gamma;
    
    gamma_local #(.WIN_W(64), .WIN_H(32)) u_gamma (
        .clk   (pix_clk),
        .rst_n (rst_n),
        .i_vs  (hsv_vs),
        .i_hs  (hsv_hs),
        .i_de  (hsv_de),
        .i_v   (v),
        .o_vs  (gamma_vs),
        .o_hs  (gamma_hs),
        .o_de  (gamma_de),
        .o_v   (v_gamma)
    );

    // ==========================================
    // 3. 颜色掩膜提取 (提取蓝色/绿色区域)
    // ==========================================
    wire [7:0] s_min_use = (i_s_min < 8'd10) ? 8'd10 :
                           (i_s_min > 8'd120) ? 8'd120 : i_s_min;

    wire [7:0] v_min_use = (i_v_min < 8'd10) ? 8'd10 :
                           (i_v_min > 8'd200) ? 8'd200 : i_v_min;

    wire mask_vs, mask_hs, mask_de;
    wire mask_color;
    
    color_mask u_color_mask (
        .clk        (pix_clk),
        .rst_n      (rst_n),
        .i_vs       (gamma_vs),
        .i_hs       (gamma_hs),
        .i_de       (gamma_de),
        .i_h        (h),
        .i_s        (s),
        .i_v        (v_gamma),
        .blue_h_min (blue_h_min),
        .blue_h_max (blue_h_max),
        .green_h_min(green_h_min),
        .green_h_max(green_h_max),
        .s_min      (s_min_use),
        .v_min      (v_min_use),
        .o_vs       (mask_vs),
        .o_hs       (mask_hs),
        .o_de       (mask_de),
        .o_mask     (mask_color)
    );
    
    assign o_mask_color = mask_color;

    // ==========================================
    // 4. ROI提取 与 Sobel 边缘检测
    // ==========================================
    wire [7:0] masked_gamma = mask_color ? v_gamma : 8'd0;

    wire sobel_vs, sobel_hs, sobel_de;
    wire [9:0] sobel_mag;
    
    sobel_vert u_sobel_vert (
        .clk    (pix_clk),
        .rst_n  (rst_n),
        .i_vs   (mask_vs),
        .i_hs   (mask_hs),
        .i_de   (mask_de),
        .i_gray (masked_gamma),
        .o_vs   (sobel_vs),
        .o_hs   (sobel_hs),
        .o_de   (sobel_de),
        .o_grad (sobel_mag)
    );
    
    wire edge_bin = sobel_de && (sobel_mag > edge_th);

    // ==========================================
    // 5. 形态学闭运算 (连接密集的车牌边缘)
    // ==========================================
    wire morph_vs, morph_hs, morph_de, morph_bin;
    
    morph_close_3x3 u_morph (
        .clk   (pix_clk),
        .rst_n (rst_n),
        .i_vs  (sobel_vs),
        .i_hs  (sobel_hs),
        .i_de  (sobel_de),
        .i_bin (edge_bin),
        .o_vs  (morph_vs),
        .o_hs  (morph_hs),
        .o_de  (morph_de),
        .o_bin (morph_bin)
    );

    wire fuse_bin = morph_bin && morph_de;
    assign o_edge_close = fuse_bin;

    // ==========================================
    // 6. 连通域 BBox 提取（自己计数坐标）
    // ==========================================
    wire bbox_vs, bbox_hs, bbox_de;
    wire bbox_valid;
    wire [11:0] bbox_x0, bbox_y0, bbox_x1, bbox_y1;
    
    // ✅ 调试信号声明
    wire [11:0] dbg_cc_x, dbg_cc_y, dbg_cc_row_cnt;
    wire dbg_cc_band_hit, dbg_cc_hit_any;
    
    cc_bbox #(
        .ROI_X0(0),
        .ROI_X1(1919),
        .ROI_Y0(0),
        .ROI_Y1(1079),
        .ROW_TH(150),
        .GAP_MAX(10),
        .MIN_BAND_H(15)
    ) u_cc_bbox (
        .clk    (pix_clk),
        .rst_n  (rst_n),
        .i_vs   (morph_vs),
        .i_hs   (morph_hs),
        .i_de   (morph_de),
        .i_bin  (morph_bin),
        .o_vs   (bbox_vs),
        .o_hs   (bbox_hs),
        .o_de   (bbox_de),
        .o_valid(bbox_valid),
        .o_x0   (bbox_x0),
        .o_y0   (bbox_y0),
        .o_x1   (bbox_x1),
        .o_y1   (bbox_y1)
    );

    assign o_box_valid = bbox_valid;
    assign o_x0 = bbox_x0;
    assign o_y0 = bbox_y0;
    assign o_x1 = bbox_x1;
    assign o_y1 = bbox_y1;

    // ==========================================
    // 7. 原始图像直通输出
    // ==========================================
    assign o_vs = i_vs;
    assign o_hs = i_hs;
    assign o_de = i_de;
    assign o_r  = i_r;
    assign o_g  = i_g;
    assign o_b  = i_b;

    // ==========================================
    // 8. 坐标输出
    // ==========================================
    assign o_x = 12'd0;
    assign o_y = 12'd0;

   

endmodule