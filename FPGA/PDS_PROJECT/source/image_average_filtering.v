module image_average_filtering
(
	input   wire				i_clk,
	input   wire				i_rst_n,

	input	wire				i_hsyn,
	input	wire				i_vsyn,
	input	wire				i_en,
	input	wire [7:0]			i_r,
	input	wire [7:0] 			i_g,
	input	wire [7:0] 			i_b,		
	
	output	wire 				o_hs,
	output	wire 				o_vs,
	output	wire 				o_en,	
	output  wire [7:0]			o_r,
	output  wire [7:0]			o_g,
	output  wire [7:0]			o_b	
);

reg	[3:0]		i_hsyn_d;
reg	[3:0]		i_vsyn_d;
reg	[3:0]		i_en_d;

reg [11:0]		sum_r;
reg [11:0]		sum_g;
reg [11:0]		sum_b;

// 近似除法: 除以9 ≈ ×364 >> 11 (364/2048 ≈ 0.1777, 实际1/9≈0.1111)
// 更精确的近似: 除以9 ≈ (×455 + 256) >> 12 (455/4096 ≈ 0.111, 加上256用于四舍五入)
reg [19:0]		avg_r_temp;
reg [19:0]		avg_g_temp;
reg [19:0]		avg_b_temp;
reg [7:0]		avg_r_final;
reg [7:0]		avg_g_final;
reg [7:0]		avg_b_final;

wire [7:0]	r_temp_11;
wire [7:0]	r_temp_12;
wire [7:0]	r_temp_13;
wire [7:0]	r_temp_21;
wire [7:0]	r_temp_22;
wire [7:0]	r_temp_23;
wire [7:0]	r_temp_31;
wire [7:0]	r_temp_32;
wire [7:0]	r_temp_33;

wire [7:0]	g_temp_11;
wire [7:0]	g_temp_12;
wire [7:0]	g_temp_13;
wire [7:0]	g_temp_21;
wire [7:0]	g_temp_22;
wire [7:0]	g_temp_23;
wire [7:0]	g_temp_31;
wire [7:0]	g_temp_32;
wire [7:0]	g_temp_33;

wire [7:0]	b_temp_11;
wire [7:0]	b_temp_12;
wire [7:0]	b_temp_13;
wire [7:0]	b_temp_21;
wire [7:0]	b_temp_22;
wire [7:0]	b_temp_23;
wire [7:0]	b_temp_31;
wire [7:0]	b_temp_32;
wire [7:0]	b_temp_33;

assign o_hs = i_hsyn_d[3];
assign o_vs = i_vsyn_d[3];
assign o_en = i_en_d[3];
assign o_r	= avg_r_final;
assign o_g	= avg_g_final;
assign o_b	= avg_b_final;

// 同步信号延迟 - 增加到4级延迟以匹配新的流水线
always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) begin
        i_hsyn_d <= 4'b0;
        i_vsyn_d <= 4'b0;
        i_en_d   <= 4'b0;
    end
    else begin
        i_hsyn_d <= {i_hsyn_d[2:0], i_hsyn};
        i_vsyn_d <= {i_vsyn_d[2:0], i_vsyn};
        i_en_d   <= {i_en_d[2:0], i_en};
    end
end

// 红色通道模板
image_template u_r_template
(
	.i_clk			(i_clk				),
	.i_rst_n		(i_rst_n			),
	.i_en			(i_en				),
	.i_data			(i_r				),
	.o_en			(					),
	.o_temp_11		(r_temp_11			),
	.o_temp_12		(r_temp_12			),
	.o_temp_13		(r_temp_13			),	
	.o_temp_21		(r_temp_21			),
	.o_temp_22		(r_temp_22			),
	.o_temp_23		(r_temp_23			),		
	.o_temp_31		(r_temp_31			),
	.o_temp_32		(r_temp_32			),
	.o_temp_33		(r_temp_33			)
);

// 绿色通道模板
image_template u_g_template
(
	.i_clk			(i_clk				),
	.i_rst_n		(i_rst_n			),
	.i_en			(i_en				),
	.i_data			(i_g				),
	.o_en			(					),
	.o_temp_11		(g_temp_11			),
	.o_temp_12		(g_temp_12			),
	.o_temp_13		(g_temp_13			),	
	.o_temp_21		(g_temp_21			),
	.o_temp_22		(g_temp_22			),
	.o_temp_23		(g_temp_23			),		
	.o_temp_31		(g_temp_31			),
	.o_temp_32		(g_temp_32			),
	.o_temp_33		(g_temp_33			)
);

// 蓝色通道模板
image_template u_b_template
(
	.i_clk			(i_clk				),
	.i_rst_n		(i_rst_n			),
	.i_en			(i_en				),
	.i_data			(i_b				),
	.o_en			(					),
	.o_temp_11		(b_temp_11			),
	.o_temp_12		(b_temp_12			),
	.o_temp_13		(b_temp_13			),	
	.o_temp_21		(b_temp_21			),
	.o_temp_22		(b_temp_22			),
	.o_temp_23		(b_temp_23			),		
	.o_temp_31		(b_temp_31			),
	.o_temp_32		(b_temp_32			),
	.o_temp_33		(b_temp_33			)
);

// 求和计算
always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) begin
        sum_r <= 12'd0;
        sum_g <= 12'd0;
        sum_b <= 12'd0;
    end
    else if(i_en_d[1]) begin // 使用延迟后的使能信号
        sum_r <= r_temp_11 + r_temp_12 + r_temp_13 +
                 r_temp_21 + r_temp_22 + r_temp_23 +
                 r_temp_31 + r_temp_32 + r_temp_33;
        sum_g <= g_temp_11 + g_temp_12 + g_temp_13 +
                 g_temp_21 + g_temp_22 + g_temp_23 +
                 g_temp_31 + g_temp_32 + g_temp_33;
        sum_b <= b_temp_11 + b_temp_12 + b_temp_13 +
                 b_temp_21 + b_temp_22 + b_temp_23 +
                 b_temp_31 + b_temp_32 + b_temp_33;
    end
    else begin
        sum_r <= 12'd0;
        sum_g <= 12'd0;
        sum_b <= 12'd0;
    end
end

// 近似除法计算: 除以9 ≈ (×455 + 256) >> 12
// 455/4096 ≈ 0.111, 加上256用于四舍五入
always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) begin
        avg_r_temp <= 20'd0;
        avg_g_temp <= 20'd0;
        avg_b_temp <= 20'd0;
    end
    else if(i_en_d[2]) begin
        // (sum × 455 + 256) >> 12
        avg_r_temp <= (sum_r * 12'd455 + 20'd256) >> 12;
        avg_g_temp <= (sum_g * 12'd455 + 20'd256) >> 12;
        avg_b_temp <= (sum_b * 12'd455 + 20'd256) >> 12;
    end
    else begin
        avg_r_temp <= 20'd0;
        avg_g_temp <= 20'd0;
        avg_b_temp <= 20'd0;
    end
end

// 输出结果限制在0-255范围内
always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) begin
        avg_r_final <= 8'd0;
        avg_g_final <= 8'd0;
        avg_b_final <= 8'd0;
    end
    else if(i_en_d[3]) begin
        // 限制输出范围在0-255
        avg_r_final <= (avg_r_temp[7:0] > 8'd255) ? 8'd255 : avg_r_temp[7:0];
        avg_g_final <= (avg_g_temp[7:0] > 8'd255) ? 8'd255 : avg_g_temp[7:0];
        avg_b_final <= (avg_b_temp[7:0] > 8'd255) ? 8'd255 : avg_b_temp[7:0];
    end
    else begin
        avg_r_final <= 8'd0;
        avg_g_final <= 8'd0;
        avg_b_final <= 8'd0;
    end
end

endmodule