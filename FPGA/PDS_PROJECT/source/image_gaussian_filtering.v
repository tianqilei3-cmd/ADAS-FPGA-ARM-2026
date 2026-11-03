module image_gaussian_filtering
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

reg [11:0]		sum_r1;
reg [11:0]		sum_g1;
reg [11:0]		sum_b1;

reg [11:0]		sum_r2;
reg [11:0]		sum_g2;
reg [11:0]		sum_b2;

reg [11:0]		sum_r3;
reg [11:0]		sum_g3;
reg [11:0]		sum_b3;

reg [7:0]		gau_r;
reg [7:0]		gau_g;
reg [7:0]		gau_b;


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
assign o_en = i_en_d[3]	;
assign o_r	= gau_r;
assign o_g	= gau_g;
assign o_b	= gau_b;

always@(posedge i_clk ) 
begin
	i_hsyn_d <=	{i_hsyn_d[2:0],i_hsyn};
	i_vsyn_d <=	{i_vsyn_d[2:0],i_vsyn};
	i_en_d	 <=	{i_en_d[2:0],i_en};
     	
end

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

always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) 
	begin
        sum_r1 <= 12'd0;
        sum_g1 <= 12'd0;
        sum_b1 <= 12'd0;
		sum_r2 <= 12'd0;
        sum_g2 <= 12'd0;
        sum_b2 <= 12'd0;
		sum_r3 <= 12'd0;
        sum_g3 <= 12'd0;
        sum_b3 <= 12'd0;
    end
    else 
	begin
        sum_r1 <= r_temp_11*1 + r_temp_12*2 + r_temp_13*1;
		sum_r2 <= r_temp_21*2 + r_temp_22*4 + r_temp_23*2;
		sum_r3 <= r_temp_31*1 + r_temp_32*2 + r_temp_33*1;
		
        sum_g1 <= g_temp_11*1 + g_temp_12*2 + g_temp_13*1;
		sum_g2 <= g_temp_21*2 + g_temp_22*4 + g_temp_23*2;
		sum_g3 <= g_temp_31*1 + g_temp_32*2 + g_temp_33*1;	
		
        sum_b1 <= b_temp_11*1 + b_temp_12*2 + b_temp_13*1;
		sum_b2 <= b_temp_21*2 + b_temp_22*4 + b_temp_23*2;
		sum_b3 <= b_temp_31*1 + b_temp_32*2 + b_temp_33*1;
    end
end

always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) 
	begin
        sum_r <= 12'd0;
        sum_g <= 12'd0;
        sum_b <= 12'd0;
    end
    else 
	begin
        sum_r <= sum_r1 + sum_r2 + sum_r3;
		sum_g <= sum_g1 + sum_g2 + sum_g3;
		sum_b <= sum_b1 + sum_b2 + sum_b3;
    end
end

always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) 
	begin
        gau_r <= 8'd0;
        gau_g <= 8'd0;
        gau_b <= 8'd0;
    end
    else 
	begin
        gau_r <= sum_r >> 4;
		gau_g <= sum_g >> 4;
		gau_b <= sum_b >> 4;
    end
end

endmodule