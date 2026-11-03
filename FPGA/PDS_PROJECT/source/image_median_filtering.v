module image_median_filtering
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

reg  [7:0]	r_temp_max_1;
reg  [7:0]	r_temp_med_1;
reg  [7:0]	r_temp_min_1;
reg  [7:0]	r_temp_max_2;
reg  [7:0]	r_temp_med_2;
reg  [7:0]	r_temp_min_2;
reg  [7:0]	r_temp_max_3;
reg  [7:0]	r_temp_med_3;
reg  [7:0]	r_temp_min_3;
reg  [7:0]	r_temp_max_min;
reg  [7:0]	r_temp_med_med;
reg  [7:0]	r_temp_min_max;
reg  [7:0]	r_med;

reg  [7:0]	g_temp_max_1;
reg  [7:0]	g_temp_med_1;
reg  [7:0]	g_temp_min_1;
reg  [7:0]	g_temp_max_2;
reg  [7:0]	g_temp_med_2;
reg  [7:0]	g_temp_min_2;
reg  [7:0]	g_temp_max_3;
reg  [7:0]	g_temp_med_3;
reg  [7:0]	g_temp_min_3;
reg  [7:0]	g_temp_max_min;
reg  [7:0]	g_temp_med_med;
reg  [7:0]	g_temp_min_max;
reg  [7:0]	g_med;

reg  [7:0]	b_temp_max_1;
reg  [7:0]	b_temp_med_1;
reg  [7:0]	b_temp_min_1;
reg  [7:0]	b_temp_max_2;
reg  [7:0]	b_temp_med_2;
reg  [7:0]	b_temp_min_2;
reg  [7:0]	b_temp_max_3;
reg  [7:0]	b_temp_med_3;
reg  [7:0]	b_temp_min_3;
reg  [7:0]	b_temp_max_min;
reg  [7:0]	b_temp_med_med;
reg  [7:0]	b_temp_min_max;
reg  [7:0]	b_med;

assign o_hs = i_hsyn_d[3];
assign o_vs = i_vsyn_d[3];
assign o_en = i_en_d[3]	;
assign o_r	= r_med;
assign o_g	= g_med;
assign o_b	= b_med;

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
        r_temp_max_1 <= 'd0; 
		r_temp_med_1 <= 'd0;
		r_temp_min_1 <= 'd0;
		r_temp_max_2 <= 'd0;
		r_temp_med_2 <= 'd0;
		r_temp_min_2 <= 'd0;
		r_temp_max_3 <= 'd0;
		r_temp_med_3 <= 'd0;
		r_temp_min_3 <= 'd0;
	end	
    else 
	begin
		r_temp_max_1 <= (r_temp_11 >= r_temp_12 && r_temp_11 >= r_temp_13)? r_temp_11:
						(r_temp_12 >= r_temp_11 && r_temp_12 >= r_temp_13)? r_temp_12:
						(r_temp_13 >= r_temp_11 && r_temp_13 >= r_temp_12)? r_temp_13:'d0;
		r_temp_med_1 <= (r_temp_11 >= r_temp_12 && r_temp_11 <= r_temp_13 || r_temp_11 <= r_temp_12 && r_temp_11 >= r_temp_13)? r_temp_11:
						(r_temp_12 >= r_temp_11 && r_temp_12 <= r_temp_13 || r_temp_12 <= r_temp_11 && r_temp_12 >= r_temp_13)? r_temp_12:
						(r_temp_13 >= r_temp_11 && r_temp_13 <= r_temp_12 || r_temp_13 <= r_temp_11 && r_temp_13 >= r_temp_12)? r_temp_13:'d0;
		r_temp_min_1 <= (r_temp_11 <= r_temp_12 && r_temp_11 <= r_temp_13)? r_temp_11:
						(r_temp_12 <= r_temp_11 && r_temp_12 <= r_temp_13)? r_temp_12:
						(r_temp_13 <= r_temp_11 && r_temp_13 <= r_temp_12)? r_temp_13:'d0;
		r_temp_max_2 <= (r_temp_21 >= r_temp_22 && r_temp_21 >= r_temp_23)? r_temp_21:
						(r_temp_22 >= r_temp_21 && r_temp_22 >= r_temp_23)? r_temp_22:
						(r_temp_23 >= r_temp_21 && r_temp_23 >= r_temp_22)? r_temp_23:'d0;
		r_temp_med_2 <= (r_temp_21 >= r_temp_22 && r_temp_21 <= r_temp_23 || r_temp_21 <= r_temp_22 && r_temp_21 >= r_temp_13)? r_temp_21:
						(r_temp_22 >= r_temp_21 && r_temp_22 <= r_temp_23 || r_temp_22 <= r_temp_21 && r_temp_22 >= r_temp_13)? r_temp_22:
						(r_temp_23 >= r_temp_21 && r_temp_23 <= r_temp_22 || r_temp_23 <= r_temp_21 && r_temp_23 >= r_temp_12)? r_temp_23:'d0;
		r_temp_min_2 <= (r_temp_21 <= r_temp_22 && r_temp_21 <= r_temp_23)? r_temp_21:
						(r_temp_22 <= r_temp_21 && r_temp_22 <= r_temp_23)? r_temp_22:
						(r_temp_23 <= r_temp_21 && r_temp_23 <= r_temp_22)? r_temp_23:'d0;
		r_temp_max_3 <= (r_temp_31 >= r_temp_32 && r_temp_31 >= r_temp_33)? r_temp_31:
						(r_temp_32 >= r_temp_31 && r_temp_32 >= r_temp_33)? r_temp_32:
						(r_temp_33 >= r_temp_31 && r_temp_33 >= r_temp_32)? r_temp_33:'d0;
		r_temp_med_3 <= (r_temp_31 >= r_temp_32 && r_temp_31 <= r_temp_33 || r_temp_31 <= r_temp_32 && r_temp_31 >= r_temp_33)? r_temp_31:
						(r_temp_32 >= r_temp_31 && r_temp_32 <= r_temp_33 || r_temp_32 <= r_temp_31 && r_temp_32 >= r_temp_33)? r_temp_32:
						(r_temp_33 >= r_temp_31 && r_temp_33 <= r_temp_32 || r_temp_33 <= r_temp_31 && r_temp_33 >= r_temp_32)? r_temp_33:'d0;
		r_temp_min_3 <= (r_temp_31 <= r_temp_32 && r_temp_31 <= r_temp_33)? r_temp_31:
						(r_temp_32 <= r_temp_31 && r_temp_32 <= r_temp_33)? r_temp_32:
						(r_temp_33 <= r_temp_31 && r_temp_33 <= r_temp_32)? r_temp_33:'d0;
	end
end

always@(posedge i_clk or negedge i_rst_n)
begin
    if(!i_rst_n)
	begin
        r_temp_max_min <= 'd0; 
		r_temp_med_med <= 'd0;
		r_temp_min_max <= 'd0;
	end	
    else 
	begin
        r_temp_max_min <= (r_temp_max_1 <= r_temp_max_2 && r_temp_max_1 <= r_temp_max_3)? r_temp_max_1:
		                  (r_temp_max_2 <= r_temp_max_1 && r_temp_max_2 <= r_temp_max_3)? r_temp_max_2:
		                  (r_temp_max_3 <= r_temp_max_1 && r_temp_max_3 <= r_temp_max_2)? r_temp_max_3:'d0;
		r_temp_med_med <= (r_temp_med_1 >= r_temp_med_2 && r_temp_med_1 <= r_temp_med_3 || r_temp_med_1 <= r_temp_med_2 && r_temp_med_1 >= r_temp_med_3)? r_temp_med_1:
		                  (r_temp_med_2 >= r_temp_med_1 && r_temp_med_2 <= r_temp_med_3 || r_temp_med_2 <= r_temp_med_1 && r_temp_med_2 >= r_temp_med_3)? r_temp_med_2:
		                  (r_temp_med_3 >= r_temp_med_1 && r_temp_med_3 <= r_temp_med_2 || r_temp_med_3 <= r_temp_med_1 && r_temp_med_3 >= r_temp_med_2)? r_temp_med_3:'d0;
		r_temp_min_max <= (r_temp_min_1 >= r_temp_min_2 && r_temp_min_1 >= r_temp_min_3)? r_temp_min_1:
		                  (r_temp_min_2 >= r_temp_min_1 && r_temp_min_2 >= r_temp_min_3)? r_temp_min_2:
						  (r_temp_min_3 >= r_temp_min_1 && r_temp_min_3 >= r_temp_min_2)? r_temp_min_3:'d0;
	end					  
end

always@(posedge i_clk or negedge i_rst_n)
begin
    if(!i_rst_n)
	begin
        g_temp_max_1 <= 'd0; 
		g_temp_med_1 <= 'd0;
		g_temp_min_1 <= 'd0;
		g_temp_max_2 <= 'd0;
		g_temp_med_2 <= 'd0;
		g_temp_min_2 <= 'd0;
		g_temp_max_3 <= 'd0;
		g_temp_med_3 <= 'd0;
		g_temp_min_3 <= 'd0;
	end	
    else 
	begin
		g_temp_max_1 <= (g_temp_11 >= g_temp_12 && g_temp_11 >= g_temp_13)? g_temp_11:
						(g_temp_12 >= g_temp_11 && g_temp_12 >= g_temp_13)? g_temp_12:
						(g_temp_13 >= g_temp_11 && g_temp_13 >= g_temp_12)? g_temp_13:'d0;
		g_temp_med_1 <= (g_temp_11 >= g_temp_12 && g_temp_11 <= g_temp_13 || g_temp_11 <= g_temp_12 && g_temp_11 >= g_temp_13)? g_temp_11:
						(g_temp_12 >= g_temp_11 && g_temp_12 <= g_temp_13 || g_temp_12 <= g_temp_11 && g_temp_12 >= g_temp_13)? g_temp_12:
						(g_temp_13 >= g_temp_11 && g_temp_13 <= g_temp_12 || g_temp_13 <= g_temp_11 && g_temp_13 >= g_temp_12)? g_temp_13:'d0;
		g_temp_min_1 <= (g_temp_11 <= g_temp_12 && g_temp_11 <= g_temp_13)? g_temp_11:
						(g_temp_12 <= g_temp_11 && g_temp_12 <= g_temp_13)? g_temp_12:
						(g_temp_13 <= g_temp_11 && g_temp_13 <= g_temp_12)? g_temp_13:'d0;
		g_temp_max_2 <= (g_temp_21 >= g_temp_22 && g_temp_21 >= g_temp_23)? g_temp_21:
						(g_temp_22 >= g_temp_21 && g_temp_22 >= g_temp_23)? g_temp_22:
						(g_temp_23 >= g_temp_21 && g_temp_23 >= g_temp_22)? g_temp_23:'d0;
		g_temp_med_2 <= (g_temp_21 >= g_temp_22 && g_temp_21 <= g_temp_23 || g_temp_21 <= g_temp_22 && g_temp_21 >= g_temp_23)? g_temp_21:
						(g_temp_22 >= g_temp_21 && g_temp_22 <= g_temp_23 || g_temp_22 <= g_temp_21 && g_temp_22 >= g_temp_23)? g_temp_22:
						(g_temp_23 >= g_temp_21 && g_temp_23 <= g_temp_22 || g_temp_23 <= g_temp_21 && g_temp_23 >= g_temp_22)? g_temp_23:'d0;
		g_temp_min_2 <= (g_temp_21 <= g_temp_22 && g_temp_21 <= g_temp_23)? g_temp_21:
						(g_temp_22 <= g_temp_21 && g_temp_22 <= g_temp_23)? g_temp_22:
						(g_temp_23 <= g_temp_21 && g_temp_23 <= g_temp_22)? g_temp_23:'d0;
		g_temp_max_3 <= (g_temp_31 >= g_temp_32 && g_temp_31 >= g_temp_33)? g_temp_31:
						(g_temp_32 >= g_temp_31 && g_temp_32 >= g_temp_33)? g_temp_32:
						(g_temp_33 >= g_temp_31 && g_temp_33 >= g_temp_32)? g_temp_33:'d0;
		g_temp_med_3 <= (g_temp_31 >= g_temp_32 && g_temp_31 <= g_temp_33 || g_temp_31 <= g_temp_32 && g_temp_31 >= g_temp_33)? g_temp_31:
						(g_temp_32 >= g_temp_31 && g_temp_32 <= g_temp_33 || g_temp_32 <= g_temp_31 && g_temp_32 >= g_temp_33)? g_temp_32:
						(g_temp_33 >= g_temp_31 && g_temp_33 <= g_temp_32 || g_temp_33 <= g_temp_31 && g_temp_33 >= g_temp_32)? g_temp_33:'d0;
		g_temp_min_3 <= (g_temp_31 <= g_temp_32 && g_temp_31 <= g_temp_33)? g_temp_31:
						(g_temp_32 <= g_temp_31 && g_temp_32 <= g_temp_33)? g_temp_32:
						(g_temp_33 <= g_temp_31 && g_temp_33 <= g_temp_32)? g_temp_33:'d0;
	end
end

always@(posedge i_clk or negedge i_rst_n)
begin
    if(!i_rst_n)
	begin
        g_temp_max_min <= 'd0; 
		g_temp_med_med <= 'd0;
		g_temp_min_max <= 'd0;
	end	
    else 
	begin
        g_temp_max_min <= (g_temp_max_1 <= g_temp_max_2 && g_temp_max_1 <= g_temp_max_3)? g_temp_max_1:
		                  (g_temp_max_2 <= g_temp_max_1 && g_temp_max_2 <= g_temp_max_3)? g_temp_max_2:
		                  (g_temp_max_3 <= g_temp_max_1 && g_temp_max_3 <= g_temp_max_2)? g_temp_max_3:'d0;
		g_temp_med_med <= (g_temp_med_1 >= g_temp_med_2 && g_temp_med_1 <= g_temp_med_3 || g_temp_med_1 <= g_temp_med_2 && g_temp_med_1 >= g_temp_med_3)? g_temp_med_1:
		                  (g_temp_med_2 >= g_temp_med_1 && g_temp_med_2 <= g_temp_med_3 || g_temp_med_2 <= g_temp_med_1 && g_temp_med_2 >= g_temp_med_3)? g_temp_med_2:
		                  (g_temp_med_3 >= g_temp_med_1 && g_temp_med_3 <= g_temp_med_2 || g_temp_med_3 <= g_temp_med_1 && g_temp_med_3 >= g_temp_med_2)? g_temp_med_3:'d0;
		g_temp_min_max <= (g_temp_min_1 >= g_temp_min_2 && g_temp_min_1 >= g_temp_min_3)? g_temp_min_1:
		                  (g_temp_min_2 >= g_temp_min_1 && g_temp_min_2 >= g_temp_min_3)? g_temp_min_2:
	                      (g_temp_min_3 >= g_temp_min_1 && g_temp_min_3 >= g_temp_min_2)? g_temp_min_3:'d0;
	end
end

always@(posedge i_clk or negedge i_rst_n)
begin
    if(!i_rst_n)
	begin
        b_temp_max_1 <= 'd0; 
		b_temp_med_1 <= 'd0;
		b_temp_min_1 <= 'd0;
		b_temp_max_2 <= 'd0;
		b_temp_med_2 <= 'd0;
		b_temp_min_2 <= 'd0;
		b_temp_max_3 <= 'd0;
		b_temp_med_3 <= 'd0;
		b_temp_min_3 <= 'd0;
	end	
    else 
	begin
		b_temp_max_1 <= (b_temp_11 >= b_temp_12 && b_temp_11 >= b_temp_13)? b_temp_11:
						(b_temp_12 >= b_temp_11 && b_temp_12 >= b_temp_13)? b_temp_12:
						(b_temp_13 >= b_temp_11 && b_temp_13 >= b_temp_12)? b_temp_13:'d0;
		b_temp_med_1 <= (b_temp_11 >= b_temp_12 && b_temp_11 <= b_temp_13 || b_temp_11 <= b_temp_12 && b_temp_11 >= b_temp_13)? b_temp_11:
						(b_temp_12 >= b_temp_11 && b_temp_12 <= b_temp_13 || b_temp_12 <= b_temp_11 && b_temp_12 >= b_temp_13)? b_temp_12:
						(b_temp_13 >= b_temp_11 && b_temp_13 <= b_temp_12 || b_temp_13 <= b_temp_11 && b_temp_13 >= b_temp_12)? b_temp_13:'d0;
		b_temp_min_1 <= (b_temp_11 <= b_temp_12 && b_temp_11 <= b_temp_13)? b_temp_11:
						(b_temp_12 <= b_temp_11 && b_temp_12 <= b_temp_13)? b_temp_12:
						(b_temp_13 <= b_temp_11 && b_temp_13 <= b_temp_12)? b_temp_13:'d0;
		b_temp_max_2 <= (b_temp_21 >= b_temp_22 && b_temp_21 >= b_temp_23)? b_temp_21:
						(b_temp_22 >= b_temp_21 && b_temp_22 >= b_temp_23)? b_temp_22:
						(b_temp_23 >= b_temp_21 && b_temp_23 >= b_temp_22)? b_temp_23:'d0;
		b_temp_med_2 <= (b_temp_21 >= b_temp_22 && b_temp_21 <= b_temp_23 || b_temp_21 <= b_temp_22 && b_temp_21 >= b_temp_23)? b_temp_21:
						(b_temp_22 >= b_temp_21 && b_temp_22 <= b_temp_23 || b_temp_22 <= b_temp_21 && b_temp_22 >= b_temp_23)? b_temp_22:
						(b_temp_23 >= b_temp_21 && b_temp_23 <= b_temp_22 || b_temp_23 <= b_temp_21 && b_temp_23 >= b_temp_22)? b_temp_23:'d0;
		b_temp_min_2 <= (b_temp_21 <= b_temp_22 && b_temp_21 <= b_temp_23)? b_temp_21:
						(b_temp_22 <= b_temp_21 && b_temp_22 <= b_temp_23)? b_temp_22:
						(b_temp_23 <= b_temp_21 && b_temp_23 <= b_temp_22)? b_temp_23:'d0;
		b_temp_max_3 <= (b_temp_31 >= b_temp_32 && b_temp_31 >= b_temp_33)? b_temp_31:
						(b_temp_32 >= b_temp_31 && b_temp_32 >= b_temp_33)? b_temp_32:
						(b_temp_33 >= b_temp_31 && b_temp_33 >= b_temp_32)? b_temp_33:'d0;
		b_temp_med_3 <= (b_temp_31 >= b_temp_32 && b_temp_31 <= b_temp_33 || b_temp_31 <= b_temp_32 && b_temp_31 >= b_temp_33)? b_temp_31:
						(b_temp_32 >= b_temp_31 && b_temp_32 <= b_temp_33 || b_temp_32 <= b_temp_31 && b_temp_32 >= b_temp_33)? b_temp_32:
						(b_temp_33 >= b_temp_31 && b_temp_33 <= b_temp_32 || b_temp_33 <= b_temp_31 && b_temp_33 >= b_temp_32)? b_temp_33:'d0;
		b_temp_min_3 <= (b_temp_31 <= b_temp_32 && b_temp_31 <= b_temp_33)? b_temp_31:
						(b_temp_32 <= b_temp_31 && b_temp_32 <= b_temp_33)? b_temp_32:
						(b_temp_33 <= b_temp_31 && b_temp_33 <= b_temp_32)? b_temp_33:'d0;
	end
end

always@(posedge i_clk or negedge i_rst_n)
begin
    if(!i_rst_n)
	begin
        b_temp_max_min <= 'd0; 
		b_temp_med_med <= 'd0;
		b_temp_min_max <= 'd0;
	end	
    else 
	begin
        b_temp_max_min <= (b_temp_max_1 <= b_temp_max_2 && b_temp_max_1 <= b_temp_max_3)? b_temp_max_1:
		                  (b_temp_max_2 <= b_temp_max_1 && b_temp_max_2 <= b_temp_max_3)? b_temp_max_2:
		                  (b_temp_max_3 <= b_temp_max_1 && b_temp_max_3 <= b_temp_max_2)? b_temp_max_3:'d0;
		b_temp_med_med <= (b_temp_med_1 >= b_temp_med_2 && b_temp_med_1 <= b_temp_med_3 || b_temp_med_1 <= b_temp_med_2 && b_temp_med_1 >= b_temp_med_3)? b_temp_med_1:
		                  (b_temp_med_2 >= b_temp_med_1 && b_temp_med_2 <= b_temp_med_3 || b_temp_med_2 <= b_temp_med_1 && b_temp_med_2 >= b_temp_med_3)? b_temp_med_2:
		                  (b_temp_med_3 >= b_temp_med_1 && b_temp_med_3 <= b_temp_med_2 || b_temp_med_3 <= b_temp_med_1 && b_temp_med_3 >= b_temp_med_2)? b_temp_med_3:'d0;
		b_temp_min_max <= (b_temp_min_1 >= b_temp_min_2 && b_temp_min_1 >= b_temp_min_3)? b_temp_min_1:
		                  (b_temp_min_2 >= b_temp_min_1 && b_temp_min_2 >= b_temp_min_3)? b_temp_min_2:
						  (b_temp_min_3 >= b_temp_min_1 && b_temp_min_3 >= b_temp_min_2)? b_temp_min_3:'d0;
	end
end                                                                        
				 
always@(posedge i_clk or negedge i_rst_n)
begin
    if(!i_rst_n)
	begin
        r_med <= 'd0; 
		g_med <= 'd0;
		b_med <= 'd0;
	end	
    else 
	begin
        r_med <= (r_temp_max_min >= r_temp_med_med && r_temp_max_min <= r_temp_min_max || r_temp_max_min <= r_temp_med_med && r_temp_max_min >= r_temp_min_max)? r_temp_max_min:
		         (r_temp_med_med >= r_temp_max_min && r_temp_med_med <= r_temp_min_max || r_temp_med_med <= r_temp_max_min && r_temp_med_med >= r_temp_min_max)? r_temp_med_med:
		         (r_temp_min_max >= r_temp_max_min && r_temp_min_max <= r_temp_med_med || r_temp_min_max <= r_temp_max_min && r_temp_min_max >= r_temp_med_med)? r_temp_min_max:'d0;
		g_med <= (g_temp_max_min >= g_temp_med_med && g_temp_max_min <= g_temp_min_max || g_temp_max_min <= g_temp_med_med && g_temp_max_min >= g_temp_min_max)? g_temp_max_min:
		         (g_temp_med_med >= g_temp_max_min && g_temp_med_med <= g_temp_min_max || g_temp_med_med <= g_temp_max_min && g_temp_med_med >= g_temp_min_max)? g_temp_med_med:
		         (g_temp_min_max >= g_temp_max_min && g_temp_min_max <= g_temp_med_med || g_temp_min_max <= g_temp_max_min && g_temp_min_max >= g_temp_med_med)? g_temp_min_max:'d0;
		b_med <= (b_temp_max_min >= b_temp_med_med && b_temp_max_min <= b_temp_min_max || b_temp_max_min <= b_temp_med_med && b_temp_max_min >= b_temp_min_max)? b_temp_max_min:
		         (b_temp_med_med >= b_temp_max_min && b_temp_med_med <= b_temp_min_max || b_temp_med_med <= b_temp_max_min && b_temp_med_med >= b_temp_min_max)? b_temp_med_med:
				 (b_temp_min_max >= b_temp_max_min && b_temp_min_max <= b_temp_med_med || b_temp_min_max <= b_temp_max_min && b_temp_min_max >= b_temp_med_med)? b_temp_min_max:'d0;
	end
end
					 
endmodule