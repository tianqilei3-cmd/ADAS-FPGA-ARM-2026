module image_dilate_gray
(
	input   wire		i_clk,
	input   wire		i_rst_n,

	input	wire		i_hsyn,
	input	wire		i_vsyn,
	input	wire		i_en,
	input	wire [7:0]	i_gray,
	
	output	wire 		o_hs,
	output	wire 		o_vs,
	output	wire 		o_en,	
	output  wire [7:0]	o_gray
);

reg	[1:0]	i_hsyn_d;
reg	[1:0]	i_vsyn_d;
reg	[1:0]	i_en_d;

reg  [7:0]  r_temp_max;

wire [7:0]	r_temp_11;
wire [7:0]	r_temp_12;
wire [7:0]	r_temp_13;
wire [7:0]	r_temp_21;
wire [7:0]	r_temp_22;
wire [7:0]	r_temp_23;
wire [7:0]	r_temp_31;
wire [7:0]	r_temp_32;
wire [7:0]	r_temp_33;

assign o_hs 	= i_hsyn_d[1];
assign o_vs 	= i_vsyn_d[1];
assign o_en 	= i_en_d[1];
assign o_gray	= r_temp_max;

always@(posedge i_clk ) 
begin
	i_hsyn_d <=	{i_hsyn_d[0],i_hsyn};
	i_vsyn_d <=	{i_vsyn_d[0],i_vsyn};
	i_en_d	 <=	{i_en_d[0],i_en};	
end

image_template u_r_template
(
	.i_clk			(i_clk				),
	.i_rst_n		(i_rst_n			),
	.i_en			(i_en				),
	.i_data			(i_gray				),
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


always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n) 
	begin
        r_temp_max <= 'd0;
    end
    else 
	begin
		r_temp_max   <= (r_temp_11 >= r_temp_12 && r_temp_11 >= r_temp_13)? r_temp_11:
						(r_temp_12 >= r_temp_11 && r_temp_12 >= r_temp_13)? r_temp_12:
						(r_temp_13 >= r_temp_11 && r_temp_13 >= r_temp_12)? r_temp_13:'d0;
    end
end

endmodule