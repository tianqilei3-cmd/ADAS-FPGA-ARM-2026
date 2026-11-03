module image_relief_filtering
(
	input   wire				i_clk,
	input   wire				i_rst_n,

	input	wire				i_hsyn,
	input	wire				i_vsyn,
	input	wire [7:0]			i_r,
	input	wire [7:0] 			i_g,
	input	wire [7:0] 			i_b,	
	input   wire				i_de,	
	
	//input	wire				i_en,
	//input	wire [7:0]			i_gray,
	
	//input   wire [7:0]      	value,	
	
	output	wire 				o_hs,
	output	wire 				o_vs,
	output	wire 				o_en,	
	output  wire [7:0]			o_gray
);

reg [1:0]	hsyn_reg;
reg [1:0]	vsyn_reg;
reg [1:0]	en_reg;
reg [7:0]	gray_reg;
reg [7:0]	gray_reg_1d;
wire signed [9:0]	relief;
parameter value=8'd100;
wire 				gray_hsyn;
wire 				gray_vsyn;
wire [23:0] 		gray_data;
wire 				gray_de;


rgb2gray_algorithm u_algorithm
(
	.i_clk			(i_clk			),	
	.i_rst_n        (i_rst_n		),
	.i_hsyn			(i_hsyn			),		
	.i_vsyn			(i_vsyn			),		
	.i_r			(i_r			),
	.i_g			(i_g			),
	.i_b			(i_b			),	
	.i_de			(i_de			),		
	.o_gray_hsyn	(gray_hsyn		),	
	.o_gray_vsyn	(gray_vsyn		),	
	.o_gray_data	(gray_data		),	
	.o_gray_de		(gray_de	    )
);

assign 		o_hs		= hsyn_reg[1];
assign 		o_vs		= vsyn_reg[1];
assign 		o_en		= en_reg[1];		
assign 		o_gray		= gray_reg_1d;

assign 		relief 		= gray_data[15:8] - gray_reg + value;

always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n)
	begin
        hsyn_reg	<= 'd0;
		vsyn_reg	<= 'd0;
		en_reg		<= 'd0;
		gray_reg	<= 'd0;	
    end
    else 
	begin
        hsyn_reg	<= {hsyn_reg[0],gray_hsyn};
		vsyn_reg	<= {vsyn_reg[0],gray_vsyn};
		en_reg		<= {en_reg[0],gray_de};
		gray_reg	<= gray_data[15:8];	      	
    end
end

always@(posedge i_clk or negedge i_rst_n) 
begin
    if(!i_rst_n)
	begin
        gray_reg_1d	<= 'd0;
    end
    else if(relief > 255)
	begin
        gray_reg_1d	<= 255;   	
    end
    else if(relief < 0)
	begin
        gray_reg_1d	<= 0;   	
    end	
    else 
	begin
        gray_reg_1d	<= relief;   	
    end	
end

endmodule