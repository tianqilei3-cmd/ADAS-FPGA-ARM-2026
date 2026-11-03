module image_edge_filtering
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
	
	output	wire 				o_hs,
	output	wire 				o_vs,
	output	wire 				o_en,	
	output  wire [7:0]			o_gray
);
wire 				gray_hsyn;
wire 				gray_vsyn;
wire [23:0] 		gray_data;
wire 				gray_de;

wire 		erode_hsyn;
wire 		erode_vsyn;
wire 		erode_de;	
wire [7:0]	erode_data;

wire 		dilate_hsyn;
wire 		dilate_vsyn;
wire 		dilate_de;	
wire [7:0]	dilate_data;

assign 		o_hs		= erode_hsyn;
assign 		o_vs		= erode_vsyn;
assign 		o_en		= erode_de;		
assign 		o_gray		= dilate_data - erode_data;

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

image_dilate_gray u_image_dilate_gray
(
	.i_clk			(i_clk				),
	.i_rst_n		(i_rst_n			),
	.i_hsyn		    (gray_hsyn				),
	.i_vsyn		    (gray_vsyn				),
	.i_en		    (gray_de				),
	.i_gray			(gray_data[7:0]				),
	.o_hs		    (dilate_hsyn		),
	.o_vs		    (dilate_vsyn		),
	.o_en		    (dilate_de			),	
	.o_gray			(dilate_data		)
);

image_erode_gray u_image_erode_gray
(
	.i_clk			(i_clk				),
	.i_rst_n		(i_rst_n			),
	.i_hsyn		    (gray_hsyn				),
	.i_vsyn		    (gray_vsyn				),
	.i_en		    (gray_de				),
	.i_gray			(gray_data[7:0]				),
	.o_hs		    (erode_hsyn			),
	.o_vs		    (erode_vsyn			),
	.o_en		    (erode_de			),	
	.o_gray			(erode_data 		)
);

endmodule