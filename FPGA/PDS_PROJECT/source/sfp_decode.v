
module sfp_decode#(
	parameter VS_POSE_DATA1	=	32'h55a101bc,
	parameter VS_POSE_DATA2	=	32'h55a102bc,		
	parameter DATA_START1   =	32'h55a105bc,
	parameter DATA_START2   =	32'h55a106bc,
	parameter DATA_END1     =	32'h55a107bc,
	parameter DATA_END2	    =	32'h55a108bc,
	parameter UNUSE_DATA   	=	32'h55a109bc

)	
(
	input			    clk_in,
	input               rst_n,
	output  reg         vs_out,           //场输出信号
	output  reg     	data_valid_out,   //数据输出有效信号
	output      [15:0]  data_out,         //数据输出信号
	
	input    		    rx_clk,
	input               rx_rst_n,
	input       [3:0]   rx_charisk_align, //rx端K码校正后接收信号
	input       [31:0]  rx_data_align     //rx端校正后发送数据 
	
);
//reg define 
reg 		sfp_line_end_t;
reg 		sfp_line_end_t1;

reg [7:0]	data_end_en_d;
reg 		sfp_line_end;
reg [7:0]	cnt_vs;	
reg 		data_start_en;	
reg 		data_end_en;
reg 		fifo_wren;	
reg [31:0]	fifo_datain;
reg     	fifo_rd_en;

reg [31:0]	rx_data_align_d0;
reg [3:0]	rx_charisk_align_d0;
reg [31:0]	rx_data_align_d1;
reg [3:0]	rx_charisk_align_d1;
reg [3:0]	data_start_en_d;

//wire define 	
wire [11:0]  rd_water_level;
	
//*****************************************************
//**                    main code
//*****************************************************

//对输入的数据进行打拍
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)begin		
		rx_charisk_align_d0 <= 1'b0;
		rx_data_align_d0 <= 16'd0;	
		rx_charisk_align_d1 <= 1'b0;
		rx_data_align_d1 <= 16'd0;							
	end
	else begin		
		rx_charisk_align_d0 <= rx_charisk_align;
		rx_data_align_d0 <= rx_data_align;		
		rx_charisk_align_d1 <= rx_charisk_align_d0;
		rx_data_align_d1 <= rx_data_align_d0;					
	end
end	

//对数据接收开始和结束信号进行移位
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)begin	
		data_end_en_d <= 8'd0;
		data_start_en_d <= 4'd0;
	end						
	else begin
		data_end_en_d <= {data_end_en_d[6:0],data_end_en};
		data_start_en_d <= {data_start_en_d[2:0],data_start_en};			
	end			
end	

//产生一行数据接收结束信号
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)	
		sfp_line_end <= 1'd0;					
	else if(data_end_en_d >0)
		sfp_line_end <= 1'd1;
    else
		sfp_line_end <= 1'd0;	
end	

//产生场输出信号
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)begin		
		vs_out <= 1'b0;	
		cnt_vs <= 8'h0;		
	end	
	else if((rx_charisk_align == 4'd1) && (rx_data_align_d0 == VS_POSE_DATA2) 
		&& ((rx_data_align_d1 == VS_POSE_DATA1)))begin
		vs_out <= 1'b1;	
		cnt_vs <= 8'h0;
	end
	else if(cnt_vs >= 5)begin
			vs_out <= 1'b0;	
			cnt_vs <= cnt_vs;	
	end			
	else begin
			vs_out <= vs_out;
			cnt_vs <= cnt_vs + 1;			
	end		
end		
	
//产生数据接收开始信号	
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)begin		
		data_start_en <= 1'b0;		
	end	
	else if((rx_charisk_align == 4'd1) && (rx_data_align == DATA_START2) 
		&& ((rx_data_align_d0 == DATA_START1)))begin
		data_start_en <= 1'b1;
	end		
	else begin
		data_start_en <= 1'b0;		
	end		
end	

//产生数据接收结束信号		
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)begin		
		data_end_en <= 1'b0;		
	end	
	else if((rx_charisk_align == 4'd1) && (rx_data_align == DATA_END2) 
		&& ((rx_data_align_d0 == DATA_END1)))begin
		data_end_en <= 1'b1;
	end		
	else begin
		data_end_en <= 1'b0;		
	end		
end	

//产生fifo写使能	
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)	
		fifo_wren <= 1'b0;		
	else if(data_start_en_d[1]) 
		fifo_wren <= 1'b1;	
	else if(data_end_en) 
		fifo_wren <= 1'b0;		
	else 
		fifo_wren <= fifo_wren;			
end	

//产生fifo写数据
always@(posedge rx_clk or negedge rx_rst_n)begin
	if(!rx_rst_n)	
		fifo_datain <= 32'b0;		
	else 
		fifo_datain <= rx_data_align_d1;					
end		
	
/*****************读数据*****************************/
	
//对一行数据接收结束信号进行时钟域切换
always@(posedge clk_in or negedge rst_n)begin
	if(!rst_n)begin	
		sfp_line_end_t <= 1'b0;	
		sfp_line_end_t1 <= 1'b0;			
    end		
	else begin
		sfp_line_end_t  <= sfp_line_end;
		sfp_line_end_t1 <= sfp_line_end_t;		
    end		
end	

//产生fifo读使能
always@(posedge clk_in or negedge rst_n)begin
	if(!rst_n)
		fifo_rd_en <= 1'b0;					
	else begin
		if(sfp_line_end_t1)	
			fifo_rd_en <= 1'b1;	
        else if(rd_water_level <= 1)
			fifo_rd_en <= 1'b0;	
        else
			fifo_rd_en <= fifo_rd_en;			
    end		
end	

//产生数据输出有效信号
always@(posedge clk_in or negedge rst_n)begin
	if(!rst_n)
		data_valid_out <= 1'b0;					
	else 
		data_valid_out <= fifo_rd_en;			
end				

 
sfp_rx_32x1024 u_sfp_rx_32x1024 (
    .wr_clk            (rx_clk),          
    .wr_rst            (vs_out),          
    .wr_en             (fifo_wren),       
    .wr_data           (fifo_datain),     
    .wr_full           ( ),            
    .almost_full       ( ),      
    .rd_clk            (clk_in),          
    .rd_rst            (vs_out),          
    .rd_en             (fifo_rd_en),      
    .rd_data           (data_out),        
    .rd_empty          (fifo_empty),        
    .rd_water_level    (rd_water_level),  
    .almost_empty      (fifo_almost_empty)
);

endmodule
