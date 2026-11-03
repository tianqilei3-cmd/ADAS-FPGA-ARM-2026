module sfp_data_align(
	input                clk_in,         //rx端时钟
	input       	     rst_n,          //rx端复位
	input       [31:0]   rx_data_in,     //rx端未校正发送数据
	input       [3:0]    rx_charisk_in,  //rx端K码未校正接收信号
	
	output  reg [31:0]   rx_data_out,    //rx端校正后发送数据 
	output  reg [3:0]    rx_charisk_out	 //rx端K码校正后接收信号
	
);
//reg define 	
reg  [31:0]   rx_data_in_d0;    
reg  [3:0]    rx_charisk_in_d0;	
reg  [31:0]   rx_data_in_d1;    
reg  [3:0]    rx_charisk_in_d1;	
reg  [3:0]    byte_ctrl;

//*****************************************************
//**                    main code
//*****************************************************

//对输入数据打拍	
always@(posedge clk_in or negedge rst_n)begin
	if(!rst_n)begin
		rx_data_in_d0 <= 1'b0;
		rx_charisk_in_d0 <= 1'b0;				
		rx_data_in_d1 <= 1'b0;
		rx_charisk_in_d1 <= 1'b0;				
	end
	else begin
		rx_data_in_d0 <= rx_data_in;
		rx_charisk_in_d0 <= rx_charisk_in;				
		rx_data_in_d1 <= rx_data_in_d0;
		rx_charisk_in_d1 <= rx_charisk_in_d0;				
	end
end	
		
//对rx端K码未校正接收信号进行寄存	
always@(posedge clk_in or negedge rst_n)begin
	if(!rst_n)begin
		byte_ctrl <= 4'd0;		
	end
	else begin
		if(rx_charisk_in_d0 > 0)
			byte_ctrl <= rx_charisk_in_d0;
		else
			byte_ctrl <= byte_ctrl;		
	end
end		
	
//对rx端的信号进行对齐	
always@(posedge clk_in or negedge rst_n)begin
	if(!rst_n)begin
		rx_data_out <= 32'd0;
		rx_charisk_out <= 4'd0;				
	end
	else begin
		if(byte_ctrl == 4'b0010)begin
			rx_data_out <= {rx_data_in_d0[7:0],rx_data_in_d1[31:8]};
			rx_charisk_out <= {rx_charisk_in_d0[0],rx_charisk_in_d1[3:1]};
		end
		else if(byte_ctrl == 4'b0100)begin
			rx_data_out <= {rx_data_in_d0[15:0],rx_data_in_d1[31:16]};
			rx_charisk_out <= {rx_charisk_in_d0[1:0],rx_charisk_in_d1[3:2]};
		end
		else if(byte_ctrl == 4'b1000)begin
			rx_data_out <= {rx_data_in_d0[23:0],rx_data_in_d1[31:24]};
			rx_charisk_out <= {rx_charisk_in_d0[2:0],rx_charisk_in_d1[3]};
		end	
		else begin
			rx_data_out <= rx_data_in_d0;	
			rx_charisk_out <= rx_charisk_in_d0;
		end	
	end
end		
	
endmodule