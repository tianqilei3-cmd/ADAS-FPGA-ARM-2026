module sfp_encoder#(
    parameter VS_POSE_DATA1 =   32'h55a101bc,
    parameter VS_POSE_DATA2 =   32'h55a102bc,       
    parameter DATA_START1   =   32'h55a105bc,
    parameter DATA_START2   =   32'h55a106bc,
    parameter DATA_END1     =   32'h55a107bc,
    parameter DATA_END2     =   32'h55a108bc,
    parameter UNUSE_DATA    =   32'h55a109bc
)   
(
    input               clk_in,            //输入时钟
    input               rst_n,             //输入复位
    input       [10:0]   h_pixel,           //输入行像素
    input               vs_in,             //输入场信号 
    input               data_valid_in,     //输入数据有效信号
    input       [15:0]  data_in,           //输入数据
    
    input               tx_clk,            //光口tx端时钟
    input               tx_rst_n,          //光口tx端复位
    output  reg [3:0]   gt_txcharisk,      //光口tx端K码发送信号
    output  reg [31:0]  gt_txdata          //光口tx端发送数据
);
//localparam define 
localparam  tx_unuse_data  = 8'b0000_0001    ;  
localparam  tx_vs_pose1    = 8'b0000_0010    ;  
localparam  tx_vs_pose2    = 8'b0000_0100    ;      
localparam  tx_data_start1 = 8'b0000_1000    ;      
localparam  tx_data_start2 = 8'b0001_0000    ;      
localparam  tx_send_data   = 8'b0010_0000    ;      
localparam  tx_data_end1   = 8'b0100_0000    ;      
localparam  tx_data_end2   = 8'b1000_0000    ;   
    
//reg define  
reg         vs_in_d0;
reg         vs_in_d1;
reg         data_valid_in_d0;
reg [15:0]  data_in_d0;
reg         vs_pose;
reg [7:0]   state;
reg [7:0]   cnt_data;
reg         fifo_rd_en;
reg [7:0]   cnt_fifo_rst;
reg         fifo_rst;
reg         data_start;  
reg         data_start_d0;  
reg [10:0] h_cnt; 

//wire define 
wire [31:0] fifo_dout;
wire        fifo_empty;
wire        fifo_almost_empty;
wire [10:0] rd_data_count;
wire [10:0]  fifo_threshold_value;

//*****************************************************
//**                    main code
//*****************************************************

assign fifo_threshold_value = (h_pixel >> 1) - 1;

//对输入的数据进行寄存
always@(posedge clk_in or negedge rst_n)begin
    if(!rst_n)begin     
        data_valid_in_d0 <= 1'b0;
        data_in_d0 <= 16'd0;                    
    end
    else begin      
        data_valid_in_d0 <= data_valid_in;
        data_in_d0 <= data_in;              
    end
end

//对输入的信号进行跨时钟域处理
always@(posedge tx_clk or negedge tx_rst_n)begin
    if(!tx_rst_n)begin
        vs_in_d0 <= 1'b0;           
        vs_in_d1 <= 1'b0;       
    end
    else begin
        vs_in_d0 <= vs_in;      
        vs_in_d1 <= vs_in_d0;           
    end
end

always@(posedge tx_clk or negedge tx_rst_n)begin
    if(!tx_rst_n)begin
        data_start_d0 <= 1'b0;              
    end
    else begin
        data_start_d0 <= data_start;                    
    end
end

//产生场信号上升沿
always@(posedge tx_clk or negedge tx_rst_n)begin
    if(!tx_rst_n)
        vs_pose <= 1'b0;        
    else if(vs_in_d0 && ~vs_in_d1)
        vs_pose <= 1'b1;
    else
        vs_pose <= 1'b0;                    
end

//数据发送开始信号
always@(posedge tx_clk or negedge tx_rst_n)begin
    if(!tx_rst_n)
        data_start <= 1'b0;     
    else if(vs_pose || fifo_empty/*h_cnt >= ((h_pixel >> 1) -1)*/)
        data_start <= 1'b0;         
    else if(rd_data_count >= fifo_threshold_value)
        data_start <= 1'b1; 
    else
        data_start <= data_start;                       
end

//产生tx端K码发送信号和发送数据
always@(posedge tx_clk or negedge tx_rst_n)begin
    if(!tx_rst_n)begin
        gt_txcharisk <= 4'd0;
        gt_txdata <= 32'd0;             
        state <= tx_unuse_data; 
        fifo_rd_en <= 1'b0;
        cnt_data <= 8'd0;
      h_cnt <= 11'd0;
    end
    else begin
        if(vs_pose)begin
            gt_txcharisk <= 4'd0;
            gt_txdata <= 32'ha151a252;              
            state <= tx_vs_pose1;   
            cnt_data <= 0;
        end
        else if(data_start && ~data_start_d0)begin
            gt_txcharisk <= 4'd0;
            gt_txdata <= 32'ha151a252;              
            state <= tx_data_start1;
            cnt_data <= 0; 
        end
        else begin
            case(state)
                tx_unuse_data:  begin
                                    cnt_data <= cnt_data + 1;
                                    fifo_rd_en <= 1'b0;
                                    if(cnt_data == 255)begin
                                        gt_txcharisk <= 4'd1;
                                        gt_txdata <= UNUSE_DATA;                
                                        state <= tx_unuse_data;                                 
                                    end
                                    else begin
                                        gt_txcharisk <= 4'd0;
                                        gt_txdata <= 32'ha151a252;              
                                        state <= tx_unuse_data;                                                                 
                                    end                         
                                end
                tx_vs_pose1:    begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= VS_POSE_DATA1;             
                                    state <= tx_vs_pose2;
                                    fifo_rd_en <= 1'b0;                                                                 
                                end
                tx_vs_pose2:    begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= VS_POSE_DATA2;             
                                    state <= tx_unuse_data;                                                                 
                                end 
                tx_data_start1: begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= DATA_START1;               
                                    state <= tx_data_start2; 
                                       fifo_rd_en <= 1'b1;                                                                                                    
                                end
                tx_data_start2: begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= DATA_START2;               
                                    state <= tx_send_data; 
                                       fifo_rd_en <= 1'b1;                             
                                end
                tx_send_data:   begin
                                    if(!fifo_empty /*&& h_cnt <= ((h_pixel >> 1) -2)*/)begin
                                        gt_txcharisk <= 4'd0;
                                        gt_txdata <= fifo_dout;             
                                        state <= tx_send_data; 
                                        fifo_rd_en <= 1'b1;
                                        h_cnt <= h_cnt + 1'b1;
                                    end   
//                                    else if(fifo_empty && h_cnt <= ((h_pixel >> 1) -2))begin
//                                        gt_txcharisk <= 4'd0;
//                                        gt_txdata <= fifo_dout;             
//                                        state <= tx_send_data; 
//                                        fifo_rd_en <= 1'b0;
//                                        h_cnt <= h_cnt;
//                                    end                               
                                    else begin
                                        h_cnt <= 11'd0;
                                        gt_txcharisk <= 4'd0;
                                        gt_txdata <= fifo_dout;                
                                        state <= tx_data_end1; 
                                        fifo_rd_en <= 1'b0;
                                    end                                 
                                end
                tx_data_end1:   begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= DATA_END1;             
                                    state <= tx_data_end2; 
                                    fifo_rd_en <= 1'b0;                                 
                                end 
                tx_data_end2:   begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= DATA_END2;             
                                    state <= tx_unuse_data; 
                                    fifo_rd_en <= 1'b0;                                 
                                end                                             
                default : begin
                                cnt_data <= cnt_data + 1;
                                if(cnt_data == 255)begin
                                    gt_txcharisk <= 4'd1;
                                    gt_txdata <= UNUSE_DATA;                
                                    state <= tx_unuse_data;                                 
                                end
                                else begin
                                    gt_txcharisk <= 4'd0;
                                    gt_txdata <= 32'ha151a252;              
                                    state <= tx_unuse_data;                                                                 
                                end                 
                          end
            endcase
        end         
    end
end

//产生fifo复位信号
always@(posedge tx_clk or negedge tx_rst_n)begin
    if(!tx_rst_n)begin
        cnt_fifo_rst <= 8'b0;   
        fifo_rst <= 1'b1;
    end
    else if(vs_pose)begin
        cnt_fifo_rst <= 8'd0;
        fifo_rst <= 1;
    end    
    else if(cnt_fifo_rst >= 100)begin
        cnt_fifo_rst <= cnt_fifo_rst;
        fifo_rst <= 0;
    end         
    else begin
        cnt_fifo_rst <= cnt_fifo_rst + 1;
        fifo_rst <= fifo_rst;   
    end                 
end

sfp_tx_2048x16 u_sfp_tx_2048x16 (
  .wr_clk(clk_in),                // input
  .wr_rst(fifo_rst),                // input
  .wr_en(data_valid_in_d0),                  // input
  .wr_data(data_in_d0),              // input [15:0]
  .wr_full(),              // output
  .almost_full(),      // output
  .rd_clk(tx_clk),                // input
  .rd_rst(fifo_rst),                // input
  .rd_en(fifo_rd_en),                  // input
  .rd_data(fifo_dout),              // output [31:0]
  .rd_water_level(rd_data_count),    // output [10:0]
  .rd_empty(fifo_empty),            // output
  .almost_empty(fifo_almost_empty)     // output
);



endmodule