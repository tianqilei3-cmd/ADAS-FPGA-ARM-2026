module sfp_8b10b_top(
    //光口接口
    input           i_p_refckn_1,              //差分参考时钟
    input           i_p_refckp_1,
    input           i_p_l2rxn,                 //差分接收数据
    input           i_p_l2rxp,
    //input           i_p_l3rxn,                   
    //input           i_p_l3rxp,
    output          o_p_l2txn,                 //差分发送数据
    output          o_p_l2txp,
    //output          o_p_l3txn,
    //output          o_p_l3txp,
    output          tx_disable,	               //tx端发送禁止使能
	//用户接口
	input           drp_clk,                   //光口输入时钟  
	input           clk_in,                    //输入时钟
	input           rst_n,                     //输入复位
	//input	[10:0]	h_pixel,                   //输入行像素
	//input           vs_in,	                   //输入场信号
	//input           data_valid_in,	           //输入数据有效信号
	//input   [15:0]  data_in,	               //输入数据

	output          vs_out,	                   //输出场信号
	output          data_valid_out,	           //输出数据有效信号
	output  [15:0]  data_out	               //输出数据   
    );
//localparam define     
localparam VS_POSE_DATA1	=	32'h55a101bc;
localparam VS_POSE_DATA2	=	32'h55a102bc;	
localparam DATA_START1      =	32'h55a105bc;
localparam DATA_START2      =	32'h55a106bc;
localparam DATA_END1        =	32'h55a107bc;
localparam DATA_END2	    =	32'h55a108bc;
localparam UNUSE_DATA	    =	32'h55a109bc; 

//wire define    
wire	    tx0_user_clk; 
wire        tx0_rst_n; 
wire [3:0]  tx0_gt_txcharisk;
wire [31:0] tx0_gt_txdata;
wire		rx1_user_clk; 
wire		rx1_rst_n; 
wire [3:0]  rx1_gt_rxcharisk;
wire [31:0] rx1_gt_rxdata;  
wire [3:0]  rx1_charisk_align;
wire [31:0] rx1_data_align; 
wire        i_wtchdg_clr_0 ;
wire        pll_rst_d;
wire        i_pll_rst_0;
wire        i_p_cfg_rst;

//*****************************************************
//**                    main code
//*****************************************************
 
assign     tx_disable 	 = 2'b00;
assign     i_p_cfg_rst	 = ~rst_n;
assign     i_pll_rst_0	 = pll_rst_d;
assign     pll_rst_d  	 = i_p_cfg_rst;
assign     i_wtchdg_clr0 = pll_rst_d;

//光口编码模块
// sfp_encoder#(
	// .VS_POSE_DATA1	    (VS_POSE_DATA1	),
	// .VS_POSE_DATA2	    (VS_POSE_DATA2	),	
	// .DATA_START1	    (DATA_START1	),
	// .DATA_START2	    (DATA_START2	),
	// .DATA_END1   	    (DATA_END1   	),
	// .DATA_END2	        (DATA_END2	    ),
	// .UNUSE_DATA	        (UNUSE_DATA	    )
// )
// u_sfp_encoder(
	// .clk_in             (clk_in         ),  //输入时钟
	// .rst_n              (rst_n          ),  //输入复位
    // .h_pixel            (h_pixel        ),  //输入行像素
	// .vs_in              (vs_in          ),  //输入场信号 
	// .data_valid_in      (data_valid_in  ),  //输入数据有效信号
	// .data_in            (data_in        ),  //输入数据
	// .tx_clk             (tx0_user_clk   ),  //光口tx端时钟
	// .tx_rst_n           (tx0_rst_n      ),  //光口tx端复位
            
	// .gt_txcharisk       (tx0_gt_txcharisk), //光口tx端K码发送信号
	// .gt_txdata	        (tx0_gt_txdata  )   //光口tx端发送数据
// ); 

//光口解码模块    
sfp_decode#(
	.VS_POSE_DATA1	    (VS_POSE_DATA1	),
	.VS_POSE_DATA2	    (VS_POSE_DATA2	),	
	.DATA_START1	    (DATA_START1	),
	.DATA_START2	    (DATA_START2	),
	.DATA_END1   	    (DATA_END1   	),
	.DATA_END2	        (DATA_END2	    ),
	.UNUSE_DATA	        (UNUSE_DATA	    )
)
u_sfp_decode(
	.clk_in             (clk_in         ),
	.rst_n              (rst_n          ),
    .rx_clk             (rx1_user_clk   ),
	.rx_rst_n           (rx1_rst_n      ),
	.rx_charisk_align   (rx1_charisk_align),
	.rx_data_align      (rx1_data_align ),
        
	.vs_out             (vs_out         ),
	.data_valid_out     (data_valid_out ),
	.data_out           (data_out       )
);   
    
//光口字对齐模块    
sfp_data_align u_sfp_data_align(
	.clk_in             (rx1_user_clk   ),  //rx端时钟
	.rst_n			    (rx1_rst_n      ),  //rx端复位
	.rx_data_in		    (rx1_gt_rxdata  ),  //rx端K码未校正接收信号
	.rx_charisk_in	    (rx1_gt_rxcharisk), //rx端未校正发送数据
    
	.rx_data_out	    (rx1_data_align ),  //rx端K码校正后接收信号
	.rx_charisk_out	    (rx1_charisk_align) //rx端校正后发送数据
);   

hsst_test hsst_test (
  .i_free_clk(drp_clk),                        // input
  .i_pll_rst_0(i_pll_rst_0),                      // input
  .i_wtchdg_clr_0(i_wtchdg_clr_0),                // input
  .o_wtchdg_st_0(),                  // output [1:0]
  .o_pll_done_0(),                    // output
  .o_txlane_done_2(),              // output
  .o_rxlane_done_2(rx1_rst_n),              // output
  .i_p_refckn_0(i_p_refckn_1),                    // input
  .i_p_refckp_0(i_p_refckp_1),                    // input
  .o_p_clk2core_tx_2(),          // output
  .i_p_tx2_clk_fr_core(1'b0),      // input
  .o_p_clk2core_rx_2(rx1_user_clk),          // output
  .i_p_rx2_clk_fr_core(rx1_user_clk),      // input
  .o_p_pll_lock_0(),                // output
  .o_p_rx_sigdet_sta_2(),      // output
  .o_p_lx_cdr_align_2(),        // output
  .o_p_pcs_lsm_synced_2(),    // output
  .i_p_l2rxn(i_p_l2rxn),                          // input
  .i_p_l2rxp(i_p_l2rxp),                          // input
  .o_p_l2txn(o_p_l2txn),                          // output
  .o_p_l2txp(o_p_l2txp),                          // output
  .i_txd_2(32'd0),                              // input [31:0]
  .i_tdispsel_2(4'd0),                    // input [3:0]
  .i_tdispctrl_2(4'd0),                  // input [3:0]
  .i_txk_2(4'd0),                              // input [3:0]
  .o_rxstatus_2(),                    // output [2:0]
  .o_rxd_2(rx1_gt_rxdata),                              // output [31:0]
  .o_rdisper_2(),                      // output [3:0]
  .o_rdecer_2(),                        // output [3:0]
  .o_rxk_2(rx1_gt_rxcharisk)                               // output [3:0]
);
/*hsst_test hsst_test (
    .i_free_clk         (drp_clk		),
    .i_pll_rst_0        (i_pll_rst_0	),
    .i_wtchdg_clr_0     (i_wtchdg_clr_0 ),
    .o_wtchdg_st_0      (),               
    .o_pll_done_0       (				),
    .o_txlane_done_2    (tx0_rst_n		),
    .o_txlane_done_3    (),              
    .o_rxlane_done_2    (),              
    .o_rxlane_done_3    (rx1_rst_n		),
    .i_p_refckn_1       (i_p_refckn_1	),
    .i_p_refckp_1       (i_p_refckp_1	),
    .o_p_clk2core_tx_2  (tx0_user_clk	),
    .o_p_clk2core_tx_3  (rx1_user_clk	),
    .i_p_tx2_clk_fr_core(tx0_user_clk	),
    .i_p_tx3_clk_fr_core(1'b0			),
    .i_p_rx2_clk_fr_core(1'b0			),
    .i_p_rx3_clk_fr_core(rx1_user_clk	),
    .o_p_pll_lock_0     (),               
    .o_p_rx_sigdet_sta_2(),   
    .o_p_rx_sigdet_sta_3(),   
    .o_p_lx_cdr_align_2 (),   
    .o_p_lx_cdr_align_3 (),   
    .o_p_pcs_lsm_synced_2(),  
    .o_p_pcs_lsm_synced_3(),  
    .i_p_l2rxn          (i_p_l2rxn		),   
    .i_p_l2rxp          (i_p_l2rxp		),   
    .i_p_l3rxn          (i_p_l3rxn		),   
    .i_p_l3rxp          (i_p_l3rxp		),   
    .o_p_l2txn          (o_p_l2txn		),   
    .o_p_l2txp          (o_p_l2txp		),   
    .o_p_l3txn          (o_p_l3txn		),   
    .o_p_l3txp          (o_p_l3txp		),   
    .i_txd_2            (tx0_gt_txdata  ),   
    .i_tdispsel_2       (4'd0			),   
    .i_tdispctrl_2      (4'd0			),   
    .i_txk_2            (tx0_gt_txcharisk),  
    .i_txd_3            (32'd0 			),   
    .i_tdispsel_3       (4'd0			),   
    .i_tdispctrl_3      (4'd0			),   
    .i_txk_3            (4'd0			),   
    .o_rxstatus_2       (),                  
    .o_rxd_2            (),                  
    .o_rdisper_2        (),                  
    .o_rdecer_2         (),                  
    .o_rxk_2            (),                  
    .o_rxstatus_3       (),                  
    .o_rxd_3            (rx1_gt_rxdata	),   
    .o_rdisper_3        (),                  
    .o_rdecer_3         (),                  
    .o_rxk_3            (rx1_gt_rxcharisk)   
);*/

endmodule
