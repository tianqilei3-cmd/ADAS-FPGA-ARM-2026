// Created by IP Generator (Version 2022.1 build 47548)
// 2025.9.21
//XTU_huray
//=============================================================================
// ADAS extension copyright (c) 2026 ADAS Project Team
// Project : FPGA ADAS Road Perception System
// Version : 3.0 ARM/FPGA fusion
// Function: 在原盘古50K DDR/HDMI工程上增加UART目标解析、决策、LED、
//           车道ROI、车道类型检测、告警叠加和可关闭夜间增强。
//=============================================================================


`timescale 1ps/1ps

`define DDR3
module test_ddr #(
  parameter VIDEO_LENGTH         = 1920                 ,
  parameter VIDEO_HIGTH          = 1080                 ,
  parameter ZOOM_VIDEO_LENGTH    = 960                 ,
  parameter ZOOM_VIDEO_HIGTH     = 540                 ,
  parameter PIXEL_WIDTH          = 32                 ,    
  parameter MEM_ROW_ADDR_WIDTH   = 15                 ,
  parameter MEM_COL_ADDR_WIDTH   = 10                 ,
  parameter MEM_BADDR_WIDTH      = 3                  ,
  parameter MEM_DQ_WIDTH         = 32                 ,
  parameter MEM_DM_WIDTH         = MEM_DQ_WIDTH/8     ,
  parameter MEM_DQS_WIDTH        = MEM_DQ_WIDTH/8     ,
  parameter M_AXI_BRUST_LEN      = 8                  ,
  parameter RW_ADDR_MIN          = 20'b0              ,
  parameter RW_ADDR_MAX          = ZOOM_VIDEO_LENGTH*ZOOM_VIDEO_HIGTH*PIXEL_WIDTH/MEM_DQ_WIDTH       ,//@540p  518400����ַ   
  parameter CTRL_ADDR_WIDTH      = MEM_ROW_ADDR_WIDTH + MEM_BADDR_WIDTH + MEM_COL_ADDR_WIDTH,
  parameter ENABLE_ADAS_IMAGE_PIPELINE = 1'b0,
  parameter ENABLE_NIGHT_ENHANCE       = 1'b0
)(
  input                                  ref_clk         ,
  input                                  rst_board       ,
  output                                 ddr_pll_lock    ,         
  output                                 ddr_init_done   ,
  //DDR 
  output                                 mem_rst_n       ,                       
  output                                 mem_ck          ,
  output                                 mem_ck_n        ,
  output                                 mem_cke         ,
  output                                 mem_cs_n        ,
  output                                 mem_ras_n       ,
  output                                 mem_cas_n       ,
  output                                 mem_we_n        ,  
  output                                 mem_odt         ,
  output     [MEM_ROW_ADDR_WIDTH-1:0]    mem_a           ,   
  output     [MEM_BADDR_WIDTH-1:0]       mem_ba          ,   
  inout      [MEM_DQS_WIDTH-1:0]         mem_dqs         ,
  inout      [MEM_DQS_WIDTH-1:0]         mem_dqs_n       ,
  inout      [MEM_DQ_WIDTH-1:0]          mem_dq          ,
  output     [MEM_DM_WIDTH-1:0]          mem_dm          ,
//MS72XX����
  output wire                               hdmi_rst   ,
  output                                   iic_tx_scl        ,
  inout                                    iic_tx_sda        ,
  output                                   iic_scl            ,
  inout                                    iic_sda            ,
  output wire                              hdmi_int_led      ,
  output wire                              fram0_done         ,
  output wire                              fram1_done         ,
  output wire                              fram2_done         ,
  output wire                              fram3_done         ,
//HDMI IN
  input wire                               pix_clk_in    ,//HDMI����ʱ�� 1080p @148.5Mhz
  input wire                               vs_in         ,//֡ͬ��
  input wire                               hs_in         ,//��ͬ��
  input wire                               de_in         ,//������Ч�ź�
  input wire [7 : 0]                       r_in          ,
  input wire [7 : 0]                       g_in          ,
  input wire [7 : 0]                       b_in          ,
//HDMI OUT
  output wire                                pix_clk_out     ,
  output reg                             r_vs_out        ,  
  output reg                             r_hs_out        , 
  output reg                             r_de_out        , 
  output reg  [7 : 0]                    r_r_out         , 
  output reg  [7 : 0]                    r_g_out         ,
  output reg  [7 : 0]                    r_b_out         ,  
//coms1	
  inout                                cmos1_scl            ,//cmos1 i2c 
  inout                                cmos1_sda            ,//cmos1 i2c 
  input                                cmos1_vsync          ,//cmos1 vsync
  input                                cmos1_href           ,//cmos1 hsync refrence,data valid
  input                                cmos1_pclk           ,//cmos1 pxiel clock
  input   [7:0]                        cmos1_data           ,//cmos1 data
  output                               cmos1_reset          , //cmos1 reset
  //coms2
  inout                                cmos2_scl            ,//cmos2 i2c 
  inout                                cmos2_sda            ,//cmos2 i2c 
  input                                cmos2_vsync          ,//cmos2 vsync
  input                                cmos2_href           ,//cmos2 hsync refrence,data valid
  input                                cmos2_pclk           ,//cmos2 pxiel clock
  input   [7:0]                        cmos2_data           ,//cmos2 data
  output                               cmos2_reset          ,
     //��ڽӿ�
  input                 i_p_refckn_1,              //��ֲο�ʱ��
  input                 i_p_refckp_1,
  input                 i_p_l2rxn,                 //��ֽ�������
  input                 i_p_l2rxp,
  //input                 i_p_l3rxn,                   
  //input                 i_p_l3rxp,
  output                o_p_l2txn,                 //��ַ�������
  output                o_p_l2txp,
  //output                o_p_l3txn,
  //output                o_p_l3txp,
  output           tx_disable,	 
  //tx�˷��ͽ�ֹʹ��
  input                 uart_rxd, 
  output                uart_txd ,
  // 原工程已占用B2/A2/B3/C5/A5/F7/F8，仅A3确认空闲。
  // A3作为ADAS综合指示：停车或报警时点亮；详细状态由HDMI叠加显示。
  output wire           adas_led_scene,
  input                 key_dehaze,
  input                 key_scene_n,
  input                 key_left_n,
  input                 key_straight_n,
  input                 key_right_n,
  input                 key_drive_n,
  input                 key_force_n

);

/******************************PARAMETER********************************************/
parameter DQ_WIDTH = MEM_DQ_WIDTH;
parameter DE_IN_WAIT  = 4'd0;
parameter DE_IN_CNT  = 4'd1;
parameter DE_IN_END  = 4'd2;

parameter DE_OUT_WAIT  = 4'd0;
parameter DE_OUT_CNT  = 4'd1;
parameter DE_OUT_END  = 4'd2;

parameter PIX_IN_WAIT = 3'd0;
parameter PIX_IN_CNT = 3'd1;
parameter PIX_IN_END = 3'd2;

parameter PIX_OUT_WAIT = 3'd0;
parameter PIX_OUT_CNT = 3'd1;
parameter PIX_OUT_END = 3'd2;
//������MAC��ַ 00-11-22-33-44-55
parameter  BOARD_MAC = 48'h00_11_22_33_44_55;     
//������IP��ַ 192.168.1.10     
parameter  BOARD_IP  = {8'd192,8'd168,8'd1,8'd10};
//Ŀ��MAC��ַ ff_ff_ff_ff_ff_ff
//parameter  DES_MAC   = 48'h58_11_22_91_38_31;
parameter  DES_MAC   = 48'h00_00_a4_0f_ff_9f;
//Ŀ��IP��ַ 192.168.1.102
parameter  DES_IP    = {8'd192,8'd168,8'd1,8'd102};
//parameter  DES_IP    = {8'd192,8'd168,8'd1,8'd10};
/******************************wire********************************************/      
wire                                   ddr_ip_clk     ;
wire                                   ddr_ip_rst_n   ;   
//д��ַͨ����                                                  
wire [3 : 0]                           M_AXI_AWID     ;
wire [CTRL_ADDR_WIDTH-1 : 0]           M_AXI_AWADDR   ;
wire                                   M_AXI_AWUSER   ;
wire                                   M_AXI_AWVALID  ;
wire                                   M_AXI_AWREADY  ;
//д����ͨ����                                                 
wire [DQ_WIDTH*8-1 : 0]                M_AXI_WDATA    ;
wire [DQ_WIDTH-1 : 0]                  M_AXI_WSTRB    ;
wire                                   M_AXI_WLAST    ;
wire [3 : 0]                           M_AXI_WUSER    ;
wire                                   M_AXI_WREADY   ;                                                
//����ַͨ����                                                 
wire [3 : 0]                           M_AXI_ARID     ;
wire                                   M_AXI_ARUSER   ;
wire [CTRL_ADDR_WIDTH-1 : 0]           M_AXI_ARADDR   ;
wire                                   M_AXI_ARVALID   ;
wire                                   M_AXI_ARREADY   ;
//������ͨ����                                                
wire  [3 : 0]                          M_AXI_RID      ;
wire  [DQ_WIDTH*8-1 : 0]               M_AXI_RDATA    ;
wire                                   M_AXI_RLAST    ;
wire                                   M_AXI_RVALID   ;
//debug������
wire  [1 : 0 ]                         init_read_clk_ctrl;
wire  [3 : 0 ]                         init_slip_step;
wire                                   force_read_clk_ctrl; 
//w_fifo
wire  [31 : 0]                        rgb_in;
wire  [31 : 0]                        video0_data_out;
wire  [31 : 0]                        video1_data_out;
wire  [31 : 0]                        video2_data_out;
wire  [31 : 0]                        video3_data_out;

//iic
wire                                 iic_clk;//10mhz
wire                                 pll_init_done;
wire                                 [11 : 0] x_act ;
wire                                 [11 : 0] y_act ;
//HDMi
wire                                vs_out;
wire                                hs_out;
wire                                de_out;
wire                                zoom_de_out;
wire [PIXEL_WIDTH - 1: 0]           zoom_data_out;
//cmos
wire                                clk_25M;
wire [1:0]                          cmos_init_done;
wire [15:0]                         cmos1_d_16bit;
wire                                cmos1_href_16bit;
wire                                cmos1_pclk_16bit;
wire[15:0]                          cmos2_d_16bit   ;
wire                                cmos2_href_16bit;
wire                                cmos2_pclk_16bit;
/******************************reg********************************************/
reg [11 : 0]              de_in_cnt    ;
reg                       de_in_d0     ;
reg                       de_in_d1     ;
reg                       vs_in_d0     ;
reg                       vs_in_d1     ;
reg [3 : 0]               de_in_state  ;
reg                       zoom_vs_in_d0   ;
reg                       zoom_vs_in_d1   ;
reg                       zoom_de_in_d0   ;
reg                       zoom_de_in_d1   ;
reg [11 : 0]              zoom_de_in_cnt  ;
reg [3 : 0]               zoom_de_in_state;
reg [11 : 0]              de_out_cnt  ;
reg                       r_de_out_d0 ;//����fifo��ҪԤ���������Զ��һ��
reg                       r_vs_out_d0 ;
reg                       de_out_d0   ;
reg                       de_out_d1   ;
reg                       vs_out_d0   ;
reg                       vs_out_d1   ;
reg [3 : 0]               de_out_state;
reg [11 : 0]              r_x_act_d0  ;
reg [11 : 0]              r_x_act     ;
reg                       video0_rd_en;
reg                       video1_rd_en;
reg                       video2_rd_en;
reg                       video3_rd_en;
reg                       video_pre_rd_flag;
wire                      w_video_pre_rd_flag;
assign w_video_pre_rd_flag = video_pre_rd_flag;
reg                         v_sync_flag;
reg [15:0]                  rstn_1ms            ;
reg [7:0]                   cmos1_d_d0          ;
reg                         cmos1_href_d0       ;
reg                         cmos1_vsync_d0      ;
reg [7:0]                   cmos2_d_d0          ;
reg                         cmos2_href_d0       ;
reg                         cmos2_vsync_d0      ;
reg [2:0]                   out_state           ;
reg [2:0]                   key_dehaze_sync     ;
reg                         key_dehaze_edge     ;
reg                         mode_toggle         ;
/******************************assign********************************************/
assign des_mac       = DES_MAC;
assign des_ip        = DES_IP;
assign eth_rst_n_0 = ddr_ip_rst_n;

always @(posedge ref_clk) begin
    if(!ddr_ip_rst_n) begin
        key_dehaze_sync <= 3'b0;
        key_dehaze_edge <= 1'b0;
        mode_toggle     <= 1'b0;
    end
    else begin
        key_dehaze_sync <= {key_dehaze_sync[1:0], key_dehaze};
        key_dehaze_edge <= ~key_dehaze_sync[1] & key_dehaze_sync[2];
        if(key_dehaze_edge)
            mode_toggle <= ~mode_toggle;
    end
end

/******************************always********************************************/
/******************************instant********************************************/
reg                         r_vs_out_1;
reg                         r_hs_out_1;
reg                         r_de_out_1;
reg  [7:0]                  r_r_out_1;
reg  [7:0]                  r_g_out_1;
reg  [7:0]                  r_b_out_1;

reg                         r_vs_out_1_1;
reg                         r_hs_out_1_1;
reg                         r_de_out_1_1;
reg  [7:0]                  r_r_out_1_1;
reg  [7:0]                  r_g_out_1_1;
reg  [7:0]                  r_b_out_1_1;

//��ֵ�˲�
wire[23:0] o_gray_data_1;
wire o_gray_de_1;
wire o_gray_hsyn_1;
wire o_gray_vsyn_1;
//��ֵ�˲�
wire[23:0] o_gray_data_2;
wire o_gray_de_2;
wire o_gray_hsyn_2;
wire o_gray_vsyn_2;
//��˹�˲�
wire[23:0] o_gray_data_3;
wire o_gray_de_3;
wire o_gray_hsyn_3;
wire o_gray_vsyn_3;
//������˹��
wire[23:0] o_gray_data_4;
wire o_gray_de_4;
wire o_gray_hsyn_4;
wire o_gray_vsyn_4;
//������˹��Ե��ȡ����
wire[23:0] o_gray_data_5;
wire o_gray_de_5;
wire o_gray_hsyn_5;
wire o_gray_vsyn_5;
//ͼ��Ҷ���̬ѧ�ݶȱ�Ե��ȡ
wire[7:0] o_gray_data_6;
wire o_gray_de_6;
wire o_gray_hsyn_6;
wire o_gray_vsyn_6;
//��ֵ��������ȡ����
wire[7:0] o_gray_data_7;
wire o_gray_de_7;
wire o_gray_hsyn_7;
wire o_gray_vsyn_7;
//ͼ�񸡵�Ч��
wire[23:0] o_gray_data_8;
wire o_gray_de_8;
wire o_gray_hsyn_8;
wire o_gray_vsyn_8;

//ͼ����ѡ��
wire  [2:0]key_choose;
// ADAS UART/decision signals in the 50MHz reference-clock domain.
wire  [2:0] adas_target_type;
wire        adas_target_valid;
wire        adas_status_valid;
wire [1:0]  adas_status_camera_id;
wire [15:0] adas_status_frame_id;
wire [7:0]  adas_status_valid_flags;
wire [7:0]  adas_status_object_flags;
wire [2:0]  adas_status_light_state;
wire [2:0]  adas_status_navigation;
wire [7:0]  adas_status_control_flags;
wire [2:0]  adas_status_arm_action;
wire        adas_bbox_valid;
wire [1:0]  adas_bbox_camera_id;
wire [15:0] adas_bbox_frame_id;
wire [1:0]  adas_bbox_slot;
wire        adas_bbox_model_id;
wire [7:0]  adas_bbox_class_id, adas_bbox_confidence;
wire [15:0] adas_bbox_x1, adas_bbox_y1, adas_bbox_x2, adas_bbox_y2;
wire        adas_uart_frame_error;
wire  [7:0] adas_uart_error_code;
wire        adas_stop_enable;
wire        adas_alarm_enable;
wire  [2:0] adas_action_code;
wire        adas_stop_effective;
wire  [2:0] adas_action_effective;
wire  [2:0] adas_alarm_code;
wire  [1:0] adas_decision_code;
wire        adas_link_timeout;
wire  [1:0] lane_type_result;
reg   [1:0] lane_type_ref_sync0, lane_type_ref_sync1;
wire  [3:0] boxes_valid_ref, boxes_model_id_ref;
wire [31:0] boxes_class_id_ref, boxes_confidence_ref;
wire [63:0] boxes_x1_ref, boxes_y1_ref, boxes_x2_ref, boxes_y2_ref;
wire        boxes_update_toggle_ref;

//ͨ�����ڽ���ѡ����Ϣ�����л�ͼ����，并接收ADAS目标位图。
wire basic_extended,basic_decision_valid,basic_seen,basic_accepted_pulse;
wire [2:0] basic_scene,basic_reject,basic_accepted_scene;
wire [7:0] basic_risk,basic_epoch,basic_age,basic_accepted_epoch;
wire [1:0] basic_movement,basic_boundary,basic_accepted_camera;
wire [15:0] basic_accepted_frame;
wire [7:0] physical_key_state,physical_key_sequence;
wire [5:0] physical_key_press;
wire [1:0] physical_scene,physical_direction;
wire physical_drive,physical_force,physical_key_active;
wire remote_driver_valid,gated_status_valid,key_effective_valid;
wire [7:0] remote_driver_state;

adas_key_control u_adas_key_control(
    .clk(ref_clk),.rst_n(rst_board),
    .key_n({key_force_n,key_drive_n,key_right_n,key_straight_n,key_left_n,key_scene_n}),
    .remote_valid(remote_driver_valid),.remote_state(remote_driver_state),
    .scene_mode(physical_scene),.direction_request(physical_direction),
    .drive_request(physical_drive),.force_request(physical_force),
    .control_active(physical_key_active),.key_sequence(physical_key_sequence),
    .key_press_pulse(physical_key_press),.key_state_byte(physical_key_state)
);

// A physical STOP overrides an old ARM GO without waiting for a new inference.
// Other key changes still need an ARM packet that is associated with the new
// control sequence; do not treat this local override as full context matching.
adas_key_epoch_gate u_key_epoch_gate(
 .drive_request(physical_drive),.status_valid(adas_status_valid),
 .controller_stop(adas_stop_enable),.controller_valid(basic_decision_valid),
 .key_sequence(physical_key_sequence),.status_epoch(basic_epoch),.accepted_epoch(basic_accepted_epoch),
 .controller_action(adas_action_code),.filtered_status_valid(gated_status_valid),
 .effective_stop(adas_stop_effective),.effective_valid(key_effective_valid),.effective_action(adas_action_effective)
);

top_uart_packet u_top_uart_packet
(
    .sys_clk   (ref_clk),
    .sys_rst_n (rst_board),
    .uart_rxd  (uart_rxd),
    .key       (physical_key_state),.key_sequence(physical_key_sequence),
    .driver_test_valid(remote_driver_valid),.driver_test_state(remote_driver_state),
    .feedback_camera(basic_accepted_camera),.feedback_frame(basic_accepted_frame),
    .feedback_epoch(basic_accepted_epoch),.feedback_scene(basic_accepted_scene),
    .feedback_action(adas_action_effective),.feedback_alarm(adas_alarm_code),
    .feedback_reject(basic_reject),.feedback_valid(key_effective_valid),
    .feedback_timeout(adas_link_timeout),.feedback_seen(basic_seen),
    .status_extended(basic_extended),.status_scene(basic_scene),.status_risk_flags(basic_risk),
    .status_epoch(basic_epoch),.status_age_units(basic_age),
    .status_movement(basic_movement),.status_boundary(basic_boundary),
    .led       (key_choose),
    .uart_txd  (uart_txd),
    .target_type       (adas_target_type),
    .target_valid      (adas_target_valid),
    .status_valid      (adas_status_valid),
    .status_camera_id  (adas_status_camera_id),
    .status_frame_id   (adas_status_frame_id),
    .status_valid_flags(adas_status_valid_flags),
    .status_object_flags(adas_status_object_flags),
    .status_light_state(adas_status_light_state),
    .status_navigation (adas_status_navigation),
    .status_control_flags(adas_status_control_flags),
    .status_arm_action (adas_status_arm_action),
    .bbox_valid        (adas_bbox_valid),
    .bbox_camera_id    (adas_bbox_camera_id),
    .bbox_frame_id     (adas_bbox_frame_id),
    .bbox_slot         (adas_bbox_slot),
    .bbox_model_id     (adas_bbox_model_id),
    .bbox_class_id     (adas_bbox_class_id),
    .bbox_confidence   (adas_bbox_confidence),
    .bbox_x1           (adas_bbox_x1),
    .bbox_y1           (adas_bbox_y1),
    .bbox_x2           (adas_bbox_x2),
    .bbox_y2           (adas_bbox_y2),
    .uart_frame_error  (adas_uart_frame_error),
    .uart_error_code   (adas_uart_error_code)
);

adas_bbox_store u_adas_bbox_store(
    .clk(ref_clk),.rst_n(rst_board),
    .status_valid(basic_seen ? basic_accepted_pulse : adas_status_valid),.status_camera_id(adas_status_camera_id),
    .status_frame_id(adas_status_frame_id),
    .bbox_valid(adas_bbox_valid && (!basic_seen || (!adas_link_timeout && basic_reject==0))),
    .bbox_camera_id(adas_bbox_camera_id),.bbox_frame_id(adas_bbox_frame_id),
    .bbox_slot(adas_bbox_slot),.bbox_model_id(adas_bbox_model_id),
    .bbox_class_id(adas_bbox_class_id),.bbox_confidence(adas_bbox_confidence),
    .bbox_x1(adas_bbox_x1),.bbox_y1(adas_bbox_y1),.bbox_x2(adas_bbox_x2),.bbox_y2(adas_bbox_y2),
    .boxes_valid(boxes_valid_ref),.boxes_model_id(boxes_model_id_ref),
    .boxes_class_id(boxes_class_id_ref),.boxes_confidence(boxes_confidence_ref),
    .boxes_x1(boxes_x1_ref),.boxes_y1(boxes_y1_ref),.boxes_x2(boxes_x2_ref),.boxes_y2(boxes_y2_ref),
    .update_toggle(boxes_update_toggle_ref));

always @(posedge ref_clk) begin
    if (!rst_board) begin lane_type_ref_sync0<=2'd0; lane_type_ref_sync1<=2'd0; end
    else begin lane_type_ref_sync0<=lane_type_result; lane_type_ref_sync1<=lane_type_ref_sync0; end
end

// 目标位图只在target_valid有效时更新并保持最终决策。
adas_basic70_control u_basic70_control(
    .clk(ref_clk),.rst_n(rst_board),.status_valid(gated_status_valid),.status_extended(basic_extended),
    .camera_id(adas_status_camera_id),.frame_id(adas_status_frame_id),
    .valid_flags(adas_status_valid_flags),.object_flags(adas_status_object_flags),.risk_flags(basic_risk),
    .control_flags(adas_status_control_flags),.light_state(adas_status_light_state),
    .navigation(adas_status_navigation),.arm_action(adas_status_arm_action),.scene(basic_scene),
    .movement(basic_movement),.boundary(basic_boundary),.epoch(basic_epoch),.age_units(basic_age),
    .stop_enable(adas_stop_enable),.alarm_enable(adas_alarm_enable),.decision_valid(basic_decision_valid),
    .link_timeout(adas_link_timeout),.seen_extended(basic_seen),.action_code(adas_action_code),
    .alarm_code(adas_alarm_code),.decision_code(adas_decision_code),.accepted_pulse(basic_accepted_pulse),
    .accepted_camera(basic_accepted_camera),.accepted_frame(basic_accepted_frame),
    .accepted_epoch(basic_accepted_epoch),.accepted_scene(basic_accepted_scene),.reject_reason(basic_reject)
);

// 使用唯一确认空闲的A3板载LED进行第一阶段硬件验证。
assign adas_led_scene = adas_stop_effective | adas_alarm_enable;


 image_average_filtering u_image_average_filtering
(
	.i_clk			(pix_clk_out				),
	.i_rst_n		(ddr_ip_rst_n && ddr_init_done),
	.i_hsyn		    (r_hs_out_1				),
	.i_vsyn		    (r_vs_out_1				),
	.i_en		    (r_de_out_1				),
	.i_r		    (r_r_out_1),
	.i_g		    (r_g_out_1),
	.i_b		    (r_b_out_1),		
	.o_hs		    (o_gray_hsyn_1		),
	.o_vs		    (o_gray_vsyn_1		),
	.o_en		    (o_gray_de_1			),	
	.o_r		    (o_gray_data_1[23:16]	),
	.o_g		    (o_gray_data_1[15:8]	),
	.o_b		    (o_gray_data_1[7:0]	)	
);



 image_gaussian_filtering u_image_gaussian_filtering
(
	.i_clk			(pix_clk_out				),
	.i_rst_n		(ddr_ip_rst_n && ddr_init_done			),
	.i_hsyn		    (r_hs_out_1				),
	.i_vsyn		    (r_vs_out_1				),
	.i_en		    (r_de_out_1				),
	.i_r		    (r_r_out_1),
	.i_g		    (r_g_out_1),
	.i_b		    (r_b_out_1),		
	.o_hs		    (o_gray_hsyn_3		),
	.o_vs		    (o_gray_vsyn_3		),
	.o_en		    (o_gray_de_3			),	
	.o_r		    (o_gray_data_3[23:16]	),
	.o_g		    (o_gray_data_3[15:8]	),
	.o_b		    (o_gray_data_3[7:0]	)	
);

image_laplacian_filtering u_image_laplacian_filtering
(
	.i_clk			(pix_clk_out				),
	.i_rst_n		(ddr_ip_rst_n && ddr_init_done			),
	.i_hsyn		    (r_hs_out_1				),
	.i_vsyn		    (r_vs_out_1				),
	.i_en		    (r_de_out_1				),
	.i_r		    (r_r_out_1				),
	.i_g		    (r_g_out_1				),
	.i_b		    (r_b_out_1				),		
	.o_hs		    (o_gray_hsyn_4		),
	.o_vs		    (o_gray_vsyn_4		),
	.o_en		    (o_gray_de_4			),	
	.o_r		    (o_gray_data_4[23:16]	),
	.o_g		    (o_gray_data_4[15:8]	),
	.o_b		    (o_gray_data_4[7:0]	)	
);

image_lapl_edge_filtering u_image_lapl_edge_filtering
(
	.i_clk			(pix_clk_out				),
	.i_rst_n		(ddr_ip_rst_n && ddr_init_done			),
	.i_hsyn		    (r_hs_out_1				),
	.i_vsyn		    (r_vs_out_1				),
	.i_en		    (r_de_out_1				),
	.i_r		    (r_r_out_1				),
	.i_g		    (r_g_out_1				),
	.i_b		    (r_b_out_1				),		
	.o_hs		    (o_gray_hsyn_5		),
	.o_vs		    (o_gray_vsyn_5		),
	.o_en		    (o_gray_de_5			),	
	.o_r		    (o_gray_data_5[23:16]	),
	.o_g		    (o_gray_data_5[15:8]	),
	.o_b		    (o_gray_data_5[7:0]	)	
);

image_edge_filtering u_image_edge_filtering
(
	.i_clk(pix_clk_out				),
	.i_rst_n(ddr_ip_rst_n && ddr_init_done			),

	.i_hsyn(r_hs_out_1				),
	.i_vsyn(r_vs_out_1				),
	.i_r(r_r_out_1				),
	.i_g(r_g_out_1				),
	.i_b(r_b_out_1				),	
	.i_de(r_de_out_1				),	
	.o_hs(o_gray_hsyn_6		),
	.o_vs(o_gray_vsyn_6		),
	.o_en(o_gray_de_6		),	
	.o_gray(o_gray_data_6	)
);

image_outline_filtering u_image_outline_filtering
(
	.i_clk(pix_clk_out),
	.i_rst_n(ddr_ip_rst_n && ddr_init_done),

	.i_hsyn(r_hs_out_1),
	.i_vsyn(r_vs_out_1),
	.i_r(r_r_out_1),
	.i_g(r_g_out_1),
	.i_b(r_b_out_1),	
	.i_de(r_de_out_1),		
	.o_hs(o_gray_hsyn_7),
	.o_vs(o_gray_vsyn_7),
	.o_en(o_gray_de_7),	
	.o_binary(o_gray_data_7)
);

image_relief_filtering u_image_relief_filtering
(
	.i_clk(pix_clk_out),
	.i_rst_n(ddr_ip_rst_n && ddr_init_done),

	.i_hsyn(r_hs_out_1),
	.i_vsyn(r_vs_out_1),
	.i_r(r_r_out_1),
	.i_g(r_g_out_1),
	.i_b(r_b_out_1),	
	.i_de(r_de_out_1),	
	.o_hs(o_gray_hsyn_8),
	.o_vs(o_gray_vsyn_8),
	.o_en(o_gray_de_8),	
	.o_gray(o_gray_data_8)
);




wire[23:0]      ispTempData; //ȥ��ǰ
wire            post_frame_vsync;
wire            post_frame_href;
wire            haze_rgb_data_en;
wire[23:0]      haze_rgb_data;
//assign ispTempData = {r_r_out_1,r_g_out_1,r_b_out_1};
assign ispTempData = {r_in,g_in,b_in};
/*Hazze_Removal_top #(
    .IMG_HDISP 	( 11'd1920  ),
    .IMG_VDISP 	( 11'd1080  )
)u_Hazze_Removal_top(
    .clk              	( pix_clk_in           ),
    .rst_n            	( ddr_init_done             ),
    .per_frame_vsync  	(  vs_in            ),//r_vs_out_1
    .per_frame_href   	( hs_in             ),//r_hs_out_1
    .per_frame_clken  	( de_in   ),//r_de_out_1
    .per_img_data     	( {r_in,g_in,b_in}      ),
    .post_frame_vsync 	( post_frame_vsync               ),
    .post_frame_href  	( post_frame_href                ),
    .post_frame_clken 	( haze_rgb_data_en  ),
    .post_img_data    	( haze_rgb_data     )
);*/

 /*image_median_filtering u_image_median_filtering
(
	.i_clk			(pix_clk_in				),
	.i_rst_n		(ddr_init_done			),
	.i_hsyn		    (post_frame_href				),
	.i_vsyn		    (post_frame_vsync				),
	.i_en		    (haze_rgb_data_en				),
	.i_r		    (haze_rgb_data[23:16]),
	.i_g		    (haze_rgb_data[15:8]),
	.i_b		    (haze_rgb_data[7:0]),		
	.o_hs		    (o_gray_hsyn_2		),
	.o_vs		    (o_gray_vsyn_2		),
	.o_en		    (o_gray_de_2			),	
	.o_r		    (o_gray_data_2[23:16]	),
	.o_g		    (o_gray_data_2[15:8]	),
	.o_b		    (o_gray_data_2[7:0]	)	
);*/

image_median_filtering u_image_median_filtering
(
	.i_clk			(pix_clk_in				),
	.i_rst_n		(ddr_init_done			),
	.i_hsyn		    (hs_in				),
	.i_vsyn		    (vs_in				),
	.i_en		    (de_in				),
	.i_r		    (ispTempData[23:16]),
	.i_g		    (ispTempData[15:8]),
	.i_b		    (ispTempData[7:0]),		
	.o_hs		    (o_gray_hsyn_2		),
	.o_vs		    (o_gray_vsyn_2		),
	.o_en		    (o_gray_de_2			),	
	.o_r		    (o_gray_data_2[23:16]	),
	.o_g		    (o_gray_data_2[15:8]	),
	.o_b		    (o_gray_data_2[7:0]	)	
);

// ================== 原视频通路选择 ==================
// 新增ADAS图像链从base_*取数；当ENABLE_ADAS_IMAGE_PIPELINE=0时，顶层仍
// 直接输出base_*，从而优先保证师兄原视频工程不受影响。
reg        base_vs, base_hs, base_de;
reg [7:0]  base_r,  base_g,  base_b;

always @(*) begin
    if (mode_toggle == 1'b1) begin
        base_vs = o_gray_vsyn_2;
        base_hs = o_gray_hsyn_2;
        base_de = o_gray_de_2;
        base_r  = o_gray_data_2[23:16];
        base_g  = o_gray_data_2[15:8];
        base_b  = o_gray_data_2[7:0];
    end
    else begin
        case (key_choose)
            3'b000: begin base_vs=o_gray_vsyn_1; base_hs=o_gray_hsyn_1; base_de=o_gray_de_1; base_r=o_gray_data_1[23:16]; base_g=o_gray_data_1[15:8]; base_b=o_gray_data_1[7:0]; end
            3'b001: begin base_vs=o_gray_vsyn_3; base_hs=o_gray_hsyn_3; base_de=o_gray_de_3; base_r=o_gray_data_3[23:16]; base_g=o_gray_data_3[15:8]; base_b=o_gray_data_3[7:0]; end
            3'b010: begin base_vs=o_gray_vsyn_4; base_hs=o_gray_hsyn_4; base_de=o_gray_de_4; base_r=o_gray_data_4[23:16]; base_g=o_gray_data_4[15:8]; base_b=o_gray_data_4[7:0]; end
            3'b011: begin base_vs=o_gray_vsyn_5; base_hs=o_gray_hsyn_5; base_de=o_gray_de_5; base_r=o_gray_data_5[23:16]; base_g=o_gray_data_5[15:8]; base_b=o_gray_data_5[7:0]; end
            3'b100: begin base_vs=o_gray_vsyn_6; base_hs=o_gray_hsyn_6; base_de=o_gray_de_6; base_r=o_gray_data_6; base_g=o_gray_data_6; base_b=o_gray_data_6; end
            3'b101: begin base_vs=o_gray_vsyn_7; base_hs=o_gray_hsyn_7; base_de=o_gray_de_7; base_r=o_gray_data_7; base_g=o_gray_data_7; base_b=o_gray_data_7; end
            3'b110: begin base_vs=o_gray_vsyn_8; base_hs=o_gray_hsyn_8; base_de=o_gray_de_8; base_r=o_gray_data_8[23:16]; base_g=o_gray_data_8[15:8]; base_b=o_gray_data_8[7:0]; end
            default: begin base_vs=r_vs_out_1; base_hs=r_hs_out_1; base_de=r_de_out_1; base_r=r_r_out_1; base_g=r_g_out_1; base_b=r_b_out_1; end
        endcase
    end
end

// ================== ADAS图像通路 ==================
wire        night_vs, night_hs, night_de;
wire [7:0]  night_r, night_g, night_b;
wire        roi_vs, roi_hs, roi_de, roi_valid;
wire [7:0]  roi_r, roi_g, roi_b;
wire [11:0] roi_x, roi_y;
wire        lane_departure_warning;
wire [1:0]  lane_departure_direction;
wire        lane_pixel_debug;
wire        overlay_vs, overlay_hs, overlay_de;
wire [7:0]  overlay_r, overlay_g, overlay_b;

// ARM模型训练在FPGA车道模块验证完成后开展。预留对接信息为：
// lane_type_result[1:0]、lane_departure_warning和lane_departure_direction[1:0]；
// 后续可由packet_code扩展命令返回ARM，不在当前第一里程碑中传输。

night_enhance u_night_enhance (
    .pix_clk        (pix_clk_out),
    .rst_n          (ddr_ip_rst_n && ddr_init_done),
    .enhance_enable (ENABLE_NIGHT_ENHANCE),
    .i_vsync        (base_vs),
    .i_hsync        (base_hs),
    .i_de           (base_de),
    .i_r            (base_r),
    .i_g            (base_g),
    .i_b            (base_b),
    .o_vsync        (night_vs),
    .o_hsync        (night_hs),
    .o_de           (night_de),
    .o_r            (night_r),
    .o_g            (night_g),
    .o_b            (night_b)
);

lane_roi_mask #(
    .H_ACTIVE (VIDEO_LENGTH),
    .V_ACTIVE (VIDEO_HIGTH)
) u_lane_roi_mask (
    .pix_clk     (pix_clk_out),
    .rst_n       (ddr_ip_rst_n && ddr_init_done),
    .i_vsync     (night_vs),
    .i_hsync     (night_hs),
    .i_de        (night_de),
    .i_r         (night_r),
    .i_g         (night_g),
    .i_b         (night_b),
    .o_vsync     (roi_vs),
    .o_hsync     (roi_hs),
    .o_de        (roi_de),
    .o_r         (roi_r),
    .o_g         (roi_g),
    .o_b         (roi_b),
    .o_roi_valid (roi_valid),
    .o_x         (roi_x),
    .o_y         (roi_y)
);

lane_type_detect #(
    .H_ACTIVE (VIDEO_LENGTH)
) u_lane_type_detect (
    .pix_clk                 (pix_clk_out),
    .rst_n                   (ddr_ip_rst_n && ddr_init_done),
    .i_vsync                 (roi_vs),
    .i_de                    (roi_de),
    .i_roi_valid             (roi_valid),
    .i_x                     (roi_x),
    .i_y                     (roi_y),
    .i_r                     (roi_r),
    .i_g                     (roi_g),
    .i_b                     (roi_b),
    .lane_type               (lane_type_result),
    .lane_departure_warning  (lane_departure_warning),
    .departure_direction     (lane_departure_direction),
    .lane_pixel_debug        (lane_pixel_debug)
);

// ref_clk域的稳定决策经过双触发器同步后进入像素时钟域。
reg       stop_pix_sync0, stop_pix_sync1;
reg       alarm_pix_sync0, alarm_pix_sync1;
reg [1:0] decision_pix_sync0, decision_pix_sync1;
reg [2:0] action_pix_sync0, action_pix_sync1;
reg [2:0] light_pix_sync0, light_pix_sync1;
reg boxes_toggle_sync0, boxes_toggle_sync1, boxes_toggle_seen;
reg [3:0] boxes_valid_pix, boxes_model_id_pix;
reg [31:0] boxes_class_id_pix;
reg [63:0] boxes_x1_pix, boxes_y1_pix, boxes_x2_pix, boxes_y2_pix;
always @(posedge pix_clk_out) begin
    if (!(ddr_ip_rst_n && ddr_init_done)) begin
        stop_pix_sync0     <= 1'b0;
        stop_pix_sync1     <= 1'b0;
        alarm_pix_sync0    <= 1'b0;
        alarm_pix_sync1    <= 1'b0;
        decision_pix_sync0 <= 2'b00;
        decision_pix_sync1 <= 2'b00;
        action_pix_sync0   <= 3'd0;
        action_pix_sync1   <= 3'd0;
        light_pix_sync0    <= 3'd0;
        light_pix_sync1    <= 3'd0;
        boxes_toggle_sync0 <= 1'b0;
        boxes_toggle_sync1 <= 1'b0;
        boxes_toggle_seen  <= 1'b0;
        boxes_valid_pix    <= 4'd0;
        boxes_model_id_pix <= 4'd0;
        boxes_class_id_pix <= 32'd0;
        boxes_x1_pix<=0; boxes_y1_pix<=0; boxes_x2_pix<=0; boxes_y2_pix<=0;
    end
    else begin
        stop_pix_sync0     <= adas_stop_effective;
        stop_pix_sync1     <= stop_pix_sync0;
        alarm_pix_sync0    <= adas_alarm_enable;
        alarm_pix_sync1    <= alarm_pix_sync0;
        decision_pix_sync0 <= adas_decision_code;
        decision_pix_sync1 <= decision_pix_sync0;
        action_pix_sync0   <= adas_action_effective;
        action_pix_sync1   <= action_pix_sync0;
        light_pix_sync0    <= adas_status_light_state;
        light_pix_sync1    <= light_pix_sync0;
        boxes_toggle_sync0 <= boxes_update_toggle_ref;
        boxes_toggle_sync1 <= boxes_toggle_sync0;
        if (boxes_toggle_sync1 != boxes_toggle_seen) begin
            boxes_toggle_seen<=boxes_toggle_sync1;
            boxes_valid_pix<=boxes_valid_ref;
            boxes_model_id_pix<=boxes_model_id_ref;
            boxes_class_id_pix<=boxes_class_id_ref;
            boxes_x1_pix<=boxes_x1_ref; boxes_y1_pix<=boxes_y1_ref;
            boxes_x2_pix<=boxes_x2_ref; boxes_y2_pix<=boxes_y2_ref;
        end
    end
end

overlay #(
    .H_ACTIVE (VIDEO_LENGTH),
    .V_ACTIVE (VIDEO_HIGTH)
) u_adas_overlay (
    .pix_clk                  (pix_clk_out),
    .rst_n                    (ddr_ip_rst_n && ddr_init_done),
    .overlay_enable           (ENABLE_ADAS_IMAGE_PIPELINE),
    .i_vsync                  (roi_vs),
    .i_hsync                  (roi_hs),
    .i_de                     (roi_de),
    .i_x                      (roi_x),
    .i_y                      (roi_y),
    .i_r                      (roi_r),
    .i_g                      (roi_g),
    .i_b                      (roi_b),
    .stop_enable              (stop_pix_sync1),
    .alarm_enable             (alarm_pix_sync1),
    .decision_code            (decision_pix_sync1),
    .action_code              (action_pix_sync1),
    .light_state              (light_pix_sync1),
    .lane_departure_warning   (lane_departure_warning),
    .boxes_valid              (boxes_valid_pix),
    .boxes_model_id           (boxes_model_id_pix),
    .boxes_class_id           (boxes_class_id_pix),
    .boxes_x1                 (boxes_x1_pix),
    .boxes_y1                 (boxes_y1_pix),
    .boxes_x2                 (boxes_x2_pix),
    .boxes_y2                 (boxes_y2_pix),
    .o_vsync                  (overlay_vs),
    .o_hsync                  (overlay_hs),
    .o_de                     (overlay_de),
    .o_r                      (overlay_r),
    .o_g                      (overlay_g),
    .o_b                      (overlay_b)
);

// 主开关为0时完全旁路新增图像链，优先验证UART+决策+LED。
wire legacy_final_vs = ENABLE_ADAS_IMAGE_PIPELINE ? overlay_vs : base_vs;
wire legacy_final_hs = ENABLE_ADAS_IMAGE_PIPELINE ? overlay_hs : base_hs;
wire legacy_final_de = ENABLE_ADAS_IMAGE_PIPELINE ? overlay_de : base_de;
wire [23:0] legacy_final_rgb = ENABLE_ADAS_IMAGE_PIPELINE ? {overlay_r,overlay_g,overlay_b} : {base_r,base_g,base_b};
wire fullres_vs,fullres_hs,fullres_de;
wire [23:0] fullres_rgb;
adas_video_output_mux u_adas_video_output_mux(
 .clk(pix_clk_in),.rst_n(ddr_ip_rst_n && ddr_init_done),.select_fullres(key_choose==3'b111),
 .raw_vs(vs_in),.raw_hs(hs_in),.raw_de(de_in),.raw_rgb({r_in,g_in,b_in}),
 .legacy_vs(legacy_final_vs),.legacy_hs(legacy_final_hs),.legacy_de(legacy_final_de),.legacy_rgb(legacy_final_rgb),
 .o_vs(fullres_vs),.o_hs(fullres_hs),.o_de(fullres_de),.o_rgb(fullres_rgb));
always @(*) begin
 r_vs_out=fullres_vs;r_hs_out=fullres_hs;r_de_out=fullres_de;
 r_r_out=fullres_rgb[23:16];r_g_out=fullres_rgb[15:8];r_b_out=fullres_rgb[7:0];
end
//�����Ҹ���������������룬�ٸĻ�������
// ����Ĵ�����������ԭ�˿� r_vs_out �ȣ�
// always @(posedge pix_clk_out) begin
    // r_vs_out <= final_vs;
    // r_hs_out <= final_hs;
    // r_de_out <= final_de;
    // r_r_out  <= r_r_out;
    // r_g_out  <= r_g_out;
    // r_b_out  <= r_b_out;
// end

//����de����
always @(posedge pix_clk_in) begin//ץ�½���
    if(!ddr_ip_rst_n) begin  
        vs_in_d0 <= 'd0;
        vs_in_d1 <= 'd0;
        de_in_d0 <= 'd0;
        de_in_d1 <= 'd0;
        de_in_cnt <= 'd0; 
        de_in_state <= 0;
    end
    else begin
       case(de_in_state) 
            DE_IN_WAIT:
            begin
                vs_in_d0 <= vs_in;
                vs_in_d1 <= vs_in_d0;
                if(!vs_in_d0 && vs_in_d1) begin
                    de_in_state <= DE_IN_CNT;//ץȡvs_in�½��أ���ץ���½���ʱ��ʼ����
                end
            end
            DE_IN_CNT:
            begin
                de_in_d0 <= de_in;
                de_in_d1 <= de_in_d0;
                if(de_in_d0 && !de_in_d1) begin
                    de_in_cnt <= de_in_cnt + 1'd1;//ץȡde�����أ�de����ʱ����
                end
                else if(de_in_cnt == VIDEO_HIGTH) begin
                    de_in_state <= DE_IN_END;
                end
            end
            DE_IN_END:
            begin
                vs_in_d0 <= vs_in;
                vs_in_d1 <= vs_in_d0;
                if(vs_in_d0 && !vs_in_d1) begin
                    de_in_cnt <= 'd0;
                    de_in_state <= DE_IN_WAIT;//ץȡvs_in�����أ���ץ��������ʱ��������
                end
            end
        endcase  
    end
end
//����zoom de����
always @(posedge pix_clk_in) begin//ץ�½���
    if(!ddr_ip_rst_n) begin  
        zoom_vs_in_d0    <= 'd0;
        zoom_vs_in_d1    <= 'd0;
        zoom_de_in_d0    <= 'd0;
        zoom_de_in_d1    <= 'd0;
        zoom_de_in_cnt   <= 'd0; 
        zoom_de_in_state <= 0;
    end
    else begin
       case(zoom_de_in_state) 
            DE_IN_WAIT:
            begin
                zoom_vs_in_d0 <= vs_in;
                zoom_vs_in_d1 <= zoom_vs_in_d0;
                if(!zoom_vs_in_d0 && zoom_vs_in_d1) begin
                    zoom_de_in_state <= DE_IN_CNT;//ץȡvs_in�½��أ���ץ���½���ʱ��ʼ����
                end
            end
            DE_IN_CNT:
            begin
                zoom_de_in_d0 <= zoom_de_out;
                zoom_de_in_d1 <= zoom_de_in_d0;
                if(zoom_de_in_d0 && !zoom_de_in_d1) begin
                    zoom_de_in_cnt <= zoom_de_in_cnt + 1'd1;//ץȡde�����أ�de����ʱ����
                end
                else if(zoom_de_in_cnt == ZOOM_VIDEO_HIGTH) begin
                    zoom_de_in_state <= DE_IN_END;
                end
            end
            DE_IN_END:
            begin
                zoom_vs_in_d0 <= vs_in;
                zoom_vs_in_d1 <= zoom_vs_in_d0;
                if(vs_in_d0 && !vs_in_d1) begin
                    zoom_de_in_cnt <= 'd0;
                    zoom_de_in_state <= DE_IN_WAIT;//ץȡvs_in�����أ���ץ��������ʱ��������
                end
            end
        endcase  
    end
end
//����de_out����
always @(posedge pix_clk_out) begin//ץ�½���
    if(!ddr_ip_rst_n) begin  
        vs_out_d0 <= 'd0;
        vs_out_d1 <= 'd0;
        de_out_d0 <= 'd0;
        de_out_d1 <= 'd0;
        de_out_cnt <= 'd0; 
        de_out_state <= 'd0;
    end
    else begin
       case(de_out_state) 
            DE_OUT_WAIT:
            begin
                vs_out_d0 <= vs_out;
                vs_out_d1 <= vs_out_d0;
                if(!vs_out_d0 && vs_out_d1) begin
                    de_out_state <= DE_OUT_CNT;//ץȡvs_in�½��أ���ץ���½���ʱ��ʼ����
                end
            end
            DE_OUT_CNT:
            begin
                de_out_d0 <= de_out;
                de_out_d1 <= de_out_d0;
                if (de_out_d0 && !de_out_d1) begin
                    de_out_cnt <= de_out_cnt + 1'd1;//ץȡde�����أ�de����ʱ����
                end
                else if(de_out_cnt == VIDEO_HIGTH) begin
                    de_out_state <= DE_OUT_END;
                end
            end
            DE_OUT_END:
            begin
                vs_out_d0 <= vs_out;
                vs_out_d1 <= vs_out_d0;
                if(vs_out_d0 && !vs_out_d1) begin
                    de_out_cnt <= 'd0;
                    de_out_state <= DE_OUT_WAIT;//ץȡvs_in�����أ���ץ��������ʱ��������
                end
            end
        endcase  
    end
end


//���������źŽ������
always @(posedge pix_clk_out) begin
    if(!ddr_ip_rst_n) begin
        r_vs_out_1 <= 'd0;
        r_hs_out_1 <= 'd0;
        r_de_out_1 <= 'd0;
        r_r_out_1  <= 'd0;
        r_g_out_1  <= 'd0;
        r_b_out_1  <= 'd0;
        v_sync_flag <= 'd0;
        video0_rd_en <= 1'b0; 
        video1_rd_en <= 1'b0; 
        video2_rd_en <= 1'b0; 
        video3_rd_en <= 1'b0; 
        video_pre_rd_flag <= 1'b0;        
        out_state <= 'd0;
    end 
    else if(ddr_init_done) begin 
        r_vs_out_d0 <= vs_out;
        r_vs_out_1    <= r_vs_out_d0;
        r_hs_out_1 <= hs_out;
        r_de_out_d0 <= de_out;
        r_de_out_1 <= r_de_out_d0;
        r_x_act_d0 <= x_act;//X���������Ŷ��+һ��
        r_x_act <= r_x_act_d0;

       if(vs_out_d0 && !vs_out_d1) begin
            video_pre_rd_flag <= 'd0;
       end
       else if(!vs_out_d0 && vs_out_d1 && !video_pre_rd_flag && (fram0_done || fram1_done || fram2_done || fram3_done)) begin
            video0_rd_en      <= 'd1;
            video1_rd_en      <= 'd1;
            video2_rd_en      <= 'd1;
            video3_rd_en      <= 'd1;
            video_pre_rd_flag <= 'd1;
            out_state         <= 'd1;
       end
       else begin
           if( fram0_done && (r_x_act >= 0) && (r_x_act < ZOOM_VIDEO_LENGTH - 1) && (y_act < ZOOM_VIDEO_HIGTH ) && (y_act >= 0)) begin//���Ͻ�����
                r_r_out_1  <=  video0_data_out[31:24];
                r_g_out_1  <=  video0_data_out[21:14];
                r_b_out_1  <=  video0_data_out[11: 4];  
                video0_rd_en <= de_out; //Ԥ����
                video1_rd_en <= 'd0; 
                video2_rd_en <= 'd0; 
                video3_rd_en <= 'd0; //'d0; 
                out_state    <= 'd2;
                if(r_x_act == ZOOM_VIDEO_LENGTH - 2) begin
                    video1_rd_en <= de_out;
                    video0_rd_en <= 'd0;
                end
            end
            if(fram1_done &&(r_x_act >= ZOOM_VIDEO_LENGTH - 1) && (r_x_act < VIDEO_LENGTH - 1) && (y_act < ZOOM_VIDEO_HIGTH )&& (y_act >= 0)) begin
                r_r_out_1  <= video1_data_out[31:24];
                r_g_out_1  <= video1_data_out[21:14];
                r_b_out_1  <= video1_data_out[11: 4];
                video0_rd_en <= 'd0; 
                video1_rd_en <= de_out; 
                video2_rd_en <= 'd0; 
                video3_rd_en <= 'd0; 
                out_state    <= 'd3;

            end  
            if( fram2_done &&(r_x_act >= 0) && (r_x_act < ZOOM_VIDEO_LENGTH - 1) && (y_act < VIDEO_HIGTH )&& (y_act >= ZOOM_VIDEO_HIGTH)) begin
                r_r_out_1  <= video2_data_out[31:24] ;
                r_g_out_1  <= video2_data_out[21:14] ;
                r_b_out_1  <= video2_data_out[11: 4] ;  
                video0_rd_en <= 'd0; 
                video1_rd_en <= 'd0; 
                video2_rd_en <= de_out; 
                video3_rd_en <= 'd0; 
                out_state    <= 'd4;
                if(r_x_act == ZOOM_VIDEO_LENGTH - 2) begin
                    video3_rd_en <= de_out;
                    video2_rd_en <= 'd0;
                end
            end    
            if(fram3_done &&(r_x_act >= ZOOM_VIDEO_LENGTH - 1) && (r_x_act < VIDEO_LENGTH - 1) && (y_act < VIDEO_HIGTH )&& (y_act >= ZOOM_VIDEO_HIGTH)) begin
                r_r_out_1  <= video3_data_out[31:24];
                r_g_out_1  <= video3_data_out[21:14];
                r_b_out_1  <= video3_data_out[11: 4];     
                video0_rd_en <= 'd0; 
                video1_rd_en <= 'd0; 
                video2_rd_en <= 'd0; 
                video3_rd_en <= de_out; 
                
                out_state    <= 'd5;
            end 
        end         
    end
    else begin
        r_vs_out_1 <= 'd0;
        r_hs_out_1 <= 'd0;
        r_de_out_1 <= 'd0;
        r_r_out_1  <= 8'hff ;
        r_g_out_1  <= 8'h00 ;
        r_b_out_1  <= 8'h00 ;
        video0_rd_en <= 1'b0; 
        video1_rd_en <= 1'b0; 
        video2_rd_en <= 1'b0; 
        video3_rd_en <= 1'b0; 
        out_state    <= 'd7; 
    end            
end

always @(posedge iic_clk)begin
	if(!pll_init_done)
	    rstn_1ms <= 16'd0;
	else if(!ddr_init_done) begin
  	    rstn_1ms <= 16'd0;  
    end
	else if(!rst_board) begin
  	    rstn_1ms <= 16'd0;  
    end
	else begin
		if(rstn_1ms == 16'h2710)
		    rstn_1ms <= rstn_1ms;
		else
		    rstn_1ms <= rstn_1ms + 1'b1;
	end
end
assign hdmi_rst = (rstn_1ms == 16'h2710);
hdmi_ctrl user_hdmi_ctrl(
    .clk         (  iic_clk    ), //input       clk,
    .rst_n       (  hdmi_rst   ), //input       rstn, 
    .init_over   (  hdmi_int_led  ), //output      init_over,
    .iic_tx_scl  (  iic_tx_scl ), //output      iic_scl,
    .iic_tx_sda  (  iic_tx_sda ), //inout       iic_sda
    .iic_scl     (  iic_scl    ), //output      iic_scl,
    .iic_sda     (  iic_sda    )  //inout       iic_sda
);
//ov5640����
//CMOS1 720P
ov5640_reg_cfg_0	coms1_reg_config(
	.clk_25M                 (clk_25M            ),//input
	.camera_rstn             (cmos1_reset        ),//input
	.initial_en              (initial_en         ),//input		
	.i2c_sclk                (cmos1_scl          ),//output
	.i2c_sdat                (cmos1_sda          ),//inout
	.reg_conf_done           (cmos_init_done[0]  ),//output config_finished
	.reg_index               (                   ),//output reg [8:0]
	.clock_20k               (                   ) //output reg
);
//CMOS2 720P
ov5640_reg_config	coms2_reg_config(
    	.clk_25M                 (clk_25M            ),//input
    	.camera_rstn             (cmos2_reset        ),//input
    	.initial_en              (initial_en         ),//input		
    	.i2c_sclk                (cmos2_scl          ),//output
    	.i2c_sdat                (cmos2_sda          ),//inout
    	.reg_conf_done           (cmos_init_done[1]  ),//output config_finished
    	.reg_index               (                   ),//output reg [8:0]
    	.clock_20k               (                   ) //output reg
    );
ov5640_power_on_delay	power_on_delay_inst(
	.clk_50M                 (ref_clk        ),//input
	.reset_n                 (ddr_ip_rst_n           ),//input	
	.camera1_rstn            (cmos1_reset    ),//output
	.camera2_rstn            (cmos2_reset    ),//output	
	.camera_pwnd             (               ),//output
	.initial_en              (initial_en     ) //output		
);
//CMOS 8bitת16bit//////////
//CMOS1
always@(posedge cmos1_pclk)
    begin
        cmos1_d_d0        <= cmos1_data    ;
        cmos1_href_d0     <= cmos1_href    ;
        cmos1_vsync_d0    <= cmos1_vsync   ;
    end

cmos_8_16bit cmos1_8_16bit(
	.pclk           (cmos1_pclk       ),//input
	.rst_n          (cmos_init_done[0]),//input
	.pdata_i        (cmos1_d_d0       ),//input[7:0]
	.de_i           (cmos1_href_d0    ),//input
	.vs_i           (cmos1_vsync_d0    ),//input
	
	.pixel_clk      (cmos1_pclk_16bit ),//output
	.pdata_o        (cmos1_d_16bit    ),//output[15:0]
	.de_o           (cmos1_href_16bit ) //output
);
//CMOS2
always@(posedge cmos2_pclk)
    begin
        cmos2_d_d0        <= cmos2_data    ;
        cmos2_href_d0     <= cmos2_href    ;
        cmos2_vsync_d0    <= cmos2_vsync   ;
    end

cmos_8_16bit cmos2_8_16bit(
	.pclk           (cmos2_pclk       ),//input
	.rst_n          (cmos_init_done[1]),//input
	.pdata_i        (cmos2_d_d0       ),//input[7:0]
	.de_i           (cmos2_href_d0    ),//input
	.vs_i           (cmos2_vsync_d0    ),//input
	
	.pixel_clk      (cmos2_pclk_16bit ),//output
	.pdata_o        (cmos2_d_16bit    ),//output[15:0]
	.de_o           (cmos2_href_16bit ) //output
);

 

ipsxb_rst_sync_v1_1 u_core_clk_rst_sync(
    .clk                        (ref_clk        ),
    .rst_n                      (rst_board       ),
    .sig_async                  (1'b1),
    .sig_synced                 (ddr_ip_rst_n   )
);

//��ʼ��˳��DDR->AXI_M & FIFO ->IIC PLL -HDMI
axi_m_arbitration #(
    .VIDEO_LENGTH     (VIDEO_LENGTH)                    ,
    .VIDEO_HIGTH      (VIDEO_HIGTH)                     ,
    .ZOOM_VIDEO_LENGTH(ZOOM_VIDEO_LENGTH )              ,
    .ZOOM_VIDEO_HIGTH (ZOOM_VIDEO_HIGTH )               ,
    .PIXEL_WIDTH      (PIXEL_WIDTH  )                            ,
	.CTRL_ADDR_WIDTH  (CTRL_ADDR_WIDTH  )                            ,
	.DQ_WIDTH	     (DQ_WIDTH  )                            ,
    .M_AXI_BRUST_LEN  (M_AXI_BRUST_LEN   )
)
user_axi_m_arbitration (
	.DDR_INIT_DONE           (ddr_init_done),
	.M_AXI_ACLK              (ddr_ip_clk   ),
	.M_AXI_ARESETN           (ddr_ip_rst_n  && ddr_init_done),
     .pix_clk_out             (pix_clk_out),//1080p 148.5m
      
	//д��ַͨ����                                                              
	.M_AXI_AWID              (M_AXI_AWID   ),
	.M_AXI_AWADDR            (M_AXI_AWADDR ),
//	.M_AXI_AWLEN             (),
	.M_AXI_AWUSER            (M_AXI_AWUSER ),
	.M_AXI_AWVALID           (M_AXI_AWVALID),
	.M_AXI_AWREADY           (M_AXI_AWREADY),
	//д����ͨ����                                                              
	.M_AXI_WDATA             (M_AXI_WDATA),
	.M_AXI_WSTRB             (M_AXI_WSTRB),
	.M_AXI_WLAST             (M_AXI_WLAST),
	.M_AXI_WUSER             (M_AXI_WUSER),
	.M_AXI_WREADY            (M_AXI_WREADY),
                                                             
	//����ַͨ����                                                              
	.M_AXI_ARID              (M_AXI_ARID),
    .M_AXI_ARUSER            (M_AXI_ARUSER),
	.M_AXI_ARADDR            (M_AXI_ARADDR),
//	.M_AXI_ARLEN             (),
	.M_AXI_ARVALID           (M_AXI_ARVALID),
	.M_AXI_ARREADY           (M_AXI_ARREADY),
	//������ͨ����                                                              
	.M_AXI_RID               (M_AXI_RID   ),
	.M_AXI_RDATA             (M_AXI_RDATA ),
	.M_AXI_RLAST             (M_AXI_RLAST ),
	.M_AXI_RVALID            (M_AXI_RVALID),
    //video
    .vs_in                   (vs_in       ),
    .vs_out                  (vs_out      ),
//fifo0�źţ�HDMI����          
    .video0_clk_in           (pix_clk_in),                                                                                                                  
    .video0_de_in            (hdmi_frame_valid    ),
    .video0_data_in          ({hdmi_frame_data[23:16],2'd0,hdmi_frame_data[15:8],2'd0,hdmi_frame_data[7:0],4'd0}  ),
    .video0_rd_en            (video0_rd_en   ),
    .video0_data_out         (video0_data_out),
    .fram0_done              (fram0_done     ),
    .video0_vs_in            (hdmi_frame_vs ),//ץ��λ
//fifo1�źţ��������
    .video1_clk_in           (pix_clk_in),                                                               
    .video1_de_in            (hdmi_frame_valid    ),
    .video1_data_in          ({hdmi_frame_data[23:16],2'd0,hdmi_frame_data[15:8],2'd0,hdmi_frame_data[7:0],4'd0} ),
    .video1_rd_en            (video1_rd_en   ),
    .video1_data_out         (video1_data_out),
    .fram1_done              (fram1_done     ),
    .video1_vs_in            (hdmi_frame_vs ),//ץ��λ  
//fifo2�źţ���CMOS1                                  
    .video2_clk_in           (cmos1_pclk_16bit),                       
    .video2_de_in            (cmos1_href_16bit ),
    .video2_data_in          ({cmos1_d_16bit[4:0],5'b0,cmos1_d_16bit[10:5],4'b0,cmos1_d_16bit[15:11],7'b0}),//27
    .video2_rd_en            (video2_rd_en   ),
    .video2_data_out         (video2_data_out),
    .fram2_done              (fram2_done     ),
    .video2_vs_in            (cmos1_vsync_d0 ),//ץ��λ
//fifo3�źţ���CMOS2                                      
    .video3_clk_in           (cmos2_pclk_16bit),                       
    .video3_de_in            (cmos2_href_16bit    ),
    .video3_data_in          ({cmos2_d_16bit[4:0],5'b0,cmos2_d_16bit[10:5],4'b0,cmos2_d_16bit[15:11],7'b0}  ),
    .video3_rd_en            (video3_rd_en   ),
    .video3_data_out         (video3_data_out),
    .fram3_done              (fram3_done     ),
    .video3_vs_in            (cmos2_vsync_d0 ),//ץ��λ
    //����
    .wr_addr_min             (RW_ADDR_MIN),
    .wr_addr_max             (RW_ADDR_MAX), 
    .y_act                   (y_act)        , 
    .x_act                   (x_act)  

);

sync_generator user_sync_gen(
    .clk       (pix_clk_out ),//1080p148.5mhz
    .rstn      (ddr_ip_rst_n && ddr_init_done),
    .vs_out    (vs_out),
    .hs_out    (hs_out),
    .de_out    (de_out),
    .de_re     (),
    .x_act     (x_act),
    .y_act     (y_act)
);

wire             hdmi_frame_vs   ;
wire             hdmi_frame_hs   ;
wire             hdmi_frame_valid;
wire    [23:0]   hdmi_frame_data ;
   
// ================== ���ƶ�λ����Դ��ʹ��ǰ��ѡ���� RGB �źţ� ==================
/*wire [7:0] plate_in_r  = pre_r_out;   // ǰ�� mux ���
wire [7:0] plate_in_g  = pre_g_out;
wire [7:0] plate_in_b  = pre_b_out;
wire       vs_src      = pre_vs_out;
wire       hs_src      = pre_hs_out;
wire       de_src      = pre_de_out;
 
// ================== ���ƶ�λ���������ź� ==================
wire [7:0] plate_r, plate_g, plate_b;
wire       plate_vs, plate_hs, plate_de;
wire       plate_box_valid;
wire [11:0] plate_x0, plate_y0, plate_x1, plate_y1;
wire       plate_mask_color, plate_edge_close;

// ================== ��̬��ֵ��֡��ֵ����Ӧ�������� RGB �������ȣ� ==================
// ���Ƚ��ƣ�Y �� (R + 2G + B) >> 2   (0~255)
wire [9:0] luma_now = ( {2'b0, plate_in_r} + {1'b0, plate_in_g, 1'b0} + {2'b0, plate_in_b} ) >> 2;

reg        vs_d0;
wire       vs_fall = vs_d0 & ~vs_src;   // ֡�߽磺vs �½��أ����� vs �����෴���ɸĳ������أ�

reg [31:0] sum_luma;
reg [21:0] cnt_luma;
reg [7:0]  avg_luma;                    // 0~255

always @(posedge pix_clk_out or negedge (ddr_ip_rst_n && ddr_init_done)) begin
  if(!(ddr_ip_rst_n && ddr_init_done)) begin
    vs_d0     <= 1'b0;
    sum_luma  <= 32'd0;
    cnt_luma  <= 22'd0;
    avg_luma  <= 8'd80;                 // �ϵ�Ĭ�ϸ����е����ȣ������һ֡��ֵ̫����
  end else begin
    vs_d0 <= vs_src;

    if(de_src) begin
      sum_luma <= sum_luma + luma_now;
      cnt_luma <= cnt_luma + 1'b1;
    end

    if(vs_fall) begin
      // 1920*1080=2,073,600 �� 2^21=2,097,152���� >>21 �������ֵ
      // ��������Ч���ز���ȫ 1080p���ɰ�����λ������
      if(cnt_luma != 0)
        avg_luma <= sum_luma[31:21];    // �ȼ��� sum_luma >> 21��ֻȡ 8bit �㹻��
      sum_luma <= 32'd0;
      cnt_luma <= 22'd0;
    end
  end
end

// ��̬��ֵ������Խ����v_min Խ�ͣ�����Խ����v_min Խ��
// ��̬��ֵ����Ӧ�߼�
// �����������(��ֵ<50)������V������Ҫ��ȥץȡ�������ƣ�����������������V�����˳������߹���š�
wire[7:0] v_min_dyn = (avg_luma < 8'd50)  ? 8'd40 :
                       (avg_luma > 8'd150) ? 8'd80 : 
                                             8'd60 ; // ��������ʹ��60

// ���Ͷ���ֵ���������ȣ���ֹ��������������ɫ��Ҳ��ֱ�ӹ̶� 35
wire [7:0] s_min_dyn = (avg_luma < 8'd50)  ? 8'd25 : 8'd35;  
    
/*	wire [7:0] v_min_dyn = 8'd70;  // ���빻��
wire [7:0] s_min_dyn = 8'd50;  // ���빻����ɫ�ʱ��ͣ�*/
/*wire [11:0] dbg_cc_x, dbg_cc_y, dbg_cc_row_cnt;
wire dbg_cc_band_hit, dbg_cc_hit_any;

plate_locate_top #(
    .H_ACTIVE(1920),
    .V_ACTIVE(1080)
) u_plate_locate_top (
    .pix_clk     (pix_clk_out),
    .rst_n       (ddr_ip_rst_n && ddr_init_done),

    .i_vs        (vs_src),
    .i_hs        (hs_src),
    .i_de        (de_src),
    .i_r         (plate_in_r),
    .i_g         (plate_in_g),
    .i_b         (plate_in_b),

    .blue_h_min  (8'd110),
    .blue_h_max  (8'd170),
    .green_h_min (8'd0),
    .green_h_max (8'd0),
    .edge_th     (8'd30),

    .i_s_min     (s_min_dyn),
    .i_v_min     (v_min_dyn),

    .o_mask_color(plate_mask_color),
    .o_edge_close(plate_edge_close),
    .o_box_valid (plate_box_valid),
    .o_x0        (plate_x0),
    .o_y0        (plate_y0),
    .o_x1        (plate_x1),
    .o_y1        (plate_y1),

    .o_vs        (plate_vs),
    .o_hs        (plate_hs),
    .o_de        (plate_de),
    .o_x         (plate_x),
    .o_y         (plate_y),
    .o_r         (plate_r),
    .o_g         (plate_g),
    .o_b         (plate_b)
    
);

// ========== ͬ���ӳ�������ǰ�����߶��룬6 �ģ�==========
reg [5:0] vs_src_d, hs_src_d, de_src_d;
always @(posedge pix_clk_out or negedge (ddr_ip_rst_n && ddr_init_done)) begin
    if(!(ddr_ip_rst_n && ddr_init_done)) begin
        vs_src_d <= 6'd0;
        hs_src_d <= 6'd0;
        de_src_d <= 6'd0;
    end else begin
        vs_src_d <= {vs_src_d[4:0], vs_src};
        hs_src_d <= {hs_src_d[4:0], hs_src};
        de_src_d <= {de_src_d[4:0], de_src};
    end
end

// ========== �����ӳٶ��루��ͬ���ӳ���ͬ���ȣ�==========
reg [11:0] x_act_d[0:5];
reg [11:0] y_act_d[0:5];
integer i;
always @(posedge pix_clk_out or negedge (ddr_ip_rst_n && ddr_init_done)) begin
    if(!(ddr_ip_rst_n && ddr_init_done)) begin
        for(i=0;i<6;i=i+1) begin
            x_act_d[i] <= 12'd0;
            y_act_d[i] <= 12'd0;
        end
    end else begin
        x_act_d[0] <= x_act;
        y_act_d[0] <= y_act;
        for(i=1;i<6;i=i+1) begin
            x_act_d[i] <= x_act_d[i-1];
            y_act_d[i] <= y_act_d[i-1];
        end
    end
end

// ========== ������ ==========
wire [11:0] px = x_act_d[5];
wire [11:0] py = y_act_d[5];
wire pix_valid = de_src_d[5];

localparam integer BOX_W = 4;

// ѡ�� bbox ��Դ
wire [11:0] bx0 = plate_x0;
wire [11:0] by0 = plate_y0;
wire [11:0] bx1 = plate_x1;
wire [11:0] by1 = plate_y1;
wire bvalid = plate_box_valid;

// BBox �Ϸ��Թ���
wire bbox_order_ok = (bx1 > bx0) && (by1 > by0);
wire [11:0] bbox_w = bbox_order_ok ? (bx1 - bx0) : 12'd0;
wire [11:0] bbox_h = bbox_order_ok ? (by1 - by0) : 12'd0;

// ������飺2.0 < w/h < 6.0
wire ratio_ok = (bbox_h != 0) &&
                (bbox_w > (bbox_h << 1)) &&
                (bbox_w < ((bbox_h << 2) + (bbox_h << 1)));

wire bvalid_ok = bvalid && bbox_order_ok && ratio_ok;

// ��߽���
wire hit_left_right =
    ( (px >= bx0) && (px <= bx0 + BOX_W - 1) && (py >= by0) && (py <= by1) ) ||
    ( (px >= bx1 - BOX_W + 1) && (px <= bx1) && (py >= by0) && (py <= by1) );

wire hit_top_bottom =
    ( (py >= by0) && (py <= by0 + BOX_W - 1) && (px >= bx0) && (px <= bx1) ) ||
    ( (py >= by1 - BOX_W + 1) && (py <= by1) && (px >= bx0) && (px <= bx1) );

wire in_box_border = bvalid_ok && pix_valid && (hit_left_right || hit_top_bottom);

// ========== ����ϳ� ==========
always @(posedge pix_clk_out or negedge (ddr_ip_rst_n && ddr_init_done)) begin
    if(!(ddr_ip_rst_n && ddr_init_done)) begin
        r_vs_out <= 1'b0;
        r_hs_out <= 1'b0;
        r_de_out <= 1'b0;
        r_r_out  <= 8'd0;
        r_g_out  <= 8'd0;
        r_b_out  <= 8'd0;
    end else begin
        r_vs_out <= vs_src_d[5];
        r_hs_out <= hs_src_d[5];
        r_de_out <= de_src_d[5];

        if(in_box_border) begin
            // ���
            r_r_out <= 8'hFF;
            r_g_out <= 8'h00;
            r_b_out <= 8'h00;
        end else begin
            // ԭʼ��Ƶ
            r_r_out <= plate_r;
            r_g_out <= plate_g;
            r_b_out <= plate_b;
        end
    end
end
*/
//������
  
Video_processing_top#(
    .HDMI_DATA_WIDTH   (24)    ,
    .HDMI_RGB_R_WIDTH  (8)     ,
    .HDMI_RGB_G_WIDTH  (8)     ,
    .HDMI_RGB_B_WIDTH  (8)     ,
    .HDMI_H_PIXEL      (1920)  ,
    .HDMI_V_PIXEL      (1080)  
)
 u_Video_processing_top(
    .sys_rst_n       (ddr_ip_rst_n && ddr_init_done),
    
    .hdmi_pix_clk    (pix_clk_in),
    .hdmi_vs         (vs_in),
    .hdmi_hs         (hs_in),
    .hdmi_de         (de_in),
    .hdmi_data       ({r_in,g_in,b_in}),
    
    .hdmi_frame_vs   (hdmi_frame_vs),
    .hdmi_frame_hs   (hdmi_frame_hs),
    .hdmi_frame_valid(hdmi_frame_valid),
    .hdmi_frame_data(hdmi_frame_data) 
);
   
   parameter H_CMOS_DISP=11'd960;
//��ڴ��䶥��ģ��   �������
/*sfp_8b10b_top u_sfp_8b10b_top(
    //��ڽӿ�
    .i_p_refckn_1               (i_p_refckn_1       ),
    .i_p_refckp_1               (i_p_refckp_1       ),
    .i_p_l2rxn                  (i_p_l2rxn          ),
    .i_p_l2rxp                  (i_p_l2rxp          ),
    .o_p_l2txn                  (o_p_l2txn          ),
    .o_p_l2txp                  (o_p_l2txp          ),
    .tx_disable                 (tx_disable         ),
	//�û��ӿ�
	.drp_clk    	    		(ref_clk            ),      //�������ʱ��  
	.clk_in     	   			(pix_clk_out           ),      //����ʱ��
	.rst_n						(ddr_ip_rst_n && ddr_init_done          ),      //���븴λ
	.vs_out						(post_frame_vsync_hsst   ),      //������ź�
	.data_valid_out				(post_frame_de_hsst      ),	    //���������Ч�ź�
	.data_out	   				(post_rgb           )       //�������
);   */
wire post_frame_vsync_hsst;
wire post_frame_de_hsst;
wire [15:0]post_rgb;

pll_cfg user_pll_cfg (
  .clkin1(ref_clk),        // input
  .pll_lock(pll_init_done),    // output
  .clkout0(iic_clk),      // output
  .clkout1(clk_25M)       // output
);
assign  pix_clk_out=pix_clk_in;
// pll_video_out user_pll_video_out (
  // .clkin1(ref_clk),        // input
  // .pll_lock(pll_lock),    // output
  // .clkout0(pix_clk_out)       // output
// );


reg [11 : 0]              cmos1_de_in_cnt    ;
reg [11 : 0]              cmos1_h_cnt        ;

reg                       cmos1_de_in_d0     ;
reg                       cmos1_de_in_d1     ;
reg                       cmos1_vs_in_d0     ;
reg                       cmos1_vs_in_d1     ;
reg [3 : 0]               cmos1_de_in_state  ;

always @(posedge cmos1_pclk_16bit) begin//ץ�½���
    if(!ddr_ip_rst_n) begin  
        cmos1_vs_in_d0 <= 'd0;
        cmos1_vs_in_d1 <= 'd0;
        cmos1_de_in_d0 <= 'd0;
        cmos1_de_in_d1 <= 'd0;
        cmos1_de_in_cnt <= 'd0; 
        cmos1_de_in_state <= 'd0;
        cmos1_h_cnt     <= 'd0;
    end
    else begin
       case(cmos1_de_in_state) 
            DE_IN_WAIT:
            begin
                cmos1_vs_in_d0 <= cmos1_vsync;
                cmos1_vs_in_d1 <= cmos1_vs_in_d0;
                if(!cmos1_vs_in_d0 && cmos1_vs_in_d1) begin
                    cmos1_de_in_state <= DE_IN_CNT;//ץȡvs_in�½��أ���ץ���½���ʱ��ʼ����
                end
            end
            DE_IN_CNT:
            begin
                cmos1_de_in_d0 <= cmos1_href_16bit;
                cmos1_de_in_d1 <= cmos1_de_in_d0;
                if(cmos1_de_in_d0 && !cmos1_de_in_d1) begin
                    cmos1_de_in_cnt <= cmos1_de_in_cnt + 1'd1;//ץȡde�����أ�de����ʱ����
                end
                if(cmos1_href_16bit) begin
                    cmos1_h_cnt <= cmos1_h_cnt +'d1;
                end
                else begin
                    cmos1_h_cnt <= 'd0;
                end
                cmos1_vs_in_d0 <= cmos1_vsync;
                cmos1_vs_in_d1 <= cmos1_vs_in_d0;
                if(cmos1_vs_in_d0 && !cmos1_vs_in_d1) begin
                    cmos1_de_in_cnt <= 'd0;
                    cmos1_de_in_state <= DE_IN_WAIT;//ץȡvs_in�����أ���ץ��������ʱ��������
                end
            end
        endcase  
    end
end


//ddr IP����
ddr_test  #
  (
   //***************************************************************************
   // The following parameters are Memory Feature
   //***************************************************************************
   .MEM_ROW_WIDTH          (MEM_ROW_ADDR_WIDTH),     
   .MEM_COLUMN_WIDTH       (MEM_COL_ADDR_WIDTH),     
   .MEM_BANK_WIDTH         (MEM_BADDR_WIDTH   ),     
   .MEM_DQ_WIDTH           (MEM_DQ_WIDTH      ),     
   .MEM_DM_WIDTH           (MEM_DM_WIDTH      ),     
   .MEM_DQS_WIDTH          (MEM_DQS_WIDTH     ),     
   .CTRL_ADDR_WIDTH        (CTRL_ADDR_WIDTH   )     
  )
  I_ipsxb_ddr_top(
   .ref_clk                (ref_clk                ),
   .resetn                 (ddr_ip_rst_n           ),
   .ddr_init_done          (ddr_init_done          ),
   .ddrphy_clkin           (ddr_ip_clk             ),
   .pll_lock               (ddr_pll_lock               ), 
    //д��ַ
   .axi_awaddr             (M_AXI_AWADDR           ),
   .axi_awuser_ap          (M_AXI_AWUSER           ),
   .axi_awuser_id          (M_AXI_AWID             ),
   .axi_awlen              (M_AXI_BRUST_LEN        ),
   .axi_awready            (M_AXI_AWREADY          ),
   .axi_awvalid            (M_AXI_AWVALID          ),
    //д����
   .axi_wdata              (M_AXI_WDATA            ),
   .axi_wstrb              (M_AXI_WSTRB            ),
   .axi_wready             (M_AXI_WREADY           ),
   .axi_wusero_id          (M_AXI_WUSER            ),
   .axi_wusero_last        (M_AXI_WLAST            ),
    //����ַ                   
   .axi_araddr             (M_AXI_ARADDR           ),
   .axi_aruser_ap          (M_AXI_ARUSER           ),
   .axi_aruser_id          (M_AXI_ARID             ),
   .axi_arlen              (M_AXI_BRUST_LEN        ),
   .axi_arready            (M_AXI_ARREADY          ),
   .axi_arvalid            (M_AXI_ARVALID          ),
    //������
   .axi_rdata              (M_AXI_RDATA             ),
   .axi_rid                (M_AXI_RID            ),
   .axi_rlast              (M_AXI_RLAST            ),
   .axi_rvalid             (M_AXI_RVALID           ),

   .apb_clk                (1'b0                   ),
   .apb_rst_n              (1'b1                   ),
   .apb_sel                (1'b0                   ),
   .apb_enable             (1'b0                   ),
   .apb_addr               (8'b0                   ),
   .apb_write              (1'b0                   ),
   .apb_ready              (                       ),
   .apb_wdata              (16'b0                  ),
   .apb_rdata              (                       ),
   .apb_int                (                       ),
   .debug_data             (                       ),
   .debug_slice_state      (                       ),
   .debug_calib_ctrl       (                       ),
   .ck_dly_set_bin         (                       ),
   .force_ck_dly_en        (1'b0                   ),
   .force_ck_dly_set_bin   (8'h05                  ),
   .dll_step               (                       ),
   .dll_lock               (                       ),
   .init_read_clk_ctrl     (2'b0                   ),                                                       
   .init_slip_step         (4'b0                   ), 
   .force_read_clk_ctrl    (1'b0                   ),  
   .ddrphy_gate_update_en  (1'b0                   ),
   .update_com_val_err_flag(                       ),
   .rd_fake_stop           (1'b0                   ),
    //�ڴ�ӿ�
   .mem_rst_n              (mem_rst_n              ),
   .mem_ck                 (mem_ck                 ),
   .mem_ck_n               (mem_ck_n               ),
   .mem_cke                (mem_cke                ),
   .mem_cs_n               (mem_cs_n               ),
   .mem_ras_n              (mem_ras_n              ),
   .mem_cas_n              (mem_cas_n              ),
   .mem_we_n               (mem_we_n               ),
   .mem_odt                (mem_odt                ),
   .mem_a                  (mem_a                  ),
   .mem_ba                 (mem_ba                 ), 
   .mem_dqs                (mem_dqs                ),
   .mem_dqs_n              (mem_dqs_n              ),
   .mem_dq                 (mem_dq                 ),
   .mem_dm                 (mem_dm                 )
  );


endmodule
