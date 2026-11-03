`timescale 1ns/1ps

//=============================================================================
// Project : FPGA ADAS Road Perception System
// Module  : packet_decode
// Version : 3.0
// Function: UART协议解析。兼容旧0x10目标位图，并解析ARM协作V2状态/检测框。
// 通用格式: 55 CMD LEN PAYLOAD CHECKSUM；校验为前述字节累加和低8位。
// CMD=11/LEN=0A: version,camera,frameL,frameH,valid,objects,light,
//                navigation,control,arm_action。
// CMD=12/LEN=10: version,camera,frameL,frameH,slot,model,class,confidence,
//                x1L,x1H,y1L,y1H,x2L,x2H,y2L,y2H。
//=============================================================================
module packet_decode #(
    parameter [7:0] PACKET_HEAD       = 8'h55,
    parameter [7:0] LED_CMD           = 8'h00,
    parameter [7:0] BEEP_CMD          = 8'h01,
    parameter [7:0] BREATH_LED_CMD    = 8'h02,
    parameter [7:0] KEY_CMD           = 8'h03,
    parameter [7:0] ADAS_TARGET_CMD   = 8'h10,
    parameter [7:0] ADAS_STATUS_CMD   = 8'h11,
    parameter [7:0] ADAS_BBOX_CMD     = 8'h12,
    parameter [7:0] PROTOCOL_VERSION  = 8'h02,
    parameter [7:0] ERR_HEAD          = 8'hE0,
    parameter [7:0] ERR_CMD           = 8'hE1,
    parameter [7:0] ERR_DATA_LEN      = 8'hE2,
    parameter [7:0] ERR_CHECKSUM      = 8'hE3,
    parameter [7:0] ERR_TARGET_TYPE   = 8'hE4,
    parameter [7:0] ERR_TIMEOUT       = 8'hE5,
    parameter [7:0] ERR_PROTOCOL      = 8'hE6,
    parameter [7:0] PARSE_OK          = 8'h00,
    parameter [31:0] TIMEOUT_CYCLES   = 32'd500000
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [7:0]  uart_rx_data,
    input  wire        uart_rx_done,
    input  wire        packet_tx_done,
    output reg  [2:0]  led_data,
    output reg         beep_data,
    output reg         breath_sw,
    output reg  [7:0]  breath_fre,
    output reg         parse_done,
    output reg  [7:0]  parse_result,
    output reg  [7:0]  parse_cmd,
    output reg  [2:0]  target_type,
    output reg         target_valid,
    output reg         status_valid,
    output reg  [1:0]  status_camera_id,
    output reg  [15:0] status_frame_id,
    output reg  [7:0]  status_valid_flags,
    output reg  [7:0]  status_object_flags,
    output reg  [2:0]  status_light_state,
    output reg  [2:0]  status_navigation,
    output reg  [7:0]  status_control_flags,
    output reg  [2:0]  status_arm_action,
    output reg         bbox_valid,
    output reg  [1:0]  bbox_camera_id,
    output reg  [15:0] bbox_frame_id,
    output reg  [1:0]  bbox_slot,
    output reg         bbox_model_id,
    output reg  [7:0]  bbox_class_id,
    output reg  [7:0]  bbox_confidence,
    output reg  [15:0] bbox_x1,
    output reg  [15:0] bbox_y1,
    output reg  [15:0] bbox_x2,
    output reg  [15:0] bbox_y2,
    output reg status_extended,
    output reg [2:0] status_scene,
    output reg [7:0] status_risk_flags,status_epoch,status_age_units,
    output reg [1:0] status_movement,status_boundary,
    output reg         driver_test_valid,
    output reg [7:0] driver_test_state,
    output reg         frame_error
);
    localparam [5:0] ST_HEAD=6'b000001, ST_CMD=6'b000010, ST_LEN=6'b000100;
    localparam [5:0] ST_DATA=6'b001000, ST_CHECK=6'b010000, ST_WAIT_TX=6'b100000;

    reg [5:0] current_state, next_state;
    reg [7:0] data_length, data_count;
    reg [7:0] payload [0:15];
    reg [7:0] checksum;
    reg data_invalid;
    reg [31:0] timeout_counter;
    integer i;

    wire timeout_hit = (current_state != ST_HEAD) &&
                       (current_state != ST_WAIT_TX) &&
                       (timeout_counter >= TIMEOUT_CYCLES - 1'b1);
    wire command_has_length = (uart_rx_data==LED_CMD) || (uart_rx_data==BEEP_CMD) ||
                              (uart_rx_data==BREATH_LED_CMD) ||
                              (uart_rx_data==ADAS_TARGET_CMD) ||
                              (uart_rx_data==ADAS_STATUS_CMD) ||
                              (uart_rx_data==ADAS_BBOX_CMD) || (uart_rx_data==8'h13) || (uart_rx_data==8'h14);
    wire command_is_supported = command_has_length || (uart_rx_data==KEY_CMD) || (uart_rx_data==8'h04);
    wire length_is_valid = ((parse_cmd==LED_CMD)&&(uart_rx_data==8'd1)) ||
                           ((parse_cmd==BEEP_CMD)&&(uart_rx_data==8'd1)) ||
                           ((parse_cmd==BREATH_LED_CMD)&&(uart_rx_data==8'd2)) ||
                           ((parse_cmd==ADAS_TARGET_CMD)&&(uart_rx_data==8'd1)) ||
                           ((parse_cmd==ADAS_STATUS_CMD)&&(uart_rx_data==8'd10)) ||
                           ((parse_cmd==ADAS_BBOX_CMD)&&(uart_rx_data==8'd16)) || ((parse_cmd==8'h13)&&(uart_rx_data==8'd16)) || ((parse_cmd==8'h14)&&(uart_rx_data==8'd2));

    always @(posedge clk) begin
        if (!rst_n) current_state <= ST_HEAD;
        else current_state <= next_state;
    end

    always @(*) begin
        next_state = current_state;
        if (timeout_hit) next_state = ST_WAIT_TX;
        else case (current_state)
            ST_HEAD: if (uart_rx_done && uart_rx_data==PACKET_HEAD) next_state=ST_CMD;
            ST_CMD: if (uart_rx_done) begin
                if (uart_rx_data==KEY_CMD || uart_rx_data==8'h04) next_state=ST_CHECK;
                else if (command_has_length) next_state=ST_LEN;
                else next_state=ST_WAIT_TX;
            end
            ST_LEN: if (uart_rx_done) next_state=length_is_valid ? ST_DATA : ST_WAIT_TX;
            ST_DATA: if (uart_rx_done && data_count==data_length-1'b1) next_state=ST_CHECK;
            ST_CHECK: if (uart_rx_done) next_state=ST_WAIT_TX;
            ST_WAIT_TX: if (packet_tx_done) next_state=ST_HEAD;
            default: next_state=ST_HEAD;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) timeout_counter <= 32'd0;
        else if (current_state==ST_HEAD || current_state==ST_WAIT_TX || uart_rx_done)
            timeout_counter <= 32'd0;
        else if (!timeout_hit) timeout_counter <= timeout_counter + 1'b1;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            data_length<=0; data_count<=0; checksum<=0; data_invalid<=0;
            led_data<=0; beep_data<=0; breath_sw<=0; breath_fre<=0;
            parse_done<=0; parse_result<=PARSE_OK; parse_cmd<=0;
            target_type<=0; target_valid<=0;driver_test_valid<=0;driver_test_state<=0;
            status_valid<=0; status_camera_id<=0; status_frame_id<=0;
            status_valid_flags<=0; status_object_flags<=0; status_light_state<=0;
            status_navigation<=0; status_control_flags<=0; status_arm_action<=0;
            status_extended<=0;status_scene<=0;status_risk_flags<=0;status_epoch<=0;
            status_age_units<=0;status_movement<=0;status_boundary<=0;
            bbox_valid<=0; bbox_camera_id<=0; bbox_frame_id<=0; bbox_slot<=0;
            bbox_model_id<=0; bbox_class_id<=0; bbox_confidence<=0;
            bbox_x1<=0; bbox_y1<=0; bbox_x2<=0; bbox_y2<=0; frame_error<=0;
            for (i=0;i<16;i=i+1) payload[i]<=0;
        end else begin
            parse_done<=0; target_valid<=0; status_valid<=0; bbox_valid<=0; frame_error<=0;driver_test_valid<=0;
            if (timeout_hit) begin
                parse_result<=ERR_TIMEOUT; parse_done<=1; frame_error<=1;
            end else case (current_state)
                ST_HEAD: if (uart_rx_done && uart_rx_data==PACKET_HEAD) begin
                    checksum<=PACKET_HEAD; parse_result<=PARSE_OK; parse_cmd<=0;
                    data_count<=0; data_invalid<=0;
                end
                ST_CMD: if (uart_rx_done) begin
                    if (command_is_supported) begin parse_cmd<=uart_rx_data; checksum<=checksum+uart_rx_data; end
                    else begin parse_result<=ERR_CMD; parse_done<=1; frame_error<=1; end
                end
                ST_LEN: if (uart_rx_done) begin
                    if (length_is_valid) begin data_length<=uart_rx_data; data_count<=0; checksum<=checksum+uart_rx_data; end
                    else begin parse_result<=ERR_DATA_LEN; parse_done<=1; frame_error<=1; end
                end
                ST_DATA: if (uart_rx_done) begin
                    payload[data_count[3:0]]<=uart_rx_data;
                    checksum<=checksum+uart_rx_data;
                    if (parse_cmd==ADAS_TARGET_CMD && data_count==0 && uart_rx_data[7:3]!=0)
                        data_invalid<=1;
                    if (data_count<data_length-1'b1) data_count<=data_count+1'b1;
                end
                ST_CHECK: if (uart_rx_done) begin
                    if (data_invalid) begin parse_result<=ERR_TARGET_TYPE; parse_done<=1; frame_error<=1; end
                    else if (uart_rx_data!=checksum) begin parse_result<=ERR_CHECKSUM; parse_done<=1; frame_error<=1; end
                    else if ((parse_cmd==ADAS_STATUS_CMD || parse_cmd==ADAS_BBOX_CMD) && payload[0]!=PROTOCOL_VERSION) begin
                        parse_result<=ERR_PROTOCOL; parse_done<=1; frame_error<=1;
                    end else if (parse_cmd==ADAS_STATUS_CMD &&
                        (payload[1][7:2]!=0 || payload[4][7:5]!=0 || payload[5][7:6]!=0 ||
                         payload[6]>4 || payload[7]>3 || payload[8][7:4]!=0 || payload[9]>3)) begin
                        parse_result<=ERR_PROTOCOL; parse_done<=1; frame_error<=1;
                    end else if (parse_cmd==8'h13 &&
                        (payload[0]!=3 || payload[1][7:2]!=0 || payload[4]>4 || payload[5][7]!=0 ||
                         payload[6][7:6]!=0 || payload[7][7:4]!=0 || payload[8]>4 || payload[9]>3 ||
                         payload[10][7:6]!=0 || payload[11]>3 || payload[12]>2 || payload[15]>3))begin
                        parse_result<=ERR_PROTOCOL;parse_done<=1;frame_error<=1;
                    end else if (parse_cmd==8'h14 &&
                        (payload[0]!=8'h54 || (payload[1]!=8'hff &&
                        ((payload[1]&8'hc0)!=8'h00 || payload[1][3:2]==2 || payload[1][1:0]==3 || (payload[1][5]&&!payload[1][4])))))begin
                        parse_result<=ERR_PROTOCOL;parse_done<=1;frame_error<=1;
                    end else if (parse_cmd==ADAS_BBOX_CMD &&
                        (payload[1][7:2]!=0 || payload[4]>3 || payload[5]>1 ||
                         {payload[9],payload[8]}>={payload[13],payload[12]} ||
                         {payload[11],payload[10]}>={payload[15],payload[14]})) begin
                        parse_result<=ERR_PROTOCOL; parse_done<=1; frame_error<=1;
                    end else begin
                        parse_result<=PARSE_OK; parse_done<=1;
                        if (parse_cmd==LED_CMD) led_data<=payload[0][2:0];
                        else if (parse_cmd==BEEP_CMD) beep_data<=payload[0][0];
                        else if (parse_cmd==BREATH_LED_CMD) begin breath_sw<=payload[0][0]; breath_fre<=payload[1]; end
                        else if (parse_cmd==ADAS_TARGET_CMD) begin target_type<=payload[0][2:0]; target_valid<=1; end
                        else if (parse_cmd==ADAS_STATUS_CMD) begin
                            status_camera_id<=payload[1][1:0]; status_frame_id<={payload[3],payload[2]};
                            status_valid_flags<=payload[4]; status_object_flags<=payload[5];
                            status_light_state<=payload[6][2:0]; status_navigation<=payload[7][2:0];
                            status_control_flags<=payload[8]; status_arm_action<=payload[9][2:0];
                            status_valid<=1;status_extended<=0;status_scene<=0;status_risk_flags<=0;
                            status_movement<=0;status_boundary<=0;status_epoch<=0;status_age_units<=0;
                            target_type<={payload[5][1],(payload[6]==8'd1),payload[5][3]};
                            target_valid<=1;
                        end else if(parse_cmd==8'h13)begin
                            status_camera_id<=payload[1][1:0];status_frame_id<={payload[3],payload[2]};
                            status_scene<=payload[4][2:0];status_valid_flags<=payload[5];
                            status_object_flags<=payload[6];status_risk_flags<=payload[7];
                            status_light_state<=payload[8][2:0];status_navigation<=payload[9][2:0];
                            status_control_flags<=payload[10];status_movement<=payload[11][1:0];
                            status_boundary<=payload[12][1:0];status_epoch<=payload[13];
                            status_age_units<=payload[14];status_arm_action<=payload[15][2:0];
                            status_extended<=1;status_valid<=1;
                        end else if (parse_cmd==8'h14)begin
                            driver_test_state<=payload[1];driver_test_valid<=1;
                        end else if (parse_cmd==ADAS_BBOX_CMD) begin
                            bbox_camera_id<=payload[1][1:0]; bbox_frame_id<={payload[3],payload[2]};
                            bbox_slot<=payload[4][1:0]; bbox_model_id<=payload[5][0];
                            bbox_class_id<=payload[6]; bbox_confidence<=payload[7];
                            bbox_x1<={payload[9],payload[8]}; bbox_y1<={payload[11],payload[10]};
                            bbox_x2<={payload[13],payload[12]}; bbox_y2<={payload[15],payload[14]};
                            bbox_valid<=1;
                        end
                    end
                end
                ST_WAIT_TX: if (packet_tx_done) begin
                    parse_result<=PARSE_OK; parse_cmd<=0; checksum<=0;
                    data_length<=0; data_count<=0; data_invalid<=0;
                end
                default: begin parse_result<=ERR_HEAD; frame_error<=1; end
            endcase
        end
    end
endmodule
