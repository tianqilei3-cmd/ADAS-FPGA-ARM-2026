`timescale 1ns/1ps
//=============================================================================
// Project : FPGA ADAS Road Perception System
// Module  : top_uart_packet
// Version : 3.0
// Function: UART物理层、V1/V2协议解析及原滤镜命令封装。
//=============================================================================
module top_uart_packet #(
    parameter [7:0] PACKET_HEAD=8'h55,
    parameter [7:0] LED_CMD=8'h00,
    parameter [7:0] BEEP_CMD=8'h01,
    parameter [7:0] BREATH_LED_CMD=8'h02,
    parameter [7:0] KEY_CMD=8'h03,
    parameter [7:0] ADAS_TARGET_CMD=8'h10,
    parameter [7:0] ADAS_STATUS_CMD=8'h11,
    parameter [7:0] ADAS_BBOX_CMD=8'h12,
    parameter [7:0] PARSE_OK=8'h00
)(
    input wire sys_clk,
    input wire sys_rst_n,
    input wire uart_rxd,
    input wire [7:0] key,
    input wire [7:0] key_sequence,
    input wire [1:0] feedback_camera,
    input wire [15:0] feedback_frame,
    input wire [7:0] feedback_epoch,
    input wire [2:0] feedback_scene,feedback_action,feedback_alarm,feedback_reject,
    input wire feedback_valid,feedback_timeout,feedback_seen,

    output wire [2:0] led,
    output wire uart_txd,
    output wire [2:0] target_type,
    output wire target_valid,
    output wire status_valid,
    output wire [1:0] status_camera_id,
    output wire [15:0] status_frame_id,
    output wire [7:0] status_valid_flags,
    output wire [7:0] status_object_flags,
    output wire [2:0] status_light_state,
    output wire [2:0] status_navigation,
    output wire [7:0] status_control_flags,
    output wire [2:0] status_arm_action,
    output wire status_extended,
    output wire [2:0] status_scene,
    output wire [7:0] status_risk_flags,status_epoch,status_age_units,
    output wire [1:0] status_movement,status_boundary,
    output wire bbox_valid,
    output wire [1:0] bbox_camera_id,
    output wire [15:0] bbox_frame_id,
    output wire [1:0] bbox_slot,
    output wire bbox_model_id,
    output wire [7:0] bbox_class_id,
    output wire [7:0] bbox_confidence,
    output wire [15:0] bbox_x1,
    output wire [15:0] bbox_y1,
    output wire [15:0] bbox_x2,
    output wire [15:0] bbox_y2,
    output wire driver_test_valid,
    output wire [7:0] driver_test_state,
    output wire uart_frame_error,
    output wire [7:0] uart_error_code
);
    localparam [7:0] ERR_HEAD=8'hE0, ERR_CMD=8'hE1, ERR_DATA_LEN=8'hE2;
    localparam [7:0] ERR_CHECKSUM=8'hE3, ERR_TARGET_TYPE=8'hE4;
    localparam [7:0] ERR_TIMEOUT=8'hE5, ERR_PROTOCOL=8'hE6;
    wire [7:0] uart_rx_data, parse_result, parse_cmd, uart_tx_data;
    wire uart_rx_done, parse_done, uart_tx_en, uart_tx_busy, packet_tx_done;
    wire [2:0] led_data;
    assign uart_error_code=parse_result;

    uart_rx u_uart_rx(.clk(sys_clk),.rst_n(sys_rst_n),.uart_rxd(uart_rxd),
        .uart_rx_done(uart_rx_done),.uart_rx_data(uart_rx_data));

    packet_decode #(
        .PACKET_HEAD(PACKET_HEAD),.LED_CMD(LED_CMD),.BEEP_CMD(BEEP_CMD),
        .BREATH_LED_CMD(BREATH_LED_CMD),.KEY_CMD(KEY_CMD),
        .ADAS_TARGET_CMD(ADAS_TARGET_CMD),.ADAS_STATUS_CMD(ADAS_STATUS_CMD),
        .ADAS_BBOX_CMD(ADAS_BBOX_CMD),.ERR_HEAD(ERR_HEAD),.ERR_CMD(ERR_CMD),
        .ERR_DATA_LEN(ERR_DATA_LEN),.ERR_CHECKSUM(ERR_CHECKSUM),
        .ERR_TARGET_TYPE(ERR_TARGET_TYPE),.ERR_TIMEOUT(ERR_TIMEOUT),
        .ERR_PROTOCOL(ERR_PROTOCOL),.PARSE_OK(PARSE_OK)
    ) u_packet_decode(
        .clk(sys_clk),.rst_n(sys_rst_n),.uart_rx_data(uart_rx_data),
        .uart_rx_done(uart_rx_done),.packet_tx_done(packet_tx_done),
        .led_data(led_data),.beep_data(),.breath_sw(),.breath_fre(),
        .parse_done(parse_done),.parse_result(parse_result),.parse_cmd(parse_cmd),
        .target_type(target_type),.target_valid(target_valid),
        .status_valid(status_valid),.status_camera_id(status_camera_id),
        .status_frame_id(status_frame_id),.status_valid_flags(status_valid_flags),
        .status_object_flags(status_object_flags),.status_light_state(status_light_state),
        .status_navigation(status_navigation),.status_control_flags(status_control_flags),
        .status_arm_action(status_arm_action),.bbox_valid(bbox_valid),
        .bbox_camera_id(bbox_camera_id),.bbox_frame_id(bbox_frame_id),
        .bbox_slot(bbox_slot),.bbox_model_id(bbox_model_id),
        .bbox_class_id(bbox_class_id),.bbox_confidence(bbox_confidence),
        .bbox_x1(bbox_x1),.bbox_y1(bbox_y1),.bbox_x2(bbox_x2),.bbox_y2(bbox_y2),
        .status_extended(status_extended),.status_scene(status_scene),
        .status_risk_flags(status_risk_flags),.status_epoch(status_epoch),.status_age_units(status_age_units),
        .status_movement(status_movement),.status_boundary(status_boundary),
        .driver_test_valid(driver_test_valid),.driver_test_state(driver_test_state),
        .frame_error(uart_frame_error));

    cmd_exe u_cmd_exe(.clk(sys_clk),.rst_n(sys_rst_n),.led_data(led_data),.led(led));
    packet_code #(.PACKET_HEAD(PACKET_HEAD),.KEY_CMD(KEY_CMD),.PARSE_OK(PARSE_OK))
    u_packet_code(.clk(sys_clk),.rst_n(sys_rst_n),.key(key),.key_sequence(key_sequence),
        .uart_tx_busy(uart_tx_busy),.parse_done(parse_done),.parse_result(parse_result),
        .parse_cmd(parse_cmd),.accepted_camera(feedback_camera),.accepted_frame(feedback_frame),
        .accepted_epoch(feedback_epoch),.accepted_scene(feedback_scene),.action_code(feedback_action),
        .alarm_code(feedback_alarm),.reject_reason(feedback_reject),.decision_valid(feedback_valid),
        .link_timeout(feedback_timeout),.seen_extended(feedback_seen),
        .packet_tx_done(packet_tx_done),.uart_tx_en(uart_tx_en),
        .uart_tx_data(uart_tx_data));
    uart_tx u_uart_tx(.clk(sys_clk),.rst_n(sys_rst_n),.uart_tx_data(uart_tx_data),
        .uart_tx_en(uart_tx_en),.uart_tx_busy(uart_tx_busy),.uart_txd(uart_txd));
endmodule
