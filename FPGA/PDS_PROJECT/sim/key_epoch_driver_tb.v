`timescale 1ns/1ps
module key_epoch_driver_tb;
 reg clk=0;always #5 clk=~clk;
 reg rst_n=0,rx_done=0,tx_done=0;reg [7:0] rx_data=0;reg [5:0] keys=6'b111111;
 wire remote_valid,status_valid,extended,parse_done,core_stop,core_valid,gated,effective_stop,effective_valid;
 wire [7:0] remote_state,key_state,key_seq,vf,obj,risk,ctrl,ep,age,parse_result;
 wire [1:0] camera,movement,boundary,scene_mode,direction;
 wire [15:0] fid;
 wire [2:0] scene,light,nav,arm,core_action,alarm,effective_action;
 wire drive,force_request,active;wire [5:0] pulses;wire [7:0] accepted_epoch;
 integer checks=0,i;reg [7:0] payload[0:15];reg [7:0] checksum;
 packet_decode decoder(.clk(clk),.rst_n(rst_n),.uart_rx_done(rx_done),.uart_rx_data(rx_data),
  .packet_tx_done(tx_done),.parse_done(parse_done),.parse_result(parse_result),
  .driver_test_valid(remote_valid),.driver_test_state(remote_state),
  .status_valid(status_valid),.status_extended(extended),.status_camera_id(camera),.status_frame_id(fid),
  .status_valid_flags(vf),.status_object_flags(obj),.status_risk_flags(risk),.status_control_flags(ctrl),
  .status_light_state(light),.status_navigation(nav),.status_arm_action(arm),.status_scene(scene),
  .status_movement(movement),.status_boundary(boundary),.status_epoch(ep),.status_age_units(age));
 adas_key_control #(.DEBOUNCE_COUNT(4)) key_control(.clk(clk),.rst_n(rst_n),.key_n(keys),
  .remote_valid(remote_valid),.remote_state(remote_state),.scene_mode(scene_mode),.direction_request(direction),
  .drive_request(drive),.force_request(force_request),.control_active(active),.key_sequence(key_seq),
  .key_press_pulse(pulses),.key_state_byte(key_state));
 adas_key_epoch_gate gate(.drive_request(drive),.status_valid(status_valid),.controller_stop(core_stop),
  .controller_valid(core_valid),.key_sequence(key_seq),.status_epoch(ep),.accepted_epoch(accepted_epoch),
  .controller_action(core_action),.filtered_status_valid(gated),.effective_stop(effective_stop),
  .effective_valid(effective_valid),.effective_action(effective_action));
 adas_basic70_control #(.TIMEOUT_CYCLES(20000),.AGE_QUANTUM_CYCLES(10)) control(
  .clk(clk),.rst_n(rst_n),.status_valid(gated),.status_extended(extended),.camera_id(camera),.frame_id(fid),
  .valid_flags(vf),.object_flags(obj),.risk_flags(risk),.control_flags(ctrl),.light_state(light),
  .navigation(nav),.arm_action(arm),.scene(scene),.movement(movement),.boundary(boundary),.epoch(ep),
  .age_units(age),.stop_enable(core_stop),.decision_valid(core_valid),.action_code(core_action),
  .alarm_code(alarm),.accepted_epoch(accepted_epoch));
 task send(input [7:0] value);begin
  @(negedge clk);rx_data=value;rx_done=1;@(negedge clk);rx_done=0;repeat(2)@(negedge clk);
 end endtask
 task done;begin repeat(5)@(negedge clk);tx_done=1;@(negedge clk);tx_done=0;repeat(4)@(negedge clk);end endtask
 task remote(input [7:0] state,input corrupt);begin
  send(8'h55);send(8'h14);send(2);send(8'h54);send(state);
  checksum=8'h55+8'h14+8'd2+8'h54+state;send(corrupt ? checksum+1 : checksum);done;
 end endtask
 task send_context(input [15:0] frame,input [7:0] epoch,input [7:0] driver,input [7:0] line);begin
  payload[0]=3;payload[1]=0;payload[2]=frame[7:0];payload[3]=frame[15:8];payload[4]=2;payload[5]=8'h69;
  payload[6]=0;payload[7]=0;payload[8]=0;payload[9]=0;payload[10]=driver;payload[11]=2;
  payload[12]=line;payload[13]=epoch;payload[14]=0;payload[15]=2;
  checksum=8'h55+8'h13+8'd16;send(8'h55);send(8'h13);send(16);
  for(i=0;i<16;i=i+1)begin send(payload[i]);checksum=checksum+payload[i];end
  send(checksum);done;
 end endtask
 task send_signal(input [15:0] frame,input [7:0] epoch,input [7:0] color,input [7:0] driver,input [7:0] wanted_arm);begin
  payload[0]=3;payload[1]=0;payload[2]=frame[7:0];payload[3]=frame[15:8];payload[4]=1;payload[5]=8'h63;
  payload[6]=0;payload[7]=0;payload[8]=color;payload[9]=0;payload[10]=driver;payload[11]=2;
  payload[12]=0;payload[13]=epoch;payload[14]=0;payload[15]=wanted_arm;
  checksum=8'h55+8'h13+8'd16;send(8'h55);send(8'h13);send(16);
  for(i=0;i<16;i=i+1)begin send(payload[i]);checksum=checksum+payload[i];end
  send(checksum);done;
 end endtask
 task expect_action(input [2:0] wanted);begin
  if(effective_action!==wanted)$fatal(1,"epoch gate action got %0d wanted %0d",effective_action,wanted);
  checks=checks+1;
 end endtask
 initial begin
  repeat(4)@(negedge clk);rst_n=1;repeat(4)@(negedge clk);expect_action(1);
  remote(8'h15,0);if(key_state!==8'h55 || key_seq!==1)$fatal(1,"remote intent not labelled/accepted");
  send_context(1,0,5,1);expect_action(1);
  send_context(2,1,5,1);expect_action(5);
  remote(8'h16,0);expect_action(1);
  send_context(3,1,13,1);expect_action(1);
  send_context(4,2,13,1);expect_action(6);
  remote(8'h36,0);expect_action(1);
  send_context(5,3,31,2);expect_action(2);
  if(alarm!==3)$fatal(1,"forced solid alarm missing");
  keys[4]=0;repeat(30)@(negedge clk);expect_action(1);
  if(drive!==0 || key_state[6]!==0 || key_seq!==4)$fatal(1,"physical STOP must override remote mode once");
  keys[4]=1;repeat(10)@(negedge clk);
  remote(8'hff,0);if(key_state!==4 || key_seq!==5)$fatal(1,"test exit must restore physical STOP");
  remote(8'h13,0);if(key_seq!==5)$fatal(1,"invalid scene changed control state");
  remote(8'h15,1);if(key_seq!==5)$fatal(1,"bad checksum changed control state");
  remote(8'h14,0);send_context(6,6,1,0);expect_action(2);
  remote(8'h18,0);if(key_seq!==6)$fatal(1,"unsupported navigation driver test accepted");
  remote(8'h10,0);if(key_state!==8'h50 || key_seq!==7)$fatal(1,"signal intent not labelled");expect_action(1);
  send_signal(7,7,1,1,1);expect_action(1);if(alarm!==0)$fatal(1,"ordinary red alarm");
  remote(8'h30,0);expect_action(1);send_signal(8,8,1,3,1);expect_action(1);if(alarm!==2)$fatal(1,"forced red alarm missing");
  remote(8'h10,0);send_signal(9,9,3,1,2);expect_action(2);if(alarm!==0)$fatal(1,"green alarm");
  remote(8'h14,0);expect_action(1);
  remote(8'hff,0);if(key_state!==4 || drive!==0)$fatal(1,"exit not parked");
  $display("PASS key epoch and remote driver integration checks=%0d",checks);$finish;
 end
 initial begin #1000000;$fatal(1,"watchdog");end
endmodule
