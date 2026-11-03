`timescale 1ns/1ps
module basic70_protocol_tb;
    reg clk=0;always #5 clk=~clk;
    reg rst_n=0,rx_done=0;
    reg [7:0] rx_data=0;
    wire done,parse_done,status_valid,extended,tx_en;
    wire [7:0] parse_cmd,parse_result,tx_data,valid_flags,objects,risk,ctrl,epoch,age;
    wire [1:0] camera,movement,boundary,accepted_camera,decision;
    wire [15:0] frame,accepted_frame;
    wire [2:0] scene,light_state,nav,arm_action,action,alarm_code,accepted_scene,reject_reason;
    wire [7:0] accepted_epoch;
    wire stop,alarm,decision_valid,timeout,seen,accepted;
    reg busy=0;
    integer busy_count=0,n=0,i,checks=0;
    reg [7:0] returned[0:31],payload[0:15];
    reg [7:0] checksum;
    packet_decode decoder(.clk(clk),.rst_n(rst_n),.uart_rx_data(rx_data),.uart_rx_done(rx_done),
        .packet_tx_done(done),.parse_done(parse_done),.parse_cmd(parse_cmd),.parse_result(parse_result),
        .status_valid(status_valid),.status_camera_id(camera),.status_frame_id(frame),
        .status_valid_flags(valid_flags),.status_object_flags(objects),.status_light_state(light_state),
        .status_navigation(nav),.status_control_flags(ctrl),.status_arm_action(arm_action),
        .status_extended(extended),.status_scene(scene),.status_risk_flags(risk),
        .status_movement(movement),.status_boundary(boundary),.status_epoch(epoch),.status_age_units(age));
    adas_basic70_control #(.TIMEOUT_CYCLES(1000),.AGE_QUANTUM_CYCLES(10)) control(
        .clk(clk),.rst_n(rst_n),.status_valid(status_valid),.status_extended(extended),
        .camera_id(camera),.frame_id(frame),.valid_flags(valid_flags),.object_flags(objects),.risk_flags(risk),
        .control_flags(ctrl),.light_state(light_state),.navigation(nav),.arm_action(arm_action),.scene(scene),
        .movement(movement),.boundary(boundary),.epoch(epoch),.age_units(age),
        .stop_enable(stop),.alarm_enable(alarm),.decision_valid(decision_valid),.link_timeout(timeout),
        .seen_extended(seen),.action_code(action),.alarm_code(alarm_code),.decision_code(decision),
        .accepted_camera(accepted_camera),.accepted_frame(accepted_frame),.accepted_epoch(accepted_epoch),
        .accepted_scene(accepted_scene),.reject_reason(reject_reason),.accepted_pulse(accepted));
    packet_code encoder(.clk(clk),.rst_n(rst_n),.key(2'b11),.uart_tx_busy(busy),
        .parse_done(parse_done),.parse_result(parse_result),.parse_cmd(parse_cmd),
        .accepted_camera(accepted_camera),.accepted_frame(accepted_frame),.accepted_epoch(accepted_epoch),
        .accepted_scene(accepted_scene),.action_code(action),.alarm_code(alarm_code),.reject_reason(reject_reason),
        .decision_valid(decision_valid),.link_timeout(timeout),.seen_extended(seen),
        .packet_tx_done(done),.uart_tx_en(tx_en),.uart_tx_data(tx_data));
    // A byte transmitter model with the same en/busy handshake as uart_tx.
    always @(posedge clk)begin
        if(!rst_n)begin busy<=0;busy_count<=0;end
        else if(tx_en && !busy)begin returned[n]=tx_data;n=n+1;busy<=1;busy_count<=5;end
        else if(busy)begin
            if(busy_count==0)busy<=0;
            else busy_count<=busy_count-1;
        end
    end
    task send_byte(input [7:0] value);begin
        @(negedge clk);rx_data=value;rx_done=1;
        @(negedge clk);rx_done=0;
        repeat(2)@(negedge clk);
    end endtask
    task wait_reply;integer limit;begin
        limit=0;
        while(!done && limit<5000)begin @(negedge clk);limit=limit+1;end
        if(limit>=5000)$fatal(1,"Reply timeout cmd=%h",parse_cmd);
        repeat(4)@(negedge clk);
        checksum=0;
        for(i=0;i<n-1;i=i+1)checksum=checksum+returned[i];
        if(n<3 || returned[0]!=8'h55 || returned[n-1]!=checksum)$fatal(1,"Reply framing/checksum failed");
    end endtask
    task query;begin
        n=0;send_byte(8'h55);send_byte(8'h04);send_byte(8'h59);wait_reply;
        if(n!=16 || returned[1]!=8'h84 || returned[2]!=12 || returned[3]!=3 || returned[13]!=8'h29)
            $fatal(1,"Version/action reply failed n=%0d",n);
    end endtask
    task context_packet(input [15:0] fid,input [7:0] ep,input [7:0] sc,input [7:0] vf,
        input [7:0] obj,input [7:0] risks,input [7:0] lamp,input [7:0] navigation,
        input [7:0] driver,input [7:0] move,input [7:0] line,input [7:0] age5,input [7:0] arm);
        begin
            payload[0]=3;payload[1]=0;payload[2]=fid[7:0];payload[3]=fid[15:8];
            payload[4]=sc;payload[5]=vf;payload[6]=obj;payload[7]=risks;
            payload[8]=lamp;payload[9]=navigation;payload[10]=driver;payload[11]=move;
            payload[12]=line;payload[13]=ep;payload[14]=age5;payload[15]=arm;
            n=0;checksum=8'h55+8'h13+8'd16;
            send_byte(8'h55);send_byte(8'h13);send_byte(16);
            for(i=0;i<16;i=i+1)begin send_byte(payload[i]);checksum=checksum+payload[i];end
            send_byte(checksum);wait_reply;
            if(n!=3 || returned[1]!=8'h13)$fatal(1,"Context ACK failed code=%h",returned[1]);
            query;
        end
    endtask
    task expect_action(input [2:0] wanted,input [2:0] cause,input is_valid);begin
        if(returned[9]!==wanted || returned[10]!==cause || returned[11][0]!==is_valid)
            $fatal(1,"Action mismatch got action=%0d cause=%0d valid=%0d wanted=%0d/%0d/%0d",returned[9],returned[10],returned[11][0],wanted,cause,is_valid);
        checks=checks+1;
    end endtask
    initial begin
        repeat(4)@(negedge clk);rst_n=1;repeat(4)@(negedge clk);
        query;expect_action(1,5,0);
        context_packet(1,1,4,8'h61,0,0,0,0,1,2,0,0,2);expect_action(2,0,1);
        if(returned[5]!=1 || returned[7]!=1)$fatal(1,"Identity echo failed");
        context_packet(2,1,4,8'h61,2,0,0,0,1,2,0,0,2);expect_action(2,0,1); // roadside person
        context_packet(3,1,4,8'h61,2,1,0,0,1,2,0,0,2);expect_action(1,1,1);
        context_packet(4,1,4,8'h60,0,0,0,0,1,2,0,0,2);expect_action(1,5,0); // inference invalid
        context_packet(5,1,1,8'h65,0,0,0,2,1,2,0,0,2);expect_action(1,5,0); // nav must not bypass unknown lamp
        context_packet(6,1,1,8'h63,0,0,1,0,3,2,0,0,2);expect_action(1,2,1); // forced red
        context_packet(7,1,1,8'h63,0,0,3,0,1,2,0,0,2);expect_action(2,0,1);
        context_packet(8,1,1,8'h63,0,0,3,0,1,1,0,0,2);expect_action(3,0,1);
        context_packet(9,1,2,8'h69,0,0,0,0,5,2,1,0,2);expect_action(5,0,1);
        context_packet(10,1,2,8'h69,0,0,0,0,21,2,2,0,2);expect_action(2,3,1);
        context_packet(11,1,4,8'h61,0,0,0,0,1,2,0,100,2);expect_action(1,6,0);
        if(returned[12]!=2 || returned[5]!=10)$fatal(1,"Expired packet was accepted");
        context_packet(10,1,4,8'h61,0,0,0,0,1,2,0,0,2);expect_action(1,7,0);
        if(returned[12]!=3)$fatal(1,"Replay packet was accepted");
        context_packet(16'hffff,2,4,8'h61,0,0,0,0,1,2,0,0,2);expect_action(2,0,1);
        context_packet(0,2,4,8'h61,0,0,0,0,1,2,0,0,2);expect_action(2,0,1);
        repeat(1100)@(negedge clk);query;expect_action(1,4,0);
        if(!returned[11][1])$fatal(1,"Timeout flag not set");
        $display("PASS basic70 protocol and controller checks=%0d",checks);
        $finish;
    end
    initial begin #1000000;$fatal(1,"Simulation watchdog");end
endmodule
