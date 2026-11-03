`timescale 1ns/1ps
// Scene-specific simulation controller. Invalid/unknown data never permits GO.
module adas_basic70_control #(
    parameter [31:0] TIMEOUT_CYCLES=32'd25000000,
    parameter [31:0] AGE_QUANTUM_CYCLES=32'd250000
)(
    input wire clk,rst_n,status_valid,status_extended,
    input wire [1:0] camera_id,
    input wire [15:0] frame_id,
    input wire [7:0] valid_flags,object_flags,risk_flags,control_flags,
    input wire [2:0] light_state,navigation,arm_action,scene,
    input wire [1:0] movement,boundary,
    input wire [7:0] epoch,age_units,
    output reg stop_enable,alarm_enable,decision_valid,link_timeout,seen_extended,accepted_pulse,
    output reg [2:0] action_code,alarm_code,
    output reg [1:0] decision_code,accepted_camera,
    output reg [15:0] accepted_frame,
    output reg [7:0] accepted_epoch,
    output reg [2:0] accepted_scene,reject_reason
);
    localparam [2:0] STOP=1,STRAIGHT=2,LEFT=3,RIGHT=4,CHANGE_LEFT=5,CHANGE_RIGHT=6;
    localparam [2:0] NONE=0,COLLISION=1,RED_VIOLATION=2,SOLID_CHANGE=3,
                     TIMEOUT=4,INVALID=5,EXPIRED=6,REPLAY=7;
    reg [31:0] timeout_counter;
    reg seen_frame[0:3];
    reg [7:0] last_epoch[0:3];
    reg [15:0] last_frame[0:3];
    wire [7:0] epoch_delta=epoch-last_epoch[camera_id];
    wire [15:0] frame_delta=frame_id-last_frame[camera_id];
    wire frame_fresh=(!seen_frame[camera_id]) ||
        ((epoch_delta!=0) && !epoch_delta[7]) ||
        ((epoch_delta==0) && (frame_delta!=0) && !frame_delta[15]);
    wire center_risk=risk_flags[0] || risk_flags[3];
    wire side_risk=control_flags[2] && (control_flags[3] ? risk_flags[2] : risk_flags[1]);
    wire go_request=valid_flags[5] && control_flags[0];
    wire moving=valid_flags[4] && control_flags[5];
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            stop_enable<=1;alarm_enable<=0;action_code<=STOP;alarm_code<=INVALID;
            decision_code<=0;decision_valid<=0;link_timeout<=0;seen_extended<=0;
            timeout_counter<=0;accepted_camera<=0;accepted_frame<=0;accepted_epoch<=0;
            accepted_scene<=0;reject_reason<=0;
            accepted_pulse<=0;
            for(i=0;i<4;i=i+1)begin seen_frame[i]<=0;last_epoch[i]<=0;last_frame[i]<=0;end
        end else begin
            accepted_pulse<=0;
            if(seen_extended && !link_timeout)begin
                if(timeout_counter>=TIMEOUT_CYCLES-1)begin
                    link_timeout<=1;stop_enable<=1;alarm_enable<=1;action_code<=STOP;
                    alarm_code<=TIMEOUT;decision_valid<=0;
                end else timeout_counter<=timeout_counter+1'b1;
            end
            if(status_valid)begin
                // Rejection stops immediately, without renewing the result lease.
                stop_enable<=1;alarm_enable<=0;action_code<=STOP;decision_code<=0;
                decision_valid<=0;
                if(!status_extended)begin
                    alarm_code<=INVALID;reject_reason<=3'd1;
                end else if(age_units>=8'd100)begin
                    seen_extended<=1;alarm_code<=EXPIRED;reject_reason<=3'd2;
                end else if(!frame_fresh)begin
                    seen_extended<=1;alarm_code<=REPLAY;reject_reason<=3'd3;
                end else begin
                    accepted_pulse<=1;
                    seen_extended<=1;seen_frame[camera_id]<=1;
                    last_epoch[camera_id]<=epoch;last_frame[camera_id]<=frame_id;
                    accepted_camera<=camera_id;accepted_frame<=frame_id;
                    accepted_epoch<=epoch;accepted_scene<=scene;reject_reason<=0;
                    timeout_counter<=age_units*AGE_QUANTUM_CYCLES;link_timeout<=0;
                    alarm_code<=NONE;
                    if(camera_id!=0 || !valid_flags[0])begin
                        alarm_code<=INVALID;
                    end else if(!valid_flags[6])begin
                        alarm_code<=INVALID;
                    end else if(center_risk)begin
                        alarm_enable<=1;alarm_code<=COLLISION;decision_valid<=1;
                        decision_code<=object_flags[1] ? 2'd3 : 2'd1;
                    end else if(valid_flags[1] && light_state==1)begin
                        decision_valid<=1;decision_code<=2'd2;
                        if(moving || (valid_flags[5] && control_flags[1]))begin
                            alarm_enable<=1;alarm_code<=RED_VIOLATION;
                        end
                    end else if(valid_flags[1] && light_state!=3)begin
                        alarm_code<=INVALID;
                    end else if(!go_request || arm_action==1)begin
                        // A stopped driver request may always restrict movement.
                        decision_valid<=valid_flags[5];
                    end else if(arm_action!=2)begin
                        alarm_code<=INVALID;
                    end else if(scene==1)begin
                        if(valid_flags[1] && light_state==3 && movement!=0)begin
                            stop_enable<=0;decision_valid<=1;
                            action_code<=movement==1 ? LEFT : movement==3 ? RIGHT : STRAIGHT;
                        end else alarm_code<=INVALID;
                    end else if(scene==2)begin
                        if(!control_flags[2])begin
                            stop_enable<=0;decision_valid<=1;action_code<=STRAIGHT;
                        end else if(!valid_flags[3] || boundary==0)alarm_code<=INVALID;
                        else if(control_flags[2] && side_risk)begin
                            stop_enable<=0;decision_valid<=1;action_code<=STRAIGHT;
                            alarm_enable<=1;alarm_code<=COLLISION;
                        end else if(control_flags[2] && boundary==2)begin
                            stop_enable<=0;decision_valid<=1;action_code<=STRAIGHT;
                            if(control_flags[4])begin alarm_enable<=1;alarm_code<=SOLID_CHANGE;end
                        end else begin
                            stop_enable<=0;decision_valid<=1;
                            action_code<=control_flags[2] ? (control_flags[3] ? CHANGE_RIGHT : CHANGE_LEFT) : STRAIGHT;
                        end
                    end else if(scene==3)begin
                        if(!valid_flags[2] || navigation==0 || (object_flags[5] && !valid_flags[1]))
                            alarm_code<=INVALID;
                        else begin
                            stop_enable<=0;decision_valid<=1;
                            action_code<=navigation==1 ? LEFT : navigation==3 ? RIGHT : STRAIGHT;
                        end
                    end else if(scene==4)begin
                        // Explicit hazard-only simulation scene, not a green-light claim.
                        stop_enable<=0;decision_valid<=1;action_code<=STRAIGHT;
                    end else alarm_code<=INVALID;
                end
            end
        end
    end
endmodule
