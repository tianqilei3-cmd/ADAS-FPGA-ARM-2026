`timescale 1ns/1ps
// Old ACKs preserved; 0x04 returns an identified action snapshot after settling.
module packet_code #(
    parameter [7:0] PACKET_HEAD=8'h55,KEY_CMD=8'h03,PARSE_OK=8'h00
)(
    input wire clk,rst_n,
    input wire [7:0] key,key_sequence,
    input wire uart_tx_busy,parse_done,
    input wire [7:0] parse_result,parse_cmd,
    input wire [1:0] accepted_camera,
    input wire [15:0] accepted_frame,
    input wire [7:0] accepted_epoch,
    input wire [2:0] accepted_scene,action_code,alarm_code,reject_reason,
    input wire decision_valid,link_timeout,seen_extended,
    output reg packet_tx_done,uart_tx_en,
    output reg [7:0] uart_tx_data
);
    reg [7:0] bytes[0:15];
    reg [4:0] count,length;
    reg [2:0] state;
    reg [1:0] settle;
    reg [7:0] command,result,checksum;
    integer i;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=0;count<=0;length<=0;settle<=0;command<=0;result<=0;
            packet_tx_done<=0;uart_tx_en<=0;uart_tx_data<=0;checksum<=0;
            for(i=0;i<16;i=i+1)bytes[i]<=0;
        end else begin
            packet_tx_done<=0;uart_tx_en<=0;
            case(state)
                0:if(parse_done)begin command<=parse_cmd;result<=parse_result;settle<=0;state<=1;end
                1:if(settle==2)begin
                    bytes[0]<=PACKET_HEAD;
                    if(result!=PARSE_OK)begin
                        bytes[1]<=result;bytes[2]<=PACKET_HEAD+result;length<=3;
                    end else if(command==KEY_CMD)begin
                        // 0x03: state snapshot + monotonically increasing key event sequence.
                        bytes[1]<=command;bytes[2]<=2;bytes[3]<=key;bytes[4]<=key_sequence;
                        bytes[5]<=PACKET_HEAD+command+8'd2+key+key_sequence;length<=6;
                    end else if(command==8'h04)begin
                        bytes[1]<=8'h84;bytes[2]<=12;bytes[3]<=3;
                        bytes[4]<={6'd0,accepted_camera};bytes[5]<=accepted_frame[7:0];bytes[6]<=accepted_frame[15:8];
                        bytes[7]<=accepted_epoch;bytes[8]<={5'd0,accepted_scene};
                        bytes[9]<={5'd0,action_code};bytes[10]<={5'd0,alarm_code};
                        bytes[11]<={5'd0,seen_extended,link_timeout,decision_valid};
                        bytes[12]<={5'd0,reject_reason};bytes[13]<=8'h29;bytes[14]<=8'h7f;
                        length<=16;
                    end else begin
                        bytes[1]<=command;bytes[2]<=PACKET_HEAD+command;length<=3;
                    end
                    count<=0;state<=2;
                end else settle<=settle+1'b1;
                2:begin
                    // Two pipeline stages keep checksum assembly away from TX launch.
                    if(length==16)begin
                        checksum<=bytes[0]+bytes[1]+bytes[2]+bytes[3]+bytes[4]+bytes[5]+bytes[6]+bytes[7]+
                                  bytes[8]+bytes[9]+bytes[10]+bytes[11]+bytes[12]+bytes[13]+bytes[14];
                    end
                    state<=3;
                end
                3:begin if(length==16)bytes[15]<=checksum;state<=4;end
                4:if(!uart_tx_busy)begin uart_tx_data<=bytes[count];uart_tx_en<=1;state<=5;end
                5:if(uart_tx_busy)state<=6;
                6:if(!uart_tx_busy)begin
                    if(count==length-1)begin packet_tx_done<=1;state<=0;end
                    else begin count<=count+1'b1;state<=4;end
                end
                default:state<=0;
            endcase
        end
    end
endmodule
