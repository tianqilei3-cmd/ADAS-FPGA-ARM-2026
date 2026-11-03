`timescale 1ns/1ps
module adas_key_epoch_gate(
 input wire drive_request,status_valid,controller_stop,controller_valid,
 input wire [7:0] key_sequence,status_epoch,accepted_epoch,
 input wire [2:0] controller_action,
 output wire filtered_status_valid,effective_stop,effective_valid,
 output wire [2:0] effective_action
);
 wire epoch_matches=(status_epoch==key_sequence)&&(accepted_epoch==key_sequence);
 assign filtered_status_valid=status_valid&&(status_epoch==key_sequence);
 assign effective_stop=controller_stop|!drive_request|!epoch_matches;
 assign effective_action=effective_stop ? 3'd1 : controller_action;
 assign effective_valid=controller_valid&&epoch_matches;
endmodule
