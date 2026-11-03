`timescale 1ns/1ps
// Both streams are in the existing HDMI pixel clock domain. A final matched
// data/sync register avoids a combinational path to the transmitter pins.
module adas_video_output_mux(
 input wire clk,rst_n,select_fullres,
 input wire raw_vs,raw_hs,raw_de,input wire [23:0] raw_rgb,
 input wire legacy_vs,legacy_hs,legacy_de,input wire [23:0] legacy_rgb,
 output reg o_vs,o_hs,o_de,output reg [23:0] o_rgb
);
 always @(posedge clk)begin
  if(!rst_n)begin o_vs<=0;o_hs<=0;o_de<=0;o_rgb<=0;end
  else if(select_fullres)begin o_vs<=raw_vs;o_hs<=raw_hs;o_de<=raw_de;o_rgb<=raw_rgb;end
  else begin o_vs<=legacy_vs;o_hs<=legacy_hs;o_de<=legacy_de;o_rgb<=legacy_rgb;end
 end
endmodule
