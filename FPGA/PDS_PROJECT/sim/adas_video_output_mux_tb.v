`timescale 1ns/1ps
module adas_video_output_mux_tb;
 reg clk=0;always #5 clk=~clk;
 reg rst_n=0,sel=0,rv=0,rh=0,rd=0,lv=0,lh=0,ld=0;
 reg [23:0] rgb=0,legacy=0;wire ov,oh,od;wire [23:0] out;integer i,checks=0;
 adas_video_output_mux dut(clk,rst_n,sel,rv,rh,rd,rgb,lv,lh,ld,legacy,ov,oh,od,out);
 initial begin
  @(posedge clk);#1;if({ov,oh,od,out}!==0)$fatal(1,"reset not blank");
  @(negedge clk);rst_n=1;
  for(i=0;i<96;i=i+1)begin
   @(negedge clk);sel=i[4];rv=i[0];rh=i[1];rd=i[2];lv=~i[0];lh=~i[1];ld=~i[2];rgb=24'h100000+i;legacy=24'habcdef-i;
   @(posedge clk);#1;
   if(sel)begin if({ov,oh,od,out}!=={rv,rh,rd,rgb})$fatal(1,"fullres pixel/sync mismatch");end
   else begin if({ov,oh,od,out}!=={lv,lh,ld,legacy})$fatal(1,"legacy pixel/sync mismatch");end
   checks=checks+1;
  end
  $display("PASS full resolution matched pixel and sync checks=%0d",checks);$finish;
 end
endmodule
