module gamma_local #(
    parameter WIN_W = 64,
    parameter WIN_H = 32
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       i_vs,
    input  wire       i_hs,
    input  wire       i_de,
    input  wire [7:0] i_v,
    output reg        o_vs,
    output reg        o_hs,
    output reg        o_de,
    output reg [7:0]  o_v
);
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            o_vs<=0; o_hs<=0; o_de<=0; o_v<=0;
        end else begin
            o_vs<=i_vs; o_hs<=i_hs; o_de<=i_de;
            o_v <= i_v; // 旁路
        end
    end
endmodule