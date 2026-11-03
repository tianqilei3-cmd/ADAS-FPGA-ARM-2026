module cmd_exe(
    //input
    input              clk       ,     //系统时钟，50MHZ
    input              rst_n     ,     //系统复位，低电平有效
    input     [2:0]    led_data  ,     //xz
    output    [2:0]    led            //xz
);

//*****************************************************
//**                    main code
//*****************************************************
assign  led = led_data  ;    //输出xz信号

endmodule