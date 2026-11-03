cfg_connect -ip 127.0.0.1 -port 65431
cfg_scan_chain
cfg_read_device_property -device_index 0 -mode 0
cfg_assign_file -device_index 0 -file D:/PDS_WORK/adas_fullres_20261028/generate_bitstream/test_ddr.sbit
cfg_program -device_index 0
cfg_read_device_property -device_index 0 -mode 3
cfg_disconnect
exit
