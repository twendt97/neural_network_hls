set ws "/home/thilo/master_thesis_code/uz_neural_network_hls_refactor"
set defines "-DVITIS_MAJOR_VERSION=2020 -DVITIS_MINOR_VERSION=1 -DVITIS_VERSION=2020.1 -D__VITIS_HLS__ -DMM_SYNTHESIS -DHLSLIB_SYNTHESIS -DHLSLIB_XILINX"
set include_paths "-I${ws} -I${ws}/include -I${ws}/Vitis_Libraries/blas/L1/include/hw"
set compiler_flags "-fexceptions"