#
# Copyright 2020 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Create a project
open_project -reset proj_training

source commonSettings.tcl

# Add design files
add_files -cflags "$defines $include_paths $compiler_flags" "kernel/Training.cpp"
#add_files -cflags "-I${ws} -I${ws}/Vitis_Libraries" VitisGemv.cpp
# Add test bench & files
add_files -tb -cflags "$defines $include_paths $compiler_flags -DTEST_TRAINING" "test/Test.cpp kernel/Mlp.cpp"
#add_files -tb result.golden.dat

# Set the top-level function
# set_top MultA
set_top BGD

# ########################################################
# Create a solution
open_solution -reset solution1
# Define technology and clock rate
set_part  {xczu9eg-ffvc900-1-e}
create_clock -period 10

# Source x_hls.tcl to determine which steps to execute
set hls_exec 3
csim_design
# Set any optimization directives
# set_directive_pipeline loop_pipeline/LOOP_J
config_interface -m_axi_addr64=0
# End of directives

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	# set_directive_interface -mode m_axi -offset slave -depth 128 -max_read_burst_length 128 -bundle gmem2 MultiLayerPerceptron hiddenWeights
	csynth_design
	
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	
	cosim_design
} elseif {$hls_exec == 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	
	cosim_design
	export_design
} else {
	# Default is to exit after setup
}

exit

