# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.18.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.18.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/weixiang/Downloads/INF573Lab3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/weixiang/Downloads/INF573Lab3/build

# Include any dependencies generated for this target.
include CMakeFiles/homographie.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/homographie.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/homographie.dir/flags.make

CMakeFiles/homographie.dir/homographie.cpp.o: CMakeFiles/homographie.dir/flags.make
CMakeFiles/homographie.dir/homographie.cpp.o: ../homographie.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/weixiang/Downloads/INF573Lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/homographie.dir/homographie.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/homographie.dir/homographie.cpp.o -c /Users/weixiang/Downloads/INF573Lab3/homographie.cpp

CMakeFiles/homographie.dir/homographie.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/homographie.dir/homographie.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/weixiang/Downloads/INF573Lab3/homographie.cpp > CMakeFiles/homographie.dir/homographie.cpp.i

CMakeFiles/homographie.dir/homographie.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/homographie.dir/homographie.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/weixiang/Downloads/INF573Lab3/homographie.cpp -o CMakeFiles/homographie.dir/homographie.cpp.s

# Object files for target homographie
homographie_OBJECTS = \
"CMakeFiles/homographie.dir/homographie.cpp.o"

# External object files for target homographie
homographie_EXTERNAL_OBJECTS =

homographie: CMakeFiles/homographie.dir/homographie.cpp.o
homographie: CMakeFiles/homographie.dir/build.make
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_calib3d.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_core.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_dnn.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_features2d.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_flann.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_highgui.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_imgcodecs.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_imgproc.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_ml.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_objdetect.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_photo.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_shape.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_stitching.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_superres.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_video.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_videoio.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_videostab.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_aruco.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_bgsegm.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_bioinspired.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_ccalib.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_datasets.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_dnn_objdetect.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_dpm.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_face.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_freetype.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_fuzzy.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_hdf.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_hfs.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_img_hash.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_line_descriptor.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_optflow.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_phase_unwrapping.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_plot.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_reg.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_rgbd.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_saliency.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_stereo.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_structured_light.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_surface_matching.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_text.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_tracking.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_xfeatures2d.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_ximgproc.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_xobjdetect.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_xphoto.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_shape.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_photo.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_calib3d.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_phase_unwrapping.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_video.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_datasets.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_plot.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_text.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_dnn.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_features2d.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_flann.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_highgui.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_ml.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_videoio.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_imgcodecs.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_objdetect.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_imgproc.3.4.2.dylib
homographie: /Users/weixiang/opt/anaconda3/envs/zopencv/lib/libopencv_core.3.4.2.dylib
homographie: CMakeFiles/homographie.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/weixiang/Downloads/INF573Lab3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable homographie"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/homographie.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/homographie.dir/build: homographie

.PHONY : CMakeFiles/homographie.dir/build

CMakeFiles/homographie.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/homographie.dir/cmake_clean.cmake
.PHONY : CMakeFiles/homographie.dir/clean

CMakeFiles/homographie.dir/depend:
	cd /Users/weixiang/Downloads/INF573Lab3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/weixiang/Downloads/INF573Lab3 /Users/weixiang/Downloads/INF573Lab3 /Users/weixiang/Downloads/INF573Lab3/build /Users/weixiang/Downloads/INF573Lab3/build /Users/weixiang/Downloads/INF573Lab3/build/CMakeFiles/homographie.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/homographie.dir/depend
