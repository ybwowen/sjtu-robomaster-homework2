# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ybwowen/Desktop/source/homework2/photograph

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ybwowen/Desktop/source/homework2/photograph/build

# Include any dependencies generated for this target.
include CMakeFiles/photograph.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/photograph.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/photograph.dir/flags.make

CMakeFiles/photograph.dir/main.cpp.o: CMakeFiles/photograph.dir/flags.make
CMakeFiles/photograph.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ybwowen/Desktop/source/homework2/photograph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/photograph.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/photograph.dir/main.cpp.o -c /home/ybwowen/Desktop/source/homework2/photograph/main.cpp

CMakeFiles/photograph.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/photograph.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ybwowen/Desktop/source/homework2/photograph/main.cpp > CMakeFiles/photograph.dir/main.cpp.i

CMakeFiles/photograph.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/photograph.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ybwowen/Desktop/source/homework2/photograph/main.cpp -o CMakeFiles/photograph.dir/main.cpp.s

# Object files for target photograph
photograph_OBJECTS = \
"CMakeFiles/photograph.dir/main.cpp.o"

# External object files for target photograph
photograph_EXTERNAL_OBJECTS =

photograph: CMakeFiles/photograph.dir/main.cpp.o
photograph: CMakeFiles/photograph.dir/build.make
photograph: /usr/local/lib/libopencv_gapi.so.4.8.0
photograph: /usr/local/lib/libopencv_highgui.so.4.8.0
photograph: /usr/local/lib/libopencv_ml.so.4.8.0
photograph: /usr/local/lib/libopencv_objdetect.so.4.8.0
photograph: /usr/local/lib/libopencv_photo.so.4.8.0
photograph: /usr/local/lib/libopencv_stitching.so.4.8.0
photograph: /usr/local/lib/libopencv_video.so.4.8.0
photograph: /usr/local/lib/libopencv_videoio.so.4.8.0
photograph: /usr/local/lib/libopencv_imgcodecs.so.4.8.0
photograph: /usr/local/lib/libopencv_dnn.so.4.8.0
photograph: /usr/local/lib/libopencv_calib3d.so.4.8.0
photograph: /usr/local/lib/libopencv_features2d.so.4.8.0
photograph: /usr/local/lib/libopencv_flann.so.4.8.0
photograph: /usr/local/lib/libopencv_imgproc.so.4.8.0
photograph: /usr/local/lib/libopencv_core.so.4.8.0
photograph: CMakeFiles/photograph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ybwowen/Desktop/source/homework2/photograph/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable photograph"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/photograph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/photograph.dir/build: photograph

.PHONY : CMakeFiles/photograph.dir/build

CMakeFiles/photograph.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/photograph.dir/cmake_clean.cmake
.PHONY : CMakeFiles/photograph.dir/clean

CMakeFiles/photograph.dir/depend:
	cd /home/ybwowen/Desktop/source/homework2/photograph/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ybwowen/Desktop/source/homework2/photograph /home/ybwowen/Desktop/source/homework2/photograph /home/ybwowen/Desktop/source/homework2/photograph/build /home/ybwowen/Desktop/source/homework2/photograph/build /home/ybwowen/Desktop/source/homework2/photograph/build/CMakeFiles/photograph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/photograph.dir/depend

