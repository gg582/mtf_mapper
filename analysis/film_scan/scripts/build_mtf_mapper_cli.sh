#!/usr/bin/env bash
# Build the mtf_mapper command-line tools without root and without the Qt GUI.
# Fetches OpenCV 4.10 (minimal modules) and TCLAP 1.2.5 into <prefix>; Eigen must already be installed
# (any Eigen 3.4+ / 5.x; pass its include dir as the 3rd argument).
#
# usage: build_mtf_mapper_cli.sh <prefix> <build_dir> <eigen_include_dir>
# result: <build_dir>/bin/mtf_mapper (run with LD_LIBRARY_PATH=<prefix>/lib)
set -euo pipefail
prefix="$(realpath -m "$1")"; build="$(realpath -m "$2")"; eigen="$3"
src="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$prefix/src" "$build"

cd "$prefix/src"
[ -d opencv-4.10.0 ] || curl -sL https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz | tar xz
[ -d tclap-1.2.5 ] || curl -sL https://sourceforge.net/projects/tclap/files/tclap-1.2.5.tar.gz/download | tar xz

mkdir -p opencv-build && cd opencv-build
cmake ../opencv-4.10.0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,highgui \
  -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_IPP=OFF -DWITH_OPENEXR=OFF \
  -DWITH_ADE=OFF -DBUILD_TIFF=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DBUILD_ZLIB=ON \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF > cmake.log
make -j"$(nproc)" > make.log && make install > install.log

cd "$build"
cmake "$src" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DMTFMAPPER_BUILD_GUI=OFF \
  -DOpenCV_DIR="$prefix/lib/cmake/opencv4" -DEIGEN3_INCLUDE_DIR="$eigen" \
  -DTCLAP_INCLUDE_DIR="$prefix/src/tclap-1.2.5/include"
make -j"$(nproc)"
