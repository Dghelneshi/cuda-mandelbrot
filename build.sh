FILES="*.cpp *.c *.cu"
FLAGS="-std=c++11 -Wall -Wextra -O3 -arch=sm_35 -res-usage -lineinfo -use-fast-math"
FOLDERS="-Iinclude"
LIBS="-lGL -lglut -lGLU"
TARGET="bin/mandelbrot"

mkdir -p bin
nvcc $FILES $FOLDERS $FLAGS $LIBS -o $TARGET