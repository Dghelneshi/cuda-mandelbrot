FILES="*.cpp *.c *.cu"
FLAGS="-std=c++11 -O3 -arch=sm_35 -res-usage -lineinfo"
FOLDERS="-Iinclude -Llib"
LIBS="-lGL -lglut -lGLU"
TARGET="bin/mandelbrot"

mkdir -p bin
nvcc $FILES $FOLDERS $FLAGS $LIBS -o $TARGET