@echo off
SETLOCAL

ECHO THIS DOESN'T WORK YET

for %%a in (../*.cpp) do (
    for /f "tokens=1 delims=_" %%n in ("%%~na") do (
        call SET files=%%files%% ../%%a
    )
)
for %%a in (../*.c) do (
    for /f "tokens=1 delims=_" %%n in ("%%~na") do (
        call SET files=%%files%% ../%%a
    )
)
for %%a in (../*.cu) do (
    for /f "tokens=1 delims=_" %%n in ("%%~na") do (
        call SET files=%%files%% ../%%a
    )
)


SET FLAGS=-std=c++11 -O3 -arch=sm_35 -res-usage -lineinfo -use_fast_math -DGLEW_STATIC -DFREEGLUT_STATIC -Xlinker /NODEFAULTLIB:library -Xlinker /LTCG
SET FOLDERS=-I../include -L../lib
SET LIBS=-lfreeglut_static
SET TARGET=../bin/mandelbrot

nvcc %files% %FOLDERS% %FLAGS% %LIBS% -o %TARGET%

ENDLOCAL
