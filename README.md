# CUDA Mandelbrot

An interactive CUDA Mandelbrot set explorer, runs on Windows and Linux.  

## Example output:
![Section from a regular Mandelbrot set](https://i.imgur.com/ncQQbIC.png)
!["Burning Ship" set](https://i.imgur.com/DA7njWK.png)

## Dependencies
* Windows: Visual Studio 2013 (other versions untested)
* Linux: C++11 compatible compiler
* CUDA 7.0 or higher
* GPU with Compute Capability 3.0 (Kepler, GTX 6xx) or higher

## Help

**IMPORTANT**: Keep in mind that by default most OS will kill your process and reset the driver if a GPU job doesn't return after a few seconds. If you intend to generate large pictures, you should increase this timeout.  
Even during normal GUI operation it is very easy to accidentally hit this limit since the application will switch from single-precision to double-precision calculations if required for the current zoom level. This causes a slowdown between 8x and 32x on most consumer GPUs (2x on some professional models which have more 64-bit ALUs).

Default mode is interactive GUI with help on screen.  
<img src="https://i.imgur.com/p3bLGqe.png" alt="GUI Screenshot" width="400px"/>  

Get command line options with `--help`.  
Tip: Use the GUI to find a nice section and dump the parameters out to generate a large image offline via the CLI (see next section).  
<img src="https://i.imgur.com/u8drxG0.png" alt="CLI Screenshot" width="400px"/>  

### Saving Parameters and Images

In GUI mode you can press **P** to write the parameters of the current view to `input.par`. **CTRL+P** will load parameters from that file.  
You can use `-f xxx.par` from the command line to load parameters at startup.

Pressing **I** will write the current image to `output.bmp` exactly as it is displayed on screen (without the text overlay). **CTRL+I** will write the supersampling buffer instead, i.e. if you are displaying at 16 samples per pixel, the image written will be 16x as large.

### Fractal Types

Mandelbrot Set (-t 0, default):  
<img src="https://i.imgur.com/uVGnfAA.png" alt="-t 0" width="400px"/>  
Exponent 3 (-t 1):  
<img src="https://i.imgur.com/7XOHUEJ.png" alt="-t 1" width="400px"/>  
Arbitrary Exponent (-t 2):  
<img src="https://i.imgur.com/jVI3me7.png" alt="-t 2" width="400px"/>  
Burning Ship (-t 3):  
<img src="https://i.imgur.com/KAuu0RT.png" alt="-t 3" width="400px"/>  

### Coloring Options

Distance function 0 (-c 0):  
<img src="https://i.imgur.com/ntNbYj6.png" alt="-c 0" width="400px"/>  
Distance function 1 (-c 1):  
<img src="https://i.imgur.com/gMHRdm9.png" alt="-c 1" width="400px"/>  
Distance function 2 (-c 2):  
<img src="https://i.imgur.com/bbkiOHA.png" alt="-c 2" width="400px"/>  
Smooth iteration count (-c 3, default):  
<img src="https://i.imgur.com/uVGnfAA.png" alt="-c 3" width="400px"/>  

## Acknowledgements

Implementation of smooth iteration count from [Íñigo Quílez, Smooth Iteration Count for Generalized Mandelbrot Sets](http://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm).  
Implementation of Hubbard-Douady distance function from [Íñigo Quílez, Distance Rendering for Fractals](http://iquilezles.org/www/articles/distancefractals/distancefractals.htm).
