# MKL-DNN Performance Test

This repository is used to test the performance of the MKL-DNN.

## How to Compile

Clone this repository and edit the Makefile. The variables ```MKLROOT``` and ```MKLDNNROOT``` are where you installed the MKL and MKL-DNN respectively. Finally, make it!

```
git clone https://github.com/GD06/mkldnn-perf.git
make
```

## How to Run

The compilation will generate an executable file ```main```. This program takes seven configurations as the input, they are ```(IN, IC, IH, IW, FH, FW, OC)```, where the input tensor has the shape ```(IN, IC, IH, IW)``` following the NCHW data layout and the filter tensor has the shape ```(OC, IC, FH, FW)``` following the OIHW data layout. 

For example, running the following command will test the performance for the input tensor with shape (100, 32, 224, 224) and the filter tensor shape (32, 32, 3, 3). We take the stride as 1 and the padding as 0 as default.

```
./main 100 32 224 224 3 3 32
```

## What is the Output

The program will output one line for each time execution. The information includes the program size (i.e. the seven parameters), the elapsed time and the achieved GFLOPs for the im2col+MKL matrix multiplication and MKL-DNN convolution forward method. The last component of the output is the validation results of comparing the output between im2col+MKL and MKL-DNN to ensure that we implement the function as we expect. 

In the file ```param.h```, we define a macro, LOOP\_TIMES, which defines the times we will repeat the im2col+MKL and MKL-DNN for recording the performance. The default is 1 which means we will execute these two methods both for only one time. We suggest use larger number can reduce the experimental deviations. However, for the LOOP\_TIMES larger than 1, the outputs of these two methods could be different. 