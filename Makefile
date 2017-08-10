MKLROOT=/home/security/intel/mkl
MKLDNNROOT=/home/security/.local
COMMON_FLAGS=-O4 -std=c++11

all:main.o mkldnn_conv.o im2col_mkl.o
	g++ $(COMMON_FLAGS) -o main $^ \
		-L ${MKLDNNROOT}/lib -lmkldnn -lmklml_intel \
		-Wl,--start-group \
		${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
		${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
		${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group \
		-lpthread -lm -ldl

%.o:%.cpp
	g++ $(COMMON_FLAGS) -o $@ -c $< -I ${MKLROOT}/include -I /home/security/.local/include

clean:
	rm main *.o
