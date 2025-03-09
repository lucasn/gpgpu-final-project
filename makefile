gpu: main.cu matrix.cu ann.cu mnist.cu
	nvcc -o ./ann main.cu matrix.cu ann.cu mnist.cu -lm

gprof: main.cu matrix.cu ann.cu mnist.cu
	nvcc -o ./ann main.cu matrix.cu ann.cu mnist.cu -lm -pg

test: test.cu matrix.cu
	nvcc -o ./test test.cu matrix.cu -lm -pg

clean:
	rm -f ann gmon.out test