all: main.out wt.out benchmarks.out

main.out: main.cpp
	g++ -std=c++11 -march=native -O3 -DNDEBUG -I ~/include -L ~/lib main.cpp -o main.out -lsdsl -ldivsufsort -ldivsufsort64 

wt.out: wt.cpp
	g++ -std=c++11 -march=native -O3 -DNDEBUG -I ~/include -L ~/lib wt.cpp -o wt.out -lsdsl -ldivsufsort -ldivsufsort64

benchmarks.out: benchmarks.cpp
	g++ -std=c++11 -march=native -O3 -DNDEBUG -I ~/include -L ~/lib benchmarks.cpp -o benchmarks.out -lsdsl -ldivsufsort -ldivsufsort64
	
run: main.out
	./main.out
