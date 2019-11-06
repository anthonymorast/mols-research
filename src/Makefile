# for building
opt = -O3
std = -std=c++11
flags = $(std) $(opt)

# for cleaning
util = utilities
gen = generate_squares
exes = $(util) $(gen)


all: LatinSquare.o Utils.o GenerateSquares.o GenerateSquares Utils
	
LatinSquare.o: LatinSquare.cpp
	g++ $(flags) -c LatinSquare.cpp
Utils.o: Utils.cpp
	g++ $(flags) -c Utils.cpp
Utils:
	g++ $(flags) LatinSquare.o Utils.o -o $(util)
GenerateSquares.o: GenerateSquares.cpp
	g++ $(flags) -c GenerateSquares.cpp
GenerateSquares: LatinSquare.o GenerateSquares.o
	g++ $(flags) LatinSquare.o GenerateSquares.o -o $(gen)

clean:
	rm *.o $(exes)
