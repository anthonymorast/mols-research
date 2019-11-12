all: LatinSquareTest LatinSquareHeaderOnlyTest

LatinSquareTest:
	g++ LatinSquareTest.cpp -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK -o lstest

LatinSquareHeaderOnlyTest:
	g++ LatinSquareTest_hdrOnly.cpp -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK -o lstest_hdr

clean:
	rm lstest
