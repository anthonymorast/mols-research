#define BOOST_TEST_MODULE LatinSquareTests
#include <boost/test/unit_test.hpp>
#include "../LatinSquare_hdrOnly.hpp"

BOOST_AUTO_TEST_CASE(EqualityTest)
{
	LatinSquare sq (4, new short[16]{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq3(4, new short[16]{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });

	//LatinSquare sq2(4, new short[16]{ 1, 3, 4, 2, 2, 4, 3, 1, 3, 1, 2, 4, 4, 2, 1, 3 });

	//LatinSquare sq6(10);
	//LatinSquare sq7(100);

	//BOOST_TEST(sq != sq2);
	//BOOST_TEST(sq == sq3);
	//BOOST_CHECK_THROW(sq7 == sq, InvalidSquareException);
}

BOOST_AUTO_TEST_CASE(ValidityTest)
{
	LatinSquare sq(4, new short[4 * 4]{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq2(4, new short[16]{ 1, 3, 4, 2, 2, 4, 3, 1, 3, 1, 2, 4, 4, 2, 1, 3 });
	LatinSquare sq3(4, new short[4 * 4]{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq4(4, new short[4 * 4]{ 1, 1, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq5(4, new short[4 * 4]{ 1, 2, 3, 4, 1, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq8(2, new short[4]{ 1, 2, 2, 1 });

	// empty squares are not valid squares...
	LatinSquare sq6(10);
	LatinSquare sq7(100);

	BOOST_TEST(sq.is_valid() == true);
	BOOST_TEST(sq2.is_valid() == true);
	BOOST_TEST(sq3.is_valid() == true);
	BOOST_TEST(sq4.is_valid() == false);
	BOOST_TEST(sq5.is_valid() == false);
	BOOST_TEST(sq6.is_valid() == false);
	BOOST_TEST(sq7.is_valid() == false);
	BOOST_TEST(sq8.is_valid() == true);
}

BOOST_AUTO_TEST_CASE(OrthogonalityTest)
{
	// invalid squares are not orthogonal
	LatinSquare sq(4, new short[4 * 4]{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq2(4, new short[16]{ 1, 3, 4, 2, 2, 4, 3, 1, 3, 1, 2, 4, 4, 2, 1, 3 });
	LatinSquare sq3(4, new short[4 * 4]{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq4(4, new short[4 * 4]{ 1, 1, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq5(4, new short[4 * 4]{ 1, 2, 3, 4, 1, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1 });
	LatinSquare sq8(2, new short[4]{ 1, 2, 2, 1 });

	// empty squares are not valid squares
	LatinSquare sq6(10);
	LatinSquare sq7(100);

	BOOST_TEST(sq.is_orthogonal(sq) == false);
	BOOST_TEST(sq.is_orthogonal(sq2) == true);
	BOOST_TEST(sq.is_orthogonal(sq3) == false);

	// empty
	BOOST_REQUIRE_THROW(sq7.is_orthogonal(sq), InvalidSquareException);
	BOOST_REQUIRE_THROW(sq6.is_orthogonal(sq), InvalidSquareException);

	// invalid values
	BOOST_REQUIRE_THROW(sq4.is_orthogonal(sq), InvalidSquareException);
	BOOST_REQUIRE_THROW(sq5.is_orthogonal(sq), InvalidSquareException);
}

BOOST_AUTO_TEST_CASE(PermuteRowsTest)
{
	LatinSquare sq(4, new short[16]{ 1, 2, 3, 4,   2, 1, 4, 3,   3, 4, 1, 2,   4, 3, 2, 1 });
	LatinSquare lst_three(4, new short[16]{1, 2, 3, 4,   3, 4, 1, 2,   2, 1, 4, 3,   4, 3, 2, 1});
	LatinSquare all_4(4, new short[16]{ 2, 1, 4, 3,  1, 2, 3, 4,  4, 3, 2, 1,  3, 4, 1, 2 });

	short values[4] = {0, 2, 1, 3};
	sq.permute_rows(values);
	BOOST_TEST(sq == lst_three);
	BOOST_TEST((sq == all_4) == false);

	sq = LatinSquare(4, new short[16]{ 1, 2, 3, 4,   2, 1, 4, 3,   3, 4, 1, 2,   4, 3, 2, 1 });
	short vals2[4] = {1, 0, 3, 2};
	sq.permute_rows(vals2);
	BOOST_TEST(sq == all_4);
	BOOST_TEST((sq == lst_three) == false);

	LatinSquare invalid(10);
	BOOST_CHECK_THROW(invalid.permute_rows(values), InvalidSquareException);
}

BOOST_AUTO_TEST_CASE(ConstructorTest)
{
	LatinSquare sq(10);
	BOOST_TEST(sq.is_valid() == false);

	sq = LatinSquare(2, new short[4]{1, 2, 2, 1});
	BOOST_TEST(sq.is_valid() == true);

	LatinSquare sq2 = sq;
	BOOST_TEST((sq == sq2) == true);
	sq2.permute_rows(new short[2]{ 1, 0 });
	BOOST_TEST((sq == sq2) == false);
}

BOOST_AUTO_TEST_CASE(RCSPermutationTest)
{
	LatinSquare sq(3, new short[9]{ 0, 1, 2, 1, 2, 0, 2, 0, 1 });
	LatinSquare chk_sq(3, new short[9]{ 0, 2, 1, 1, 0, 2, 2, 1, 0 });
	short perm[3] = {2, 0, 1};
	sq.rcs_permutation(perm);
	BOOST_TEST(sq == chk_sq);

	// TODO: need many many more test cases
}

BOOST_AUTO_TEST_CASE(SymbolPermutationTest)
{
	LatinSquare sq(3, new short[9]{ 0, 1, 2, 1, 2, 0, 2, 0, 1 });
	LatinSquare chk_sq(3, new short[9]{ 1, 0, 2, 0, 2, 1, 2, 1, 0 });
	short perm[3] = { 1, 0, 2 };
	sq.permute_symbols(perm);
	BOOST_TEST(sq == chk_sq);

	// TODO: need many many more test cases
}

BOOST_AUTO_TEST_CASE(ColumnsPermutationTest)
{
	LatinSquare sq(3, new short[9]{0, 1, 2, 1, 2, 0, 2, 0, 1});
	LatinSquare chk_sq(3, new short[9]{2, 0, 1, 0, 1, 2, 1, 2, 0});
	short perm[3] = {2, 0, 1};
	sq.permute_cols(perm);
	BOOST_TEST(sq == chk_sq);

	LatinSquare sq4(4, new short[16]{ 1, 2, 3, 4,
	   					 			  2, 1, 4, 3,
									  3, 4, 1, 2,
									  4, 3, 2, 1 });
	LatinSquare chk_sq4(4, new short[16]{2, 1, 4, 3,
										 1, 2, 3, 4,
										 4, 3, 2, 1,
										 3, 4, 1, 2});
  short perm2[4] = {1, 0, 3, 2};
	sq4.permute_cols(perm2);
	BOOST_TEST(sq4 == chk_sq4);
}

BOOST_AUTO_TEST_CASE(ColumnsAndRowPermuteTest)
{

}
