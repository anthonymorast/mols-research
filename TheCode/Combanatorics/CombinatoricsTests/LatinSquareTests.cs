using System;
using Combanatorics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;

namespace CombinatoricsTests
{
    [TestClass]
    public class LatinSquareTests
    {
        [TestMethod]
        public void CreateSquaresTest()
        {
            LatinSquare square1 = new LatinSquare(10);
            LatinSquare square2 = new LatinSquare(3, new List<int>{ 1, 2, 3, 3, 1, 2, 2, 3, 1 });
            LatinSquare square3 = new LatinSquare(4, new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 1, 2 });

            //Assert.AreEqual(square3.GetOrder(), 4);
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), "Bad row in creating the square.")]
        public void BadRowSquareTest()
        {
            LatinSquare square3 = new LatinSquare(4, new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 4, 1, 2 });
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), "Bad column in creating the square.")]
        public void BadColSquareTest()
        {
            LatinSquare square3 = new LatinSquare(4, new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 2, 1 });
        }

        [TestMethod]
        [ExpectedException(typeof(Exception), "Not enough values in square.")]
        public void IncorrectNumberOfValuesTest ()
        {
            LatinSquare square3 = new LatinSquare(3, new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 2, 1 });
        }

        [TestMethod]
        public void FillSquaresTest()
        {
            LatinSquare square1 = new LatinSquare(4);
            square1.Fill(new List<int>{ 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 1, 2 });

            Assert.AreEqual(square1.GetOrder(), 4);
        }

        [TestMethod]
        public void GetOrderTest()
        {
            LatinSquare square = new LatinSquare(4);
            LatinSquare sqaure1 = new LatinSquare(4, new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 1, 2 });

            Assert.AreEqual(square.GetOrder(), square.GetOrder());
        }

        [TestMethod]
        public void SetValuesTest()
        {
            LatinSquare square1 = new LatinSquare(3);
            square1.SetValues(new List<int> { 1, 2, 3, 2, 3, 1, 3, 1, 2 });

            Assert.AreEqual(square1.GetOrder(), 3);
        }

        [TestMethod]
        public void GetElementAtPositionTest()
        {
            int correctElementCount = 0;
            LatinSquare square1 = new LatinSquare(4);
            square1.Fill(new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 1, 2 });

            if (square1.GetElementAtPosition(1, 1) == 1)    // correct - count = 1
                correctElementCount++;
            if (square1.GetElementAtPosition(1, 2) == 2)    // correct - count = 2
                correctElementCount++;
            if (square1.GetElementAtPosition(1, 3) == 3)    // correct - count = 3
                correctElementCount++;
            if (square1.GetElementAtPosition(1, 4) == 4)    // correct - count = 4
                correctElementCount++;
            if (square1.GetElementAtPosition(4, 4) == 2)    // correct - count = 5
                correctElementCount++;
            if (square1.GetElementAtPosition(3, 3) == 2)    // correct - count = 6
                correctElementCount++;
            if (square1.GetElementAtPosition(2, 4) == 4)    // incorrect - count = 6
                correctElementCount++;
            if (square1.GetElementAtPosition(2, 1) == 2)    // correct - count = 7
                correctElementCount++;
            if (square1.GetElementAtPosition(3, 4) == 1)    // correct - count = 8
                correctElementCount++;
            if (square1.GetElementAtPosition(4, 3) == 2)    // incorrect - count = 8
                correctElementCount++;
            if (square1.GetElementAtPosition(3, 1) == 3)    // correct - count = 9
                correctElementCount++;

            Assert.AreEqual(correctElementCount, 9);
        }

        [TestMethod]
        public void PermuteRowsTest()
        {
            LatinSquare square1 = new LatinSquare(3);
            square1.SetValues(new List<int> { 1, 2, 3, 2, 3, 1, 3, 1, 2 });

            Console.WriteLine(square1.ToString());
            Console.WriteLine();
            square1.PermuteRows(new List<int> { 1, 3, 2 });
            Console.WriteLine(square1.ToString());
            Console.WriteLine();
            square1.PermuteRows(new List<int> { 2, 3, 1 });
            Console.WriteLine(square1.ToString());
            Console.WriteLine();

            LatinSquare square2 = new LatinSquare(4);
            square2.Fill(new List<int> { 1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 2, 1, 4, 3, 1, 2 });

            Console.WriteLine(square2.ToString());
            Console.WriteLine();
            square2.PermuteRows(new List<int> { 4, 1, 2, 3 });
            Console.WriteLine(square2.ToString());
            Console.WriteLine();
            square2.PermuteRows(new List<int> { 2, 3, 4, 1});
            Console.WriteLine(square2.ToString());
            Console.WriteLine();
        }


        [TestMethod]
        public void CheckGoodValuesTest()
        {

        }

        [TestMethod]
        public void IsMutuallyOrthogonalTest()
        {

        }

        [TestMethod]
        public void IsSameIsotopyTest()
        {

        }

        [TestMethod]
        public void IsSameMainClassTest()
        {

        }
    }
}
