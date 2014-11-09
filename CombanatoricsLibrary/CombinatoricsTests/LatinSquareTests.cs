using System;
using CombanatoricsLibrary;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

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
        }

        [TestMethod]
        public void CheckValuesTest()
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
