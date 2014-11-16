using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Combanatorics
{
    public class LatinSquare
    {
        /// <summary>
        /// The size of the Latin Square (square will be size x size)
        /// </summary>
        private int squareOrder = 0;

        /// <summary>
        /// List of all the values within the Latin Square
        /// </summary>
        private List<int> values = new List<int>();

        /// <summary>
        /// Create an empty Latin square of size n x n 
        /// </summary>
        /// <param name="n"></param>
        public LatinSquare(int order)
        {
            squareOrder = order;
        }

        /// <summary>
        /// Create a Latin square of size n x n and fill it with the values. The list of values should be
        /// "row major". That is, the first n (where n is the order of the square) elements should be the 
        /// entries for row one in the square. The second n should be row two, etc.
        /// </summary>
        /// <param name="n"></param>
        /// <param name="listOfValues"></param>
        public LatinSquare(int order, List<int> listOfValues)
        {
            squareOrder = order;

            // check if the list of values contains n squared entries (a square)
            SetValues(listOfValues);
        }

        /// <summary>
        /// Returns the order of the Latin square.
        /// </summary>
        /// <returns></returns>
        internal int GetOrder()
        {
            return this.squareOrder;
        }

        /// <summary>
        /// Set the values of the Latin square.
        /// </summary>
        /// <param name="valueList"></param>
        /// <returns></returns>
        internal void SetValues(List<int> valueList)
        {
            string errorMessage = "";

            if (valueList.Count != (squareOrder * squareOrder))
            {
                throw new Exception("Incorrect number of values to fill Latin Square");
            }

            if (CheckValues(valueList, out errorMessage))
                this.values = valueList;
            else
            {
                throw new Exception(errorMessage);
            }
        }

        /// <summary>
        /// Checks to ensure the values will create a Latin square
        /// </summary>
        /// <param name="valueList"></param>
        /// <returns></returns>
        internal bool CheckValues(List<int> valueList, out string error)
        {
            // create rows and column lists
            List<int[]> rows = new List<int[]>();
            List<int[]> cols = new List<int[]>();
            for (int i = 0; i < squareOrder; i++)
            {
                rows.Add(new int[squareOrder]);
                cols.Add(new int[squareOrder]);
            }

            // fill the row and columns lists
            for (int i = 0; i < squareOrder; i++)
            {
                for (int j = 0; j < squareOrder; j++)
                {
                    int currentElemnet = valueList[(i * squareOrder) + j];
                    if ( currentElemnet > squareOrder || currentElemnet < 0)
                    {
                        error = string.Format(
                            @"Element '{0}' in row {1} column {2} is outside of valid range values. A Latin square should contain elements from 1 to N, where N is the order of the square",
                            currentElemnet, i + 1, j + 1);
                        return false;
                    }

                    rows[i][j] = currentElemnet;
                    cols[j][i] = currentElemnet;
                }
            }

            // check for same elements in rows or columns
            for (int i = 0; i < squareOrder; i++)
            {
                if (rows[i].Distinct().Count() < squareOrder)
                {
                    error = "Row " + (i+1).ToString() + " doees not contain distinct elemnts.";
                    return false;
                }

                if (cols[i].Distinct().Count() < squareOrder)
                {
                    error = "Column " + (i+1).ToString() + " doees not contain distinct elemnts.";
                    return false;
                }
            }

            error = "";
            return true;
        }

        /// <summary>
        /// Returns a specific element in the square by calculating the offset within
        /// the values list.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="col"></param>
        /// <returns></returns>
        internal int GetElement(int row, int col)
        {
            if (row > squareOrder || col > squareOrder)
                throw new ApplicationException(string.Format("Index {0},{1} not in square of size {2}x{2}.", row, col, squareOrder));
            
            row--;
            col--;
            return values[(row * squareOrder) + col];
        }

        /// <summary>
        /// <para> Override the ToString() method to either print the Latin square to
        /// the screen or to display that the Latin square is empty
        /// </para>
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string ret_string = "";

            if (values.Count == 0)
                return "Empty Latin square.\n";

            for (var i = 0; i < values.Count; i++)
            {
                if (i % squareOrder == 0 && i != 0)
                {
                    ret_string += "\n";
                }

                ret_string += Convert.ToString(values[i]) + " ";
            }

            return ret_string;
        }

    }

    public static class LatinSquareExtensions
    {
        /// <summary>
        /// Fills the Latin square with the values in listOfValues. The list of values should be
        /// "row major". That is, the first n (where n is the order of the square) elements should be the 
        /// entries for row one in the square. The second n should be row two, etc.
        /// </summary>
        /// <param name="ls"></param>
        /// <param name="listOfValues"></param>
        public static void Fill(this LatinSquare ls, List<int> listOfValues)
        {
            ls.SetValues(listOfValues);
        }

        /// <summary>
        /// Returns the order of this Latin square.
        /// </summary>
        /// <param name="ls"></param>
        /// <returns></returns>
        public static int GetOrder(this LatinSquare ls)
        {
            return ls.GetOrder();
        }

        /// <summary>
        /// Determins if this square is mutually orhogonal with a given square.
        /// </summary>
        /// <param name="ls"></param>
        /// <param name="checkSquare"></param>
        public static bool IsOrthogonal(this LatinSquare ls, LatinSquare checkSquare)
        {
            // squares not the same size
            if (ls.GetOrder() != checkSquare.GetOrder())
                return false;

            int rows = ls.GetOrder();
            int cols = rows;
            List<Tuple<int, int>> pairs = new List<Tuple<int, int>>();

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    var currentPair = new Tuple<int, int>(ls.GetElement(i+1, j+1), checkSquare.GetElement(i+1, j+1));
                    if (pairs.Contains(currentPair))
                        return false;

                    pairs.Add(currentPair);
                }
            }

            return true;
        }

        /// <summary>
        /// Returns true if the square belongs to the same isotopy class as the given square.
        /// </summary>
        /// <param name="ls"></param>
        /// <param name="checkSquare"></param>
        /// <returns></returns>
        public static bool IsSameIsotopyClass(this LatinSquare ls, LatinSquare checkSquare)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            // squares not the same size, can't be some isotopy class
            if (ls.GetOrder() != checkSquare.GetOrder())
                return false;

            return false;
        }

        /// <summary>
        /// Returns true if the square belongs to the same main class as the given square.
        /// </summary>
        /// <param name="ls"></param>
        /// <param name="checkSquare"></param>
        /// <returns></returns>
        public static bool IsSameMainClass(this LatinSquare ls, LatinSquare checkSquare)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            // squares not the same size, can't be some isotopy class 
            if (ls.GetOrder() != checkSquare.GetOrder())
                return false;

            return false;
        }

        /// <summary>
        /// Returns the element at a a specific row and column within the Latin square.
        /// </summary>
        /// <param name="ls"></param>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public static int GetElementAtPosition(this LatinSquare ls, int row, int column)
        {
            return ls.GetElement(row, column);
        }

        /// <summary>
        /// Swaps the rows in the Latin square in accordance to the newIndices list of integers.
        /// This list must be the same size as the order of the Latin square. E.g. if the list contains
        /// { 1, 3, 4, 5, 6, 2} (assuming the order of the square is 6) then row 1 doesn't change,
        /// row 2 swaps with row 6, row 3 with row 2, row 4 with row 3, row 5 with row 4, and row 6 
        /// with row 5.
        /// </summary>
        /// <param name="ls"></param>
        /// <param name="newIndices"></param>
        public static void PermuteRows (this LatinSquare ls, List<int> newIndices)
        {
            if (newIndices.Count != ls.GetOrder())
                throw new ApplicationException("Not enough indicies to swap in PermuteRows method.");
            if (newIndices.Select(x => x > ls.GetOrder()).Count() == 0)
                throw new ApplicationException("Invalid index in new indices list in PermuteRows method");

            int squareOrder = ls.GetOrder();

            List<int[]> oldRows = new List<int[]>();
            List<int[]> newRows = new List<int[]>();
            for (int i = 0; i < squareOrder; i++)
            {
                oldRows.Add(new int[squareOrder]);
                newRows.Add(new int[squareOrder]);
            }

            // fill the row and columns lists
            for (int i = 0; i < squareOrder; i++)
                for (int j = 0; j < squareOrder; j++)
                    oldRows[i][j] = ls.GetElementAtPosition(i + 1, j + 1);

            List<int> newVals = new List<int>();
            for (int i = 0; i < squareOrder; i++)
            {
                newRows[i] = oldRows[newIndices[i] - 1];
                newVals = newVals.Concat(newRows[i]).ToList();
            }

            ls.Fill(newVals);
        }

        
    }
}
