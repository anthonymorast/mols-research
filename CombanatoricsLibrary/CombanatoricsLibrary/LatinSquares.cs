using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CombanatoricsLibrary
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
        /// Create a Latin square of size n x n 
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
        public int GetOrder()
        {
            return this.squareOrder;
        }

        /// <summary>
        /// Set the values of the Latin square.
        /// </summary>
        /// <param name="valueList"></param>
        /// <returns></returns>
        public void SetValues(List<int> valueList)
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
        private bool CheckValues(List<int> valueList, out string error)
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
    }
}
