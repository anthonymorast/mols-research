using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CombanatoricsLibrary
{
    public class LatinSquare
    {
        public LatinSquare()
        {
            return;
        }

        public void PrintSquare (int[][] square)
        {
            if (this.IsLatinSquare(square))
            {
                for (int i = 0; i < this.Size(square).Item1; i++)
                {
                    for (int j = 0; j < this.Size(square).Item1; j++)
                    {
                        Console.Write(square[i][j] + " ");
                    }
                    Console.WriteLine();
                }
            }
            else
                throw new ApplicationException("This is not a Latin Square.");
        }

        public bool IsLatinSquare(int[][] square)
        {
            int[] checkArray = new int[square.Count()];

            for (int i = 0; i < square.Count(); i++ )
            {
                for (int j = 0; j < square.Count(); j++)
                {
                    if (checkArray[square[i][j]%square.Count()] == 1)
                        return false;
                    checkArray[j] = 1;
                }
                checkArray = new int[square.Count()];
            }
            return true;
        }

        public bool IsMutuallyOrthogonal(int[][] square1, int[][] square2)
        {
            return false;
        }

        public Tuple<int,int> Size (int[][] square)
        {
            return new Tuple<int,int>(square.Count(), square.Count());
        }
    }
}
