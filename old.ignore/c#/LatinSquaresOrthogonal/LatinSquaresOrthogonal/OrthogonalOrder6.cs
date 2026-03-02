using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Combanatorics;
using System.IO;

namespace LatinSquaresOrthogonal
{
    class OrthogonalOrder6
    {
        public static void FindOrthogonalOrder6 (List<LatinSquare> squares)
        {
            Console.WriteLine("Reduced found, count = {0}", squares.Count);

            List<Tuple<int, int>> pairs = new List<Tuple<int, int>>();
            for (int i = 0; i < squares.Count; i++)
            {
                LatinSquare currentSquare = squares[i];
                Console.WriteLine(i);
                for (int j = i; j < squares.Count; j++)
                {
                    if (currentSquare.IsOrthogonal(squares[j]))
                        pairs.Add(new Tuple<int,int>(i,j));
                }
            }

            List<int> distinct = new List<int>();

            foreach (var pair in pairs)
            {
                if (!distinct.Contains(pair.Item1))
                    distinct.Add(pair.Item1);
                if (!distinct.Contains(pair.Item2))
                    distinct.Add(pair.Item2);
            }   

            using (StreamWriter sw = new StreamWriter("orthogonalPairs6.dat"))
            {
                foreach (var pair in pairs)
                {
                    sw.WriteLine("{0} {1}", pair.Item1 + 1, pair.Item2 + 1);
                }
            }

            for (int i = 0; i < distinct.Count; i++)
            {
                if (squares[i].IsNormal())
                    Console.WriteLine("{0}\n", squares[i]);
            }
        }
    }
}
