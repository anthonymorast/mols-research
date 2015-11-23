using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Combanatorics;
using System.IO;
using System.Threading;

namespace LatinSquaresOrthogonal
{
    class OrthogonalOrder6
    {
        public static void FindOrthogonalOrder6 (List<LatinSquare> squares)
        {
            Console.WriteLine("Reduced found, count = {0}", squares.Count);
            
            // create threads
            List<Tuple<int,int>> thread1Pairs = new List<Tuple<int,int>>();
            List<LatinSquare> thread1squares = new List<LatinSquare>(); // squares 0 - 51,960
            int thread1Start = 0;
            Thread thread1 = new Thread(() => { thread1Pairs = GetLocalPairs(squares, thread1squares, thread1Start); });

            List<Tuple<int, int>> thread2Pairs = new List<Tuple<int, int>>(); 
            List<LatinSquare> thread2squares = new List<LatinSquare>();  // squares 51,961 - 110,961
            int thread2Start = 51961;
            Thread thread2 = new Thread(() => { thread2Pairs = GetLocalPairs(squares, thread2squares, thread2Start); });

            List<Tuple<int, int>> thread3Pairs = new List<Tuple<int, int>>();
            List<LatinSquare> thread3squares = new List<LatinSquare>(); // squares 110,962 - 179,961
            int thread3tart = 110962;
            Thread thread3 = new Thread(() => { thread3Pairs = GetLocalPairs(squares, thread3squares, thread3tart); });

            List<Tuple<int, int>> thread4Pairs = new List<Tuple<int, int>>();
            List<LatinSquare> thread4squares = new List<LatinSquare>(); // squares 179,962 - 262,962
            int thread4Start = 179962;
            Thread thread4 = new Thread(() => { thread4Pairs = GetLocalPairs(squares, thread4squares, thread4Start); });

            List<Tuple<int, int>> thread5Pairs = new List<Tuple<int, int>>();
            List<LatinSquare> thread5squares = new List<LatinSquare>();  // squares 262,963 - 366,963
            int thread5Start = 262963;
            Thread thread5 = new Thread(() => { thread5Pairs = GetLocalPairs(squares, thread5squares, thread5Start); });

            List<Tuple<int, int>> thread6Pairs = new List<Tuple<int, int>>();
            List<LatinSquare> thread6squares = new List<LatinSquare>();  // squares 366,964 - 504,964
            int thread6Start = 366964;
            Thread thread6 = new Thread(() => { thread6Pairs = GetLocalPairs(squares, thread6squares, thread6Start); });

            List<Tuple<int, int>> thread7Pairs = new List<Tuple<int, int>>();
            List<LatinSquare> thread7squares = new List<LatinSquare>();  // squares 504,965 - 712,965
            int thread7Start = 504965;
            Thread thread7 = new Thread(() => { thread7Pairs = GetLocalPairs(squares, thread7squares, thread7Start); });

            List<Tuple<int, int>> thread8Pairs = new List<Tuple<int, int>>();
            List<LatinSquare> thread8squares = new List<LatinSquare>();  // squares 712,966 - 1128960 
            int thread8Start = 712966;
            Thread thread8 = new Thread(() => { thread8Pairs = GetLocalPairs(squares, thread8squares, thread8Start); });

            // fill thread square lists
            for (int i = 0; i < squares.Count; i++)
            {
                LatinSquare curr = squares[i];
                if (i <= 51960)
                    thread1squares.Add(curr);
                else if (i <= 110961)
                    thread2squares.Add(curr);
                else if (i <= 179961)
                    thread3squares.Add(curr);
                else if (i <= 262962)
                    thread4squares.Add(curr);
                else if (i <= 366963)
                    thread5squares.Add(curr);
                else if (i <= 504964)
                    thread6squares.Add(curr);
                else if (i <= 712965)
                    thread7squares.Add(curr);
                else
                    thread8squares.Add(curr);
            }

            // start and join threads
            List<Thread> threads = new List<Thread> { thread1, thread2, thread3, thread4, thread5, thread6, thread7, thread8 };
            foreach (var thread in threads)
                thread.Start();

            foreach (var thread in threads)
                thread.Join();

            List<Tuple<int, int>> pairs = new List<Tuple<int, int>>();
            //for (int i = 0; i < squares.Count; i++)
            //{
            //    LatinSquare currentSquare = squares[i];
            //    Console.WriteLine(i);
            //    for (int j = i; j < squares.Count; j++)
            //    {
            //        if (currentSquare.IsOrthogonal(squares[j]))
            //            pairs.Add(new Tuple<int,int>(i,j));
            //    }
            //}

            Console.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7}", thread1Pairs.Count, thread2Pairs.Count, thread3Pairs.Count, thread4Pairs.Count, thread5Pairs.Count,
                thread6Pairs.Count, thread7Pairs.Count, thread8Pairs.Count);

            pairs.Concat(thread1Pairs);
            pairs.Concat(thread2Pairs);
            pairs.Concat(thread3Pairs);
            pairs.Concat(thread4Pairs);
            pairs.Concat(thread5Pairs);
            pairs.Concat(thread6Pairs);
            pairs.Concat(thread7Pairs);
            pairs.Concat(thread8Pairs);

            Console.WriteLine(pairs.Count);

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

        public static List<Tuple<int,int>> GetLocalPairs (List<LatinSquare> squares, List<LatinSquare> mySquares, int startIndex)
        {
            // go through mysquares list and check orthogonailty with all squares in squares, can start at beginning index of mysquares
            // that is, if my squares are squares 222,000 thru 444,000 then start comparing and squares[222000]
            List<Tuple<int, int>> localPairs = new List<Tuple<int, int>>();

            for (int i = 0; i < mySquares.Count; i++)
            {
                Console.WriteLine(i);
                for (int j = i + startIndex; j < squares.Count; j++)
                {
                    if (mySquares[i].IsOrthogonal(squares[j]))
                        localPairs.Add(new Tuple<int, int>(i, j));
                }
            }
                return localPairs;
        }
    }
}
