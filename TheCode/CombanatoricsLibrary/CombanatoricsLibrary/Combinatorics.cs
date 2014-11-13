using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;

namespace CombanatoricsLibrary
{
    /// <summary>
    /// This class will implement many common operations used in combinatorics. These will be
    /// similar to combinations and permutations, producing all combinations of a certain number,
    /// etc. 
    /// </summary>
    public class Combinatorics
    {
        /// <summary>
        /// This method takes in an integer and produces all the possible permutations of the integer.
        /// For example, passing in '4' will produce a List of Arrays of the format enty one = 1,2,3,4 
        /// entry two = 1,2,4,3; etc.
        /// </summary>
        /// <param name="order">The of which to find permutations</param>
        public static List<long[]> ProducePermutations(long order)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            return new List<long[]>();
        }

        /// <summary>
        /// This method takes in an integer and produces all the possible permutations of the integer.
        /// For example, passing in '4' will produce a List of Arrays of the format enty one = 1,2,3,4 
        /// entry two = 1,2,4,3; etc. It will also write these values out to a file.
        /// </summary>
        /// <param name="order"> The of which to find permutations</param>
        /// <param name="filename"> THe file path with filename to output permutations.</param>
        /// <returns></returns>
        public static List<long[]> ProducePermutations (long order, string filename)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            return new List<long[]>();
        }

        public static List<long[]> ProduceCombinations (long order)
        {
            throw new NotImplementedException("Implementation saved for future release.");

            return new List<long[]>();
        }

        /// <summary>
        /// Returns the number of permutations of a given set of numbers, n Pick k.
        /// For values larger than approximately P(50, 40) use the BigInteger in the 
        /// Systems.Numerics and the BigInterger implementation of this algorithm in this
        /// library, otherwise an overflow will occur.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        public static long Permutations(long n, long k)
        {
            if (k > n)
                throw new ApplicationException(string.Format(
                        "Cannot find number of permutations for {0} pick {1}, k > n.", n, k));

            if (k == 0)
                return 1;
            else if (k == 1)
                return n;

            long result = 1;
            long stop = (n - k);

            for (long i = n; i > stop; i--)
                result *= i;
            Console.WriteLine("n = {0} k = {1} i = {2}", n, k, result);

            return result;
        }


        
        /// <summary>
        /// The BigInteger implementation to find all permutations of a given set of numbers, n pick k.
        /// This implementation should be used if very large values, values larger than 1.9e+19, are
        /// expected, as this will overflow a 64 bit integer.
        /// </summary>
        /// <param name="n"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        //public static BigInteger Permutations(int n, int k)
        //{
        //    throw new NotImplementedException("This method will be added in a future release.");
        //    if (k > n)
        //        throw new ApplicationException(string.Format(
        //                "Cannot find number of permutations for {0} pick {1}, k > n.", n, k));

        //    if (k == 0)
        //        return 1;
        //    else if (k == 1)
        //        return n;

        //    BigInteger result = 0;

        //    return result;
        //}

        /// <summary>
        /// Returns the total number of combinations of a given set of numbers, n Choose k.
        /// For values larger than approximately 50 C 40 use the BigInteger in the 
        /// Systems.Numerics and the BigInterger implementation of this algorithm in this
        /// library, otherwise an overflow will occur.
        /// </summary>
        /// <param name="order"></param>
        /// <returns></returns>
        public static long Combinations(long n, long k)
        {
            if (k > n)
                throw new ApplicationException(string.Format(
                          "Cannot find number of combinations for {0} choose {1}, k > n.", n, k));
            if (k == 0)
                return 1;
            else if (k == 1)
                return n;

            long result = 1;  

            for (long d = 1; d <= k; d++)
            {
                result *= n--;
                result /= d;
            }

            return result;
        }

        /// <summary>
        /// The BigInteger implementation to find all combinations of a given set of numbers, n choose k.
        /// This implementation should be used if very large values, values larger than 1.9e+19, are
        /// expected, as this will overflow a 64 bit integer.
        /// </summary>
        /// <param name="n"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        //public static BigInteger Combinations(int n, int k)
        //{
        //    throw new NotImplementedException("This method will be added in a future release.");
        //    if (k > n)
        //        throw new ApplicationException(string.Format(
        //                "Cannot find number of combinations for {0} pick {1}, k > n.", n, k));

        //    if (k == 0)
        //        return 1;
        //    else if (k == 1)
        //        return n;

        //    BigInteger result = 0;

        //    return result;
        //}
    }
}
