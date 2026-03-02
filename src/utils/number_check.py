import sys
import numpy as np
from multiprocessing import Pool
import time

if __name__=='__main__':
    if len(sys.argv) < 3:
        print("Filename and order are required: python3 number_check.py <order> <file>")
        exit()
    
    start = time.time()
    order = int(sys.argv[1])
    filename = sys.argv[2]

    with open(filename) as f:
        length = order**2
        lines = f.readlines()
        totals = {}

        count = 1
        for line in lines:
            if count % 100000 == 0:
                print(f"Processed {count} out of {len(lines) + 1}.")
            tot = 0
            s = 0
            col = 1
            row = 1
            for i in line:
                if i == '\n':
                    continue
                
                col += 1
                s += (int(i) + 1)**(col + row)

                if col == (order + 1):
                    col = 1
                    row += 1
                    tot += np.log(s)
                    s = 0

            if tot not in totals.values():
                totals[line] = tot
            else:
                other = None
                for t in totals:
                    if totals[t] == tot:
                        other = t
                        break
                print(f"\t{tot}\n\t{line}\n\t{other}\n\t{totals[other]}")

            count += 1
    end = time.time()   
    print(f"# Distinct Totals: {len(totals)}\n# Squares: {len(lines)}\nAll Distinct? {len(totals) == len(lines)}\nMax: {max(totals.values())}\
            \nTime Taken: {(end - start):.6f} sec/{((end-start)/60):.6f} min")
