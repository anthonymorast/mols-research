import sys
import numpy as np
from multiprocessing import Pool
import time

def get_value(square):
    tot = 0
    s = 0
    col = 1
    row = 1
    for i in square:
        if i == '\n':
            continue
        
        col += 1
        s += (int(i) + 1)**(col + row)

        if col == (order + 1):
            col = 1
            row += 1
            tot += np.log(s)
            s = 0

    return square, tot

def build_dict(results):
    totals = {}
    for result in results:
        sq, tot = result
        if tot not in totals.values():
            totals[sq] = tot
        else:
            other = None
            for t in totals:
                if totals[t] == tot:
                    other = t
                    break
            print(f"\t{tot}\n\t{sq}\n\t{other}\n\t{totals[other]}")
    return totals
        

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

        processes = 20
        with Pool(processes) as p:
            print(f"Determining square values.")
            results = p.map(get_value, lines)
            print(f"Aggregating results.")
            values_per_list = int(len(results)/processes)
            extra = len(results) - (values_per_list*processes)
            result_lists = []
            for i in range(processes-1):
                result_lists.append(results[(values_per_list*i):(values_per_list*(i+1))])
            result_lists.append(results[(processes-1)*values_per_list:])

            union_dicts = p.map(build_dict, result_lists)
            for d in union_dicts:
                totals = {**totals, **d}

    end = time.time()   
    print(f"# Distinct Totals: {len(totals)}\n# Squares: {len(lines)}\nAll Distinct? {len(totals) == len(lines)}\nMax: {max(totals.values())}\
            \nTime Taken: {(end - start):.6f} sec/{((end-start)/60):.6f} min")
