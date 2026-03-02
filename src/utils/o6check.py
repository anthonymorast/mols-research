import numpy as np

sq1 = "012345253104345012430251124530501423"
sq2 = "012345503124345012430251124530251403"

order = 6
tot = 0
s = 0
col = 1
row = 1
for i in sq2:
    if i == '\n':
        continue
    
    col += 1
    s += (int(i) + 1)**(col + row)

    if col == (order + 1):
        print(f"{row}={s}")
        col = 1
        row += 1
        tot += np.log(s)
        s = 0