fid = fopen('4_squares.dat');

order = 4;
dets = [];
while !feof(fid)
    sq = num2str(fgetl(fid));    
    mat_sq = [];
    for i=0:order-1
      values = [];
      for j=0:order-1
        values = [values, str2num(sq((i*order + j)+1))];
      endfor
      mat_sq = [mat_sq; values];
    endfor
    #display(mat_sq);
    #display(det(mat_sq));
    
    dets = [dets, det(mat_sq)];
  end
  
  display(dets);