#include "common.h"

int main() 
{
    ifstream sqfile; 
    sqfile.open("5_squares.dat");
    ifstream perm_file;
    perm_file.open("4_perm.dat");

    vector<LatinSquare> squares;
    string line;
    while(getline(sqfile, line)) 
    {
        LatinSquare sq(5, get_array_from_line(line, 25));
        squares.push_back(sq);
    }
    sqfile.close();

    vector<short*> permVec;
    while(getline(perm_file, line)) 
    {
        permVec.push_back(get_array_from_line(line, 4));
    }
    perm_file.close();

    vector<string> all;
    for(auto it = squares.begin(); it != squares.end(); it++) 
    {
        LatinSquare sq = (*it);
        short* perm = new short[5];
        for(auto permIt = permVec.begin(); permIt != permVec.end(); permIt++)
        {
            short* perm4 = (*permIt);
            perm[0] = 0;    // fix first row (normalized)
            for(int i = 1; i < 5; i++)
            {
                perm[i] = perm4[i-1] + 1;
            }
            sq.permute_rows(perm);
            all.push_back(sq.flatstring_no_space());
        }
    }
    cout << all.size() << endl;

    return 0;
}
