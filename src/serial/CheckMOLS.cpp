#include "../CheckMOLS.h"

int main(int argc, char* argv[])
{

	if (argc < 2)
	{
		print_usage();
		return 0;
	}

	short order = stoi(string(argv[1]));
	string filename_norm = string(argv[2]);
	string filename_nmo = to_string(order-1) + "_perm.dat";
	bool cont = true;

   	if (!file_exists(filename_nmo))
	{
		cout << filename_nmo << " does not exist. Please use the utilites to generate the file." << endl;
		cont = false;
	}
	if(!file_exists(filename_norm))
	{
		cout << filename_norm << " does not exist." << endl;
		cont = false;
	}

	if (!cont)
		return 0;

	ifstream normfile; normfile.open(filename_norm);
	string line;
	unordered_set<string> allSqs;
	while(getline(normfile, line))
	{
		LatinSquare sq(order, get_array_from_line(line, order*order));
		allSqs.insert(sq.flatstring_no_space());
	}
	normfile.close();

    cout << allSqs.size() << endl; 

    return 0;
}
