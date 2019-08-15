#include "Utils.h"

int main(int argc, char* argv[]) 
{
	string selection;
	int int_sel;
	while(true)
	{
		print_menu();
		cout << endl << "Select an option: ";
		cin >> selection;
		int_sel = stoi(selection);

		switch(int_sel)
		{
			case 1:
				permutation_menu();
				break;
			case 2:
				convert_mckay_menu();
				break;
			case 3:
				return 0;
			default:
				cout << "Invalid option." << endl;
				break;
		}
		cout << endl;
	}
	return 0;
}

void permutation_menu() 
{
	int n;
	cout << "Number of values to permute: ";
	cin >> n;
	permutations(n);
}

void convert_mckay_menu() 
{
	string filename; 
	cout << "Enter filename of squares in McKay format (1 per line): ";
   	cin >> filename;
	convert_mckay(filename);
}

void convert_mckay(string filename) 
{
	if(!file_exists(filename)) 
	{
		cout << "ERROR: file " << filename << " does not exists." << endl;
		return;
	}

	string line;
	ifstream in;
	in.open(filename);
	ofstream out;
	out.open(filename + ".new");
	while(getline(in, line)) 
	{
		istringstream ss(line);
		int size = line.length();
		const char* linearr = line.c_str();
		for(int i = 0; i < size; i++) 
		{
			out << linearr[i] << " ";
		}
		out << endl;
	}

	cout << "Successfully created file " << (filename + ".new") << endl;
	return;
}

void print_menu() 
{
	cout << "1. Create permutation file" << endl;
	cout << "2. Convert file from McKay site format" << endl;
	cout << "3. Exit" << endl;
}

void permutations(short n) 
{
	cout << endl << "Creating permutations of 0..." << to_string(n) << endl;
	cout << "Writing to file " << to_string(n) << "_perm.dat" << endl;
	if(n > 9) 
		cout << "This may take some time..." << endl;
	short *vals = new short[n];
	for(short i = 0; i < n; i++) 
	{
		vals[i] = i;
	}
	
	ofstream out;
	out.open(to_string(n) + "_perm.dat");
	if(!out.is_open())
	{
		cout << "Error opening permutations file." << endl;
		return;
	}
	create_permutations_file(vals, n, n, out);
	cout << "Done." << endl;
}

void create_permutations_file(short *vals, short size, short n, ofstream &out)
{
	if(size == 1) 
	{
		print_arr(vals, n, out);
		return;
	}

	for(short i = 0; i < size; i++) 
	{
		create_permutations_file(vals, size-1, n, out);
		if(size % 2 == 1)
			swap(vals[0], vals[size-1]);
		else
			swap(vals[i], vals[size-1]);
	}
}

void print_arr(short *vals, short size, ofstream &out)
{
	for(short i = 0; i < size; i++)
		out << vals[i] << " ";
	out << endl;
}
