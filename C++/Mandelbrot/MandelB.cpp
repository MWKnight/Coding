#include <iostream>
#include <complex>
#include <fstream>

using namespace std;

int Mandelbrot(int max_iter, int max_val, complex<double> c);

int main(int argc, char* argv[]) {

	const complex<double> i(0.0, 1.0);
	
	int max_iter = 150;
	int max_val = 2;
	int x_size = stoi(argv[1]);
	int y_size = stoi(argv[2]);
	double x_min = atof(argv[3]);
	double x_max = atof(argv[4]);
	double y_min = atof(argv[5]);
	double y_max = atof(argv[6]);
	double h_x = (x_max - x_min)/ x_size;
	double h_y = (y_max - y_min)/ y_size;
	complex<double> val;
	int m_Mat[x_size][y_size];
	
	for(int k = 0; k< x_size; k++) {
		for(int j = 0; j < y_size; j++) {
			
			val = x_min + k*h_x + i*(y_min + j*h_y);

			m_Mat[j][k] = Mandelbrot(max_iter, max_val, val);

		}
		
	}

	ofstream myfile ("data.dat");

	if (myfile.is_open()) {

		for (int l = 0; l < y_size; l++) {

			for (int m = 0; m < x_size; m++) {

				myfile << m_Mat[l][m] << ", ";

			}

			myfile << "\n";

		}
	}
	
	return 0;
}

int Mandelbrot(int max_iter, int max_val, complex<double> c) {
	
	const complex<double> i(0.0, 1.0);
	
	complex<double> z = c;

	for(int i = 1; i < max_iter; i++) {
		
		z = z*z + c;

		if(abs(z) > max_val) {
			return max_iter - i;
		}	
	}

	return 0;
}
