#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <tuple>

using namespace Eigen;
using namespace std;

const double learning_rate = 2;
const int n_iter = 500;
const int n(784), m_train(60000), m_test(10000), n_digit(10);

Eigen::MatrixXd read_csv(std::string file, int rows, int cols) 
{
	std::ifstream in(file);
	std::string line;

	int row = 0;
	int col = 0;

	Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

	if (in.is_open()) {

		while (std::getline(in, line)) {
			char *ptr = (char *) line.c_str();
			int len = line.length();
			col = 0;
			char *start = ptr;
			for (int i = 0; i < len; i++) {

				if (ptr[i] == ',') {
					res(row, col++) = atof(start);
					start = ptr + i + 1;
				}
			}
			res(row, col) = atof(start);
			row++;
		}
		in.close();
	}
	return res;
}

void remove_row(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
	unsigned int numRows = matrix.rows()-1;
	unsigned int numCols = matrix.cols();

	if( rowToRemove < numRows )
		matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

	matrix.conservativeResize(numRows,numCols);
}

void remove_col(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols()-1;

	if( colToRemove < numCols )
		matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);

	matrix.conservativeResize(numRows,numCols);
}

inline double sigmoid(double z)
{
	return 1.0/(1.0+exp(-z));
} 

VectorXd compute_h(const Eigen::VectorXd& w, const double& b, const Eigen::MatrixXd& X)
{
	Eigen::VectorXd z = X * w;
	z = z.array() + b;
	z = z.unaryExpr(&sigmoid);
	return z; 
}

tuple<double, VectorXd, double> compute_cost(const Eigen::VectorXd& w, const double& b, const Eigen::MatrixXd& X, const Eigen::VectorXd& Y, int digit)
{
	double m = X.rows();

	VectorXd h = compute_h(w, b, X);

	VectorXd Y_tmp = (Y.array() == digit).cast<double>();

	double cost = h.array().log().matrix().dot(Y_tmp);

	cost += (1 - h.array()).log().matrix().dot((1-Y_tmp.array()).matrix());
	cost *= -(1/m);

	VectorXd dw = (1/m)*(X.transpose() * (h - Y_tmp).array().matrix());
	double db = (1/m)*(h - Y_tmp).sum();

	return make_tuple(cost, dw, db);
}


VectorXd prediction_max(MatrixXd pred)
{
	VectorXd p(pred.rows(),1);
	Eigen::MatrixXd::Index max_index;
	for (int i(0); i < pred.rows(); ++i) {
		pred.row(i).maxCoeff(&max_index);
		p(i) = max_index;
	}
	return p;
}

int main(int argc, const char * argv[]) 
{
	cout << "Read the data" << endl;

	MatrixXd X_train = read_csv("mnist_train.csv", m_train, n+1);
	MatrixXd X_test = read_csv("mnist_test.csv", m_test, n+1);

	VectorXd Y_train = X_train.col(0);
	VectorXd Y_test = X_test.col(0);

	remove_col(X_train, 0);
	remove_col(X_test, 0);

	// max normalize
	X_train *= (1.0/255.0);
	X_test *= (1.0/255.0);

	MatrixXd probabilities_train(m_train, n_digit);
	MatrixXd probabilities_test(m_test, n_digit);

	cout << "===============================" << endl;
	cout << "X_train rows : " << X_train.rows() << endl;	
	cout << "X_train cols : " << X_train.cols() << endl;	
	cout << "Y_train rows : " << Y_train.rows() << endl;
	cout << "Y_train cols : " << Y_train.cols() << endl;
	cout << "===============================" << endl;
	cout << "X_test rows  : " << X_test.rows() << endl;	
	cout << "X_test cols  : " << X_test.cols() << endl;	
	cout << "Y_test rows  : " << Y_test.rows() << endl;
	cout << "Y_test cols  : " << Y_test.cols() << endl;
	cout << "===============================" << endl;

	cout << "Start Gradient Descent" << endl;

	cout << "===============================" << endl;


	#pragma omp parallel for
	for (int d=0; d < n_digit; ++d)
	{
		double cost(0);
		VectorXd w = VectorXd::Zero(n,1);
		double b(0);
		VectorXd dw = VectorXd::Zero(n,1);
		double db(0);

		/*
		   cout << "===============================" << endl;
		   cout << "Computing weights and bias for digit : " << d << endl;
		   cout << "===============================" << endl;
		   cout << "iter cost" << endl;
		 */

		for (int i(0); i <= n_iter; ++i) 
		{
			tie(cost, dw, db) = compute_cost(w, b, X_train, Y_train, d);
			w -= learning_rate * dw;
			b -= learning_rate * db;

			/*
			   if (i % 5 == 0)
			   cout << i << " " << cost << endl; 
			 */
		}
		probabilities_train.col(d) = compute_h(w, b, X_train);
		probabilities_test.col(d) = compute_h(w, b, X_test);
	}


	VectorXd pred_train = prediction_max(probabilities_train);	
	VectorXd pred_test = prediction_max(probabilities_test);	

	cout << "Training accuracy : " << (Y_train.array() == pred_train.array()).cast<double>().sum()/static_cast<double>(m_train) << endl;
	cout << "Test accuracy : " << (Y_test.array() == pred_test.array()).cast<double>().sum()/static_cast<double>(m_test) << endl;

	return 0;
}



