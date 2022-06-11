#include <iostream>
#include <cmath>
#include <cstring>
#include "mpi.h"
using std::cout;
using std::endl;

const int N = 40;
int nx = 41, ny = 41, nt = 5, nit = 50;
double rho = 1.0;
double dx = 2.0 / (static_cast<double>(nx) - 1.0);
double dy = 2.0 / (static_cast<double>(ny) - 1.0);
double dt = .01, nu = .02;

double u[41][41], v[41][41], p[41][41], b[41][41];
double pn[41][41], un[41][41], vn[41][41];

void copy_column(double(*p1)[41][41], int column1, int column2)
{
	for (int i = 0; i < 41; ++i)
	{
		(*p1)[i][column1] = (*p1)[i][column2];
	}
}

void set_values(double(*p1)[41][41], bool isRow, int location, double values)
{
	if (isRow)
	{
		for (int i = 0; i < 41; ++i)
		{
			(*p1)[location][i] = values;
		}
	}
	else
	{
		for (int i = 0; i < 41; ++i)
		{
			(*p1)[i][location] = values;
		}
	}
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int begin = rank * (N / size);
	int end = (rank + 1) * (N / size);
	for (int n = 0; n < nt; ++n)
	{
		for (int j = begin; j < end; ++j)
		{
			for (int i = 1; i < nx - 1; ++i)
			{
				if (j <= 39&&j>=1)
				{
					b[j][i] = rho * (1.0 / dt *
						((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) -
						pow(((u[j][i + 1] - u[j][i - 1]) / (2 * dx)), 2) - 2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) *
							(v[j][i + 1] - v[j][i - 1]) / (2 * dx)) - pow(((v[j + 1][i] - v[j - 1][i]) / (2 * dy)), 2));
				}
			}
		}
		if (rank == 0)
		{
			MPI_Gather(MPI_IN_PLACE, (end - begin) * 41, MPI_DOUBLE, &(b[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Gather(&(b[begin][0]), (end - begin) * 41, MPI_DOUBLE, &(b[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		MPI_Bcast(&(b[0][0]), 40 * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		for (int it = 0; it < nit; ++it)
		{
			::memcpy(pn, p, sizeof(p));
			MPI_Barrier(MPI_COMM_WORLD);
			for (int j = begin; j < end; ++j)
			{
				for (int i = 1; i < nx - 1; ++i)
				{
					if (j <= 39 && j >= 1)
					{
						p[j][i] = (pow(dy, 2) * (pn[j][i + 1] + pn[j][i - 1]) +
							pow(dx, 2) * (pn[j + 1][i] + pn[j - 1][i]) -
							b[j][i] * pow(dx, 2) * pow(dy, 2))
							/ (2 * (pow(dx, 2) + pow(dy, 2)));
					}
				}
			}
			if (rank == 0)
			{
				MPI_Gather(MPI_IN_PLACE, (end - begin) * 41, MPI_DOUBLE, &(p[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			}
			else
			{
				MPI_Gather(&(p[begin][0]), (end - begin) * 41, MPI_DOUBLE, &(p[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			}
			MPI_Bcast(&(p[0][0]), 40 * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
			copy_column(&p, 40, 39);
			::memcpy(p[0], p[1], sizeof(p[1]));
			copy_column(&p, 0, 1);
			set_values(&p, true, 40, 0.0);
		}
		::memcpy(un, u, sizeof(u));
		::memcpy(vn, v, sizeof(v));
		MPI_Barrier(MPI_COMM_WORLD);
		for (int j = begin; j < end; ++j)
		{
			for (int i = 1; i < nx - 1; ++i)
			{
				if (j <= 39 && j >= 1)
				{
					u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
						- un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
						- dt / (2.0 * rho * dx) * (p[j][i + 1] - p[j][i - 1])
						+ nu * dt / pow(dx, 2) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1])
						+ nu * dt / pow(dy, 2) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);

					v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
						- vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
						- dt / (2.0 * rho * dx) * (p[j][i + 1] - p[j][i - 1])
						+ nu * dt / pow(dx, 2) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1])
						+ nu * dt / pow(dy, 2) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
				}
			}
		}
		if (rank == 0)
		{
			MPI_Gather(MPI_IN_PLACE, (end - begin) * 41, MPI_DOUBLE, &(u[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(MPI_IN_PLACE, (end - begin + 1) * 41, MPI_DOUBLE, &(v[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Gather(&(u[begin][0]), (end - begin) * 41, MPI_DOUBLE, &(u[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Gather(&(v[begin][0]), (end - begin) * 41, MPI_DOUBLE, &(v[0][0]), (end - begin) * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}
		MPI_Bcast(&(u[0][0]), 40 * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&(v[0][0]), 40 * 41, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		set_values(&u, true, 0, 0.0);
		set_values(&u, false, 0, 0.0);
		set_values(&u, false, 40, 0.0);
		set_values(&u, true, 40, 1.0);
		set_values(&v, true, 0, 0.0);
		set_values(&v, true, 40, 0.0);
		set_values(&v, false, 0, 0.0);
		set_values(&v, false, 40, 0.0);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (rank == 0)
	{
		for (int j = 0; j < ny; ++j)
		{
			if (j == 39)
			{
				cout << "Line: " << j << endl;
				{
					for (int i = 0; i < nx; ++i)
					{
						cout << u[j][i] << ' ';
					}
					cout << endl;
					for (int i = 0; i < nx; ++i)
					{
						cout << v[j][i] << ' ';
					}
					cout << endl;
				}
			}
		}
	}
	MPI_Finalize();
	return 0;
}
