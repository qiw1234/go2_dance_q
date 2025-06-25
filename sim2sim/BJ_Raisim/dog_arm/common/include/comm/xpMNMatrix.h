
#ifndef _XP_MATRIX_MXN_TEMP_H_
#define _XP_MATRIX_MXN_TEMP_H_
#include<iostream>
#include <cstring>
#include <math.h>
//#include <qpOASES.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
//using namespace qpOASES;
using namespace std;
template<typename T,int R,int C>
class xpMNMatrix
{   
public:
	xpMNMatrix(){memset(data,0,sizeof(data));}
	xpMNMatrix(const xpMNMatrix &M)
	{
		for(int i=0;i<R;i++)
			for(int j=0;j<C;j++)
				data[i][j]=M.data[i][j];
	}
	xpMNMatrix(const T &v)
	{
		for(int i=0;i<R;i++)
			for(int j=0;j<C;j++)
				data[i][j]=v;
	}
    xpMNMatrix(const T v1,const T v2,const T v3)
    {
        data[0][0]=v1;
        data[1][0]=v2;
        data[2][0]=v3;
    }
    xpMNMatrix(const T v1,const T v2,const T v3,const T v4)
    {
        data[0][0]=v1;
        data[1][0]=v2;
        data[2][0]=v3;
        data[3][0]=v4;
    }
	xpMNMatrix operator+(const xpMNMatrix &M)
	{ 
		xpMNMatrix o;
		for(int i=0;i<R;i++)
		{
			for(int j=0;j<C;j++)
			{
				o.data[i][j]=data[i][j] + M.data[i][j];
				//data[i][j] = o.data[i][j];
			}
		}
		return (o); 
	}
    xpMNMatrix operator+=(const xpMNMatrix &M)
    {
        for(int i=0;i<R;i++)
        {
            for(int j=0;j<C;j++)
            {
                data[i][j]=data[i][j] + M.data[i][j];
            }
        }
        return (*this);
    }
    inline T& operator()(int i,int j)
	{
		return data[i][j];
	}
    inline T& operator()(int i)
	{
		return data[i][0];
	}
    inline T& operator[](int i)
	{
		return data[i][0];
	}
    xpMNMatrix cross(const xpMNMatrix &M)
    {
        //a=(X1,Y1,Z1),b=(X2,Y2,Z2),
        //a×b=（Y1Z2-Y2Z1,Z1X2-Z2X1,X1Y2-X2Y1）
         xpMNMatrix o(0.0f);
         o.data[0][0] = data[1][0]*M.data[2][0]-M.data[1][0]*data[2][0];
         o.data[1][0] = data[2][0]*M.data[0][0]-M.data[2][0]*data[0][0];
         o.data[2][0] = data[0][0]*M.data[1][0]-M.data[0][0]*data[1][0];
         return o;
    }

	xpMNMatrix operator-(const xpMNMatrix &M)
	{
		xpMNMatrix o;
		for(int i=0;i<R;i++)
		{
			for(int j=0;j<C;j++)
			{
				o.data[i][j]=data[i][j] - M.data[i][j];
			}
		}          
		return o;  
	}
	xpMNMatrix identity()
	{
		xpMNMatrix o(0.0);
		for (int i = 0; i<R; i++)
		{
			for (int j = 0; j<C; j++)
			{
				if(i==j)
					o.data[i][j] = 1.0f;
			}
		}

		return o;
	}
    inline T length()
    {
        T sum = 0.0f;
        for (int i = 0; i < R; i++){
            sum+=sqrt(data[i][0]*data[i][0]);
        }

        return sum;
    }
	template<int N>
	xpMNMatrix<T,R,N> operator*(const xpMNMatrix<T,C,N> &M)
	{
		xpMNMatrix<T,R,N> o;  
		for (int i = 0; i < R; i++){  
			for (int j = 0; j < N; j++){  
				o.data[i][j] = 0;  
				for (int k = 0; k < C; k++){  
					o.data[i][j] += data[i][k] * M.data[k][j];  
				} 

			}  
		}          
		return o; 
	} 
	template<int row,int col>
	xpMNMatrix<T, row, col> block(int from_i,int from_j)
	{
		xpMNMatrix<T, row, col> o;
		for (int i = from_i; i < row+ from_i; i++) {
			for (int j = from_j; j < col+ from_j; j++) {
					o.data[i][j] += data[i+ from_i][j+ from_j];
			}
		}
		return o;
	}
	template<int row, int col>
	void insert(int from_i, int from_j, xpMNMatrix<T, row, col> insert_mt)
	{
		for (int i = from_i; i < row + from_i; i++) {
			for (int j = from_j; j < col + from_j; j++) {
				(*this).data[i][j] = insert_mt.data[i- from_i][j- from_j];
			}
		}
	}



	xpMNMatrix operator*(const T &v)
	{
		xpMNMatrix o;  
		for (int i = 0; i < R; i++){  
			for (int j = 0; j < C; j++){  
                o.data[i][j] = data[i][j]*(T)v;
			}  
		}          
		return o; 
	} 
	xpMNMatrix operator/(const T &v)
	{
		xpMNMatrix o;  
		for (int i = 0; i < R; i++){  
			for (int j = 0; j < C; j++){  
				o.data[i][j] = data[i][j]/v;                 
			}  
		}          
		return o; 
	} 
	xpMNMatrix& operator=(const xpMNMatrix &M)
	{ 
		for (int i = 0; i < R; i++){  
			for (int j = 0; j < C; j++){  
				data[i][j] = M.data[i][j];  
			}  
		}  

		return (*this); 
	}

	xpMNMatrix<T,C,R> transpose()
	{
		xpMNMatrix<T,C,R> o;
		for(int i=0;i<R;i++)
			for(int j=0;j<C;j++)
				o.data[j][i]=data[i][j];
		return o;  
	}
    template<int row, int col>
    void copy_to_eigen(xpMNMatrix<T,row,col> &src,Eigen::Matrix<T,row,col> &dst) {
        for(int i=0;i<row;i++)
            for(int j=0;j<col;j++)
                dst(i,j)=src.data[i][j];
    }
    template<int row, int col>
    void copy_to_MNMatrix(Eigen::Matrix<T,row,col> &src,xpMNMatrix<T,row,col> &dst) {
        for(int i=0;i<row;i++)
            for(int j=0;j<col;j++)
                dst.data[i][j] = src(i,j);
    }
    template<int row, int col>
    void copy_to_DynamicMNMatrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &src,xpMNMatrix<T,row,col> &dst) {
        for(int i=0;i<row;i++)
            for(int j=0;j<col;j++)
                dst.data[i][j] = src(i,j);
    }
#ifndef __USE_SINGLE_PRECISION__
    void convert_to_real_t(double* target,int nRows,
        int nCols) {
        int count = 0;

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                target[count] = data[i][j];
                count++;
            }
        }
    }
    void copy_real_t_to_mt(double* source) {
        for (int i = 0; i < R; i++) {
            data[i][0] = source[i];
        }
    }
#else
    void convert_to_real_t(float* target,int nRows,
        int nCols) {
        int count = 0;

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                target[count] = data[i][j];
                count++;
            }
        }
    }
    void copy_real_t_to_mt(float* source) {
        for (int i = 0; i < R; i++) {
            data[i][0] = source[i];
        }
    }
#endif
    xpMNMatrix<T,C,R> inverse()
    {
        Eigen::Matrix<T,R,C> o;
        Eigen::Matrix<T,C,R> o_inv;
        xpMNMatrix<T,C,R> result;
        copy_to_eigen((*this),o);
        o_inv = o.inverse();
        copy_to_MNMatrix(o_inv,result);
        return result;
    }
    xpMNMatrix<T,C,R> pseudoInverse(
                       double sigmaThreshold = 0.001)
    {
      xpMNMatrix<T,C,R> result;
      Eigen::Matrix<T,R,C> o;
      copy_to_eigen((*this),o);
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(o);

      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> invMatrix;
      if(matrix.rows() == 1 && matrix.cols() == 1)
      {
        invMatrix.resize(1, 1);
        if(matrix.coeff(0, 0) > sigmaThreshold)
        {
          invMatrix.coeffRef(0, 0) = 1.0f / matrix.coeff(0, 0);
        }
        else
        {
          invMatrix.coeffRef(0, 0) = 0.0;
        }
        copy_to_DynamicMNMatrix(invMatrix,result);
        return result;
      }

      Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
      long const nrows(svd.singularValues().rows());
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> invS;
      invS = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nrows, nrows);
      for(int i = 0; i < nrows; ++i)
      {
        if(svd.singularValues().coeff(i) > sigmaThreshold)
        {
          invS.coeffRef(i, i) = 1.0f / svd.singularValues().coeff(i);
        }
      }
      invMatrix = svd.matrixV() * invS * svd.matrixU().transpose();
      copy_to_DynamicMNMatrix(invMatrix,result);
      return result;
    }
	void print(const std::string &name)
	{
		printf("%s_Matrix = ######################\n",name.c_str());
		for(int i=0;i<R;i++)
		{
			for(int j=0;j<C;j++)
			{
				printf("%f ",data[i][j]);
			}
			printf("\n");
		}
	}
	~xpMNMatrix(){}
	T data[R][C];
};
#endif

