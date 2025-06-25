#ifndef _XP_COMMON_H_
#define _XP_COMMON_H_

#include <iostream>
#include <cppTypes.h>
#include <math.h>
#include <algorithm>
#include <unistd.h>
#include <ctime>
#include <QString>
#include <QFile>
#include <QList>
#include <QTextStream>
#include <fstream>
#include <string>
#include <iostream>
#define USING_SIM_ESTIMATOR 0
namespace comm
{
    const double pi = 3.14159265358979323946f;
    const float g = 9.80665f;
    const int MOTOR_DISABLE_CMD = 13;


    template<typename T = float>
    Mat3<T> Rot_X(const T& a)
    {
    	Mat3<T> return_v = Mat3<T>::Zero();
    	float cos_a = cos(a);
    	float sin_a = sin(a);
    	return_v(1,2) = -sin_a;
    	return_v(2,1) = sin_a;
    	return_v(0,0) = 1.0f;
    	return_v(1,1) = cos_a;
    	return_v(2,2) = cos_a;
    	return return_v;
    }
    template<typename T = float>
    Mat3<T> Rot_Y(const T& a)
    {
    	Mat3<T> return_v = Mat3<T>::Zero();
    	float cos_a = cos(a);
    	float sin_a = sin(a);

    	return_v(1,1) = 1.0f;
    	return_v(0,2) = sin_a;
    	return_v(2,0) = -sin_a;
    	return_v(0,0) = cos_a;
    	return_v(2,2) = cos_a;

    	return return_v;
    }
    template<typename T = float>
    Mat3<T> Rot_Z(const T& a)
    {
    	Mat3<T> return_v = Mat3<T>::Zero();
    	float cos_a = cos(a);
    	float sin_a = sin(a);
    	return_v(2,2) = 1.0f;
    	return_v(0,0) = cos_a;
    	return_v(1,1) = cos_a;
    	return_v(0,1) = -sin_a;
    	return_v(1,0) = sin_a;


    	return return_v;
    }

    template<typename T = float>
    Mat3<T> Rot_ZYX(const Vec3<T> &a)
    {
        Mat3<T> return_v = Mat3<T>::Zero();
        return_v = Rot_Z(a[2])*Rot_Y(a[1])*Rot_X(a[0]);
        return return_v;
    }
    template <typename T = float>
    void pseudoInverse(const DMat<T>& matrix, DMat<T>& invMatrix,double sigmaThreshold = 0.001)
    {
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
        return;
      }

      Eigen::JacobiSVD<DMat<T>> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
      long const nrows(svd.singularValues().rows());
      DMat<T> invS;
      invS = DMat<T>::Zero(nrows, nrows);
      for(int i = 0; i < nrows; ++i)
      {
        if(svd.singularValues().coeff(i) > sigmaThreshold)
        {
          invS.coeffRef(i, i) = 1.0f / svd.singularValues().coeff(i);
        }
      }
      invMatrix = svd.matrixV() * invS * svd.matrixU().transpose();
    }

    template<typename T>
    T comm_fabs(Vec3<float> v)
    {
        return sqrt(pow(v(0), 2) + pow(v(1), 2) + pow(v(2), 2));
    }

    template<typename T>
    inline const T& limit( const T& x, const T& min_x, const T& max_x )
    {
        return max( min( x, max_x ), min_x );
    }

    inline void print_vec3(char *s,Vec3<float> v) {
        std::cout << "############---" << s << "\n";
        for (int i = 0; i < 3; i++) {
                std::cout << v[i] << "\t";
         }
         std::cout << "\n";
    }
    template<int n>
    inline void print_vec(char *s,Eigen::Matrix<float,n,1> v) {
        std::cout << "############---" << s << "\n";
        for (int i = 0; i < n; i++) {
                std::cout << v[i] << "\t";
         }
         std::cout << "\n";
    }

    template<int m,int n>
    inline void print_mat(char *s,Eigen::Matrix<float,m,n> v) {
        std::cout << "############---" << s << "\n";
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
            {
                std::cout << v(i,j) << "\t";
            }
            std::cout << "\n";
         }
        std::cout << "\n";

    }





    //sign











    template<typename T>
    void convert_leg_pos_and_force(Vec3<T> *in,Vec3<T> *out)
    {
        out[0] = in[1];
        out[1] = in[0];
        out[3] = in[2];
        out[2] = in[3];
    }
    template<typename T>
    void convert_state_phase(Vec4<T> &in,Vec4<T> &out)
    {
        out[0] = in[1];
        out[1] = in[0];
        out[3] = in[2];
        out[2] = in[3];
    }

    template<typename T>
    void convert_joint_ij_full(T *in,T *out)
    {
        for(int i = 0;i<6;i++)
        {
            out[i] = in[i];
        }


        int extra_i = 6;
        out[0+extra_i] = in[3+extra_i];
        out[1+extra_i] = -in[4+extra_i];
        out[2+extra_i] = -in[5+extra_i];

        out[3+extra_i] = in[0+extra_i];
        out[4+extra_i] = -in[1+extra_i];
        out[5+extra_i] = -in[2+extra_i];

        out[6+extra_i] = in[9+extra_i];
        out[7+extra_i] = -in[10+extra_i];
        out[8+extra_i] = -in[11+extra_i];

        out[9+extra_i] = in[6+extra_i];
        out[10+extra_i] = -in[7+extra_i];
        out[11+extra_i] = -in[8+extra_i];

    }

    template<typename T>
    void convert_joint_ij(T *in,T *out)
    {
        int extra_i = 0;

#if _USE_ANKLE || _GOAT
        out[0+extra_i] = in[4+extra_i];
        out[1+extra_i] = -in[5+extra_i];
        out[2+extra_i] = -in[6+extra_i];
        out[3+extra_i] = in[7+extra_i];

        out[4+extra_i] = in[0+extra_i];
        out[5+extra_i] = -in[1+extra_i];
        out[6+extra_i] = -in[2+extra_i];
        out[7+extra_i] = in[3+extra_i];

        out[8+extra_i] = in[12+extra_i];
        out[9+extra_i] = -in[13+extra_i];
        out[10+extra_i] = -in[14+extra_i];
        out[11+extra_i] = in[15+extra_i];

        out[12+extra_i] = in[8+extra_i];
        out[13+extra_i] = -in[9+extra_i];
        out[14+extra_i] = -in[10+extra_i];
        out[15+extra_i] = in[11+extra_i];
#elif _USE_ARM
        out[0+extra_i] = in[3+extra_i];
        out[1+extra_i] = -in[4+extra_i];
        out[2+extra_i] = -in[5+extra_i];

        out[3+extra_i] = in[0+extra_i];
        out[4+extra_i] = -in[1+extra_i];
        out[5+extra_i] = -in[2+extra_i];

        out[6+extra_i] = in[9+extra_i];
        out[7+extra_i] = -in[10+extra_i];
        out[8+extra_i] = -in[11+extra_i];

        out[9+extra_i] = in[6+extra_i];
        out[10+extra_i] = -in[7+extra_i];
        out[11+extra_i] = -in[8+extra_i];

        out[12+extra_i] = in[12+extra_i];
        out[13+extra_i] = in[13+extra_i];
        out[14+extra_i] = in[14+extra_i];
        out[15+extra_i] = in[15+extra_i];
        out[16+extra_i] = in[16+extra_i];
        out[17+extra_i] = in[17+extra_i];
#elif _USE_ARM_Raisim
        out[0+extra_i] = in[3+extra_i];
        out[1+extra_i] = -in[4+extra_i];
        out[2+extra_i] = -in[5+extra_i];

        out[3+extra_i] = in[0+extra_i];
        out[4+extra_i] = -in[1+extra_i];
        out[5+extra_i] = -in[2+extra_i];

        out[6+extra_i] = in[9+extra_i];
        out[7+extra_i] = -in[10+extra_i];
        out[8+extra_i] = -in[11+extra_i];

        out[9+extra_i] = in[6+extra_i];
        out[10+extra_i] = -in[7+extra_i];
        out[11+extra_i] = -in[8+extra_i];

        out[12+extra_i] = in[12+extra_i];
        out[13+extra_i] = in[13+extra_i];
        out[14+extra_i] = in[14+extra_i];
        out[15+extra_i] = in[15+extra_i];
        out[16+extra_i] = in[16+extra_i];
        out[17+extra_i] = in[17+extra_i];
#else

        out[0+extra_i] = in[3+extra_i];
        out[1+extra_i] = -in[4+extra_i];
        out[2+extra_i] = -in[5+extra_i];

        out[3+extra_i] = in[0+extra_i];
        out[4+extra_i] = -in[1+extra_i];
        out[5+extra_i] = -in[2+extra_i];

        out[6+extra_i] = in[9+extra_i];
        out[7+extra_i] = -in[10+extra_i];
        out[8+extra_i] = -in[11+extra_i];

        out[9+extra_i] = in[6+extra_i];
        out[10+extra_i] = -in[7+extra_i];
        out[11+extra_i] = -in[8+extra_i];
#endif


    }
    template<typename T = float>
    int Read_csv_line(QString *filepath, QList<T> *out,int *numrow,int *numcol)
    {
        QStringList csvList;
        csvList.clear();
        int row = 0,col = 0;

        QFile csvFile(*filepath);
        if (csvFile.open(QIODevice::ReadWrite)) //对csv文件进行读写操作
        {
            QTextStream stream(&csvFile);
            while (!stream.atEnd())
            {
                csvList.push_back(stream.readLine()); //保存到List当中
            }
            csvFile.close();
        }
        else
            return -1;
        Q_FOREACH(QString strLine, csvList)   //遍历List
        {
            row++;
            QStringList valsplit = strLine.split(","); //分隔字符串
            col = 0;
            Q_FOREACH(QString strData, valsplit)
            {
                col++;
                out->append((T)strData.toDouble());
            }
        }
        *numrow = row;
        *numcol = col;
        return 1;
    }

    template<typename T = float>
    int read_csv(string* filename,vector<vector<T>> * outArray,int* row,int* col)
    {
        int i = 0;
        int j = 0;
        ifstream inFile(*filename);
        if(!inFile.is_open())
        {
            return -1;
        }
        if(inFile.peek() == EOF)
            return -1;
        string lineStr;
        //vector<vector<string> > strArray;//用来保存读取出来的数据，可以看成是一个二维数组，类型一般是string，其他类型可以转换
        while(getline(inFile,lineStr)) //这里的循环是每次读取一整行的数据,把结果保存在lineStr中，lineStr是用逗号分割开的
        {
            //打印整行字符串
            //cout<<lineStr<<endl;
            //将结果保存为二维表结构
            stringstream ss(lineStr); //这里stringstream是一个字符串流类型，用lineStr来初始化变量 ss
            string str;
            vector<T> lineArray;
            //按照逗号进行分割
            j = 0;
            while(getline(ss,str,',')) //getline每次把按照逗号分割之后的每一个字符串都保存在str中
            {
                lineArray.push_back(atof(str.c_str())); //这里将str保存在lineArray中
                j++;
            }
            i++;
            outArray->push_back(lineArray); //这里把lineArray保存在strArray。   这里的lineArray和lineArray有点类似于python中的list，只是固定了要保存的数据类型
        }
        *row = i;
        *col = j;
        return 1;
    }












}
#endif
