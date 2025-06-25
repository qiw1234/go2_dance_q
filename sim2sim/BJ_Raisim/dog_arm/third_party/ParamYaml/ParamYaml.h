#ifndef PARAM_YAML_H
#define PARAM_YAML_H

#include <dynacore_yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <assert.h>
//#include <cppTypes.h>
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/include/eigen3/Eigen/Geometry>
#define YAML_PROTECTED 0
#define YAML_PATH std::string("config/")

using namespace std;

class xpYaml {
public:
    xpYaml(const std::string &yaml_name)
    {
//        yaml_path.clear();
//        config_.reset();
//        yaml_path.append(YAML_PATH);
//        yaml_path.append(yaml_name);
//        yaml_path.append(".yaml");
        //access(yaml_path, F_OK);
        //printf("Yaml Init\n");
        try
        {
            config_ = dynacore_YAML::LoadFile(yaml_name);
            fileLoaded = true;
        }
        catch(dynacore_YAML::BadFile &e)
        {
            printf("LoadFile error:%s\n",e.msg.c_str());
            fileLoaded = false;
        #if YAML_PROTECTED
            exit(0);
        #endif
        }
        //printf("Yaml Init Success\n");
    }

    virtual ~xpYaml()
    {
        yaml_path.clear();
        config_.reset();
    }

    template <typename T>
    void write(const std::string node, T val)
    {
        if(!config_[node])//find node
        {
            //printf("Not Find [%s]\n", node);
            cout << "Not Find [" << node << "]" << endl;
            exit(0);
        }
        config_[node] = val;
        std::ofstream file(yaml_path);
        file << config_ << std::endl;
    }

    template <typename T>
    void read(const std::string node, T& val)
    {

        if(!config_[node])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s]\n", node);
            exit(0);
        #else
            config_[node] = T(0);
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
            //printf("Not Find [%s]\n", node);
        #endif
        }
        val = config_[node].as<T>();
        //printf("val = %x\n", val);
    }

    template <typename T>
    void read(const std::string node, Eigen::Matrix<T, 2, 1>& val)
    {
        if(!config_[node])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s]\n", node);
            exit(0);
        #else
            std::vector<T> v = {0,0};
            config_[node] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 2; i++)
        {
            val(i) = config_[node][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, Eigen::Matrix<T, 3, 1>& val)
    {
        if(!config_[node])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s]\n", node);
            exit(0);
        #else
            std::vector<T> v = {0,0,0};
            config_[node] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 3; i++)
        {
            val(i) = config_[node][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, Eigen::Matrix<T, 4, 1>& val)
    {
        if(!config_[node])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s]\n", node);
            exit(0);
        #else
            std::vector<T> v = {0,1,2,3};
            config_[node] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 4; i++)
        {
            val(i) = config_[node][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, Eigen::Matrix<T, 6, 1>& val)
    {
        if(!config_[node])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s]\n", node);
            exit(0);
        #else
            std::vector<T> v = {0,1,2,3,4,5};
            config_[node] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 6; i++)
        {
            val(i) = config_[node][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, const std::string member, T& val)
    {
        if(!config_[node][member])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s][%s]\n", node, member);
            exit(0);
        #else
            config_[node][member] = T(0);
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        val = config_[node][member].as<T>();
    }

    template <typename T>
    void read(const std::string node, const std::string member, Eigen::Matrix<T, 2, 1>& val)
    {
        if(!config_[node][member])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s][%s]\n", node, member);
            exit(0);
        #else
            std::vector<T> v = {0,0};
            config_[node][member] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 2; i++)
        {
            val(i) = config_[node][member][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, const std::string member, Eigen::Matrix<T, 3, 1>& val)
    {
        if(!config_[node][member])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s][%s]\n", node, member);
            exit(0);
        #else
            std::vector<T> v = {0,0,0};
            config_[node][member] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 3; i++)
        {
            val(i) = config_[node][member][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, const std::string member, Eigen::Matrix<T, 4, 1>& val)
    {
        if(!config_[node][member])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s][%s]\n", node, member);
            exit(0);
        #else
            std::vector<T> v = {0,1,2,3};
            config_[node][member] = v;
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        for(int i = 0; i < 4; i++)
        {
            val(i) = config_[node][member][i].as<T>();
        }
    }

    template <typename T>
    void read(const std::string node, const std::string member1, const std::string member2, T& val)
    {
        if(!config_[node][member1][member2])//find node
        {
        #if YAML_PROTECTED
            printf("Not Find [%s][%s]\n", node, member);
            exit(0);
        #else
            config_[node][member1][member2] = T(0);
            std::ofstream file(yaml_path);
            file << config_ << std::endl;
        #endif
        }
        val = config_[node][member1][member2].as<T>();
    }


    bool getString(const std::string &key, std::string &str_value);
    bool getString(const std::string &category, const std::string &key, std::string & str_value);

    template<typename T>
    bool getVector(const std::string &key, std::vector<T> &vec_value) {
      try {
        vec_value = config_[key].as<std::vector<T> >();
      } catch (std::exception &e) {
        return false;
      }
      return true;
    }

    template<typename T>
    bool getVector(const std::string &category, const std::string &key, std::vector<T> &vec_value) {
      try {
        vec_value = config_[category][key].as<std::vector<T>>();
      } catch (std::exception &e) {
        return false;
      }
      return true;
    }

    template<typename T>
    bool get2DArray(const std::string &category, const std::string &key, std::vector<std::vector<T> > &vec_value) {
      try {
        vec_value = config_[category][key].as<std::vector<std::vector<T> > >();
      } catch (std::exception &e) {
        return false;
      }
      return true;
    }

    template<typename T>
    bool getValue(const std::string &key, T &T_value) {
      try {
        T_value = config_[key].as<T>();
      } catch (std::exception &e) {
        return false;
      }
      return true;
    }

    template<typename T>
    bool getValue(const std::string &category, const std::string &key, T &T_value) {
      try {
        T_value = config_[category][key].as<T>();
        return true;
      } catch (std::exception &e) {
        return false;
      }
      return true;
    }

    bool getBoolean(const std::string & category, const std::string &key, bool &bool_value){
      try {
        bool_value = config_[category][key].as<bool>();
        return true;
      } catch (std::exception &e) {
        return false;
      }
      return true;
    }


    std::vector<std::string> getKeys() {
      std::vector<std::string> v;
      v.reserve(config_.size());
      for(auto it = config_.begin(); it != config_.end(); it++) {
        v.push_back(it->first.as<std::string>());
      }
      return v;
    }


    bool getBoolean(const std::string &key, bool &bool_value);

    bool getInteger(const std::string &key, int &int_value);

    bool fileOpenedSuccessfully() {
      return fileLoaded;
    }

private:
    dynacore_YAML::Node config_;
    std::string yaml_path;
    bool fileLoaded = false;
};

#endif
