#include "ParamYaml.h"
bool xpYaml::getString(const std::string &key, std::string &str_value) {
  try {
    str_value = config_[key].as<std::string>();
  } catch (std::exception &e) {
    return false;
  }
  return true;
}

bool xpYaml::getString(const std::string &category, const std::string &key, std::string &str_value) {
  try {
    str_value = config_[category][key].as<std::string>();
  } catch(std::exception &e) {
    return false;
  }
  return true;
}

bool xpYaml::getBoolean(const std::string &key, bool &bool_value) {
  try {
    bool_value = config_[key].as<bool>();
  } catch (std::exception &e) {
    return false;
  }
  return true;
}

bool xpYaml::getInteger(const std::string &key, int &int_value) {
  try {
    int_value = config_[key].as<int>();
  } catch (std::exception &e) {
    return false;
  }
  return true;
}
