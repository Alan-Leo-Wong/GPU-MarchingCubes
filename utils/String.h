#pragma once
#include <io.h>
#include <string>

using std::string;

#ifndef DELIMITER
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define DELIMITER "\\"
#else
#define DELIMITER "/"
#endif
#endif

inline void checkDir(const string &filename) {
  size_t dir_idx = filename.find_last_of(DELIMITER);
  if (dir_idx != string::npos) {
    // cout << "output dir = " << filename.substr(0, dir_idx) << endl;
    string dir = filename.substr(0, dir_idx);
    if (_access(dir.c_str(), 0) == -1) {
      string command = "mkdir " + dir;
      system(command.c_str());
    }
  }
}