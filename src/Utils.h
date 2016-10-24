#ifndef UTILS_H
#define UTILS_H

#include "Chrono.h"
#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <chrono>
#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

namespace utils {

struct gen_rand {
  double factor;
  double offset;
public:
  gen_rand(double r = 2.0) : factor(r / RAND_MAX), offset(r / 2) {}
  double operator()() {
    return rand() * factor - offset;
  }
};

inline double sigmoid(double x) {
  //Typical sigmoid function created from input x
  //param x: input value
  //return: sigmoided value
  return 1 / (1 + exp(-x));
}

// Derivative of sigmoid function
inline double deriv_sigmoid(double x) {
  return sigmoid(x)*(1 - sigmoid(x));
};
}
#endif // UTILS_H