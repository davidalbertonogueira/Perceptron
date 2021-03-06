//============================================================================
// Name : ffPerceptron.cpp
// Author : David Nogueira
//============================================================================
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "Sample.h"
#include "Utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert> // for assert()

class Perceptron {
public:
  //Simple feed forward perceptron
  Perceptron() {
    //initialize weights
    m_weights.clear();
  };

  ~Perceptron() {
    m_weights.clear();
  };

 void  GetInputInnerProdWithWeights(const std::vector<double> &input,
                               double * output) const;

 bool GetBooleanOutput(const std::vector<double> &input) const;

  void UpdateWeight(const std::vector<double> &x,
                    double error,
                    double learning_rate);

  void Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
             double learning_rate,
             int max_iterations);

private:
  std::vector<double>  m_weights;
};

#endif //PERCEPTRON_H