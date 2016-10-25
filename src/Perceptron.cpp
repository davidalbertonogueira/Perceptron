//============================================================================
// Name : ffPerceptron.cpp
// Author : David Nogueira
//============================================================================
#include "Perceptron.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>

#define USE_SIGMOID 0
#define ZERO_WEIGHT_INITIALIZATION 1


bool Perceptron::GetOutput(const std::vector<double> &x) const {
  assert(x.size() == m_weights.size());
  double inner_prod = std::inner_product(begin(x),
                                         end(x),
                                         begin(m_weights),
                                         0.0);

#if USE_SIGMOID == 1
  double y = utils::sigmoid(inner_prod);
  return (y > 0.5) ? true : false;
#else
  return (inner_prod > 0) ? true : false;
#endif
};

void Perceptron::UpdateWeight(const std::vector<double> &x,
                              double error,
                              double learning_rate) {
  for (uint32_t i = 0; i < m_weights.size(); i++)
    m_weights[i] += x[i] * learning_rate *  error;
};

void Perceptron::Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
                       double learning_rate,
                       int max_iterations) {
  size_t num_examples = training_sample_set_with_bias.size();
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();

  m_weights.resize(num_features);

  //initialize weight vector
  std::generate_n(m_weights.begin(),
                  num_features,
                  (ZERO_WEIGHT_INITIALIZATION) ? 
                  utils::gen_rand(0) : utils::gen_rand());

  std::cout << "Starting weights:\t";
  for (auto m_weightselement : m_weights)
    std::cout << m_weightselement << "\t";
  std::cout << std::endl;

  for (int i = 0; i < max_iterations; i++) {
    int error_count = 0;
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      bool prediction = GetOutput(training_sample_with_bias.input_vector());
      bool correct_output = training_sample_with_bias.output_value();
      if (prediction != correct_output) {
        error_count++;
        double error = (correct_output ? 1 : 0) - (prediction ? 1 : 0);
        UpdateWeight(training_sample_with_bias.input_vector(),
                     error,
                     learning_rate);
      }
    }
    if (error_count == 0) break;
  }

  std::cout << "Final weights:\t\t";
  for (auto m_weightselement : m_weights)
    std::cout << m_weightselement << "\t";
  std::cout << std::endl;
};


