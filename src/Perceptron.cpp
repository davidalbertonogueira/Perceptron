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


bool Perceptron::GetOutput(const std::vector<double> &x) {
  double inner_prod = std::inner_product(begin(x),
                                         end(x),
                                         begin(m_weights),
                                         0.0);

#if USE_SIGMOID == 1
  double y = utils::sigmoid(inner_prod);
  return (y > 0.5) ? true : false;
#else
  return (inner_prod > m_threshold) ? true : false;
#endif
};

void Perceptron::UpdateWeight(const std::vector<double> &x,
                              double error) {
  for (uint32_t i = 0; i < m_weights.size(); i++)
    m_weights[i] += x[i] * m_learning_rate *  error;
};

void Perceptron::Train(const std::vector<TrainingSample> &training_sample_set,
                       bool bias_already_in,
                       bool zero_weight_initialization) {
  std::vector<TrainingSample> training_sample_set_with_bias(training_sample_set);

  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(m_bias_value);
    }
  }
  int num_examples = training_sample_set_with_bias.size();
  int num_features = training_sample_set_with_bias[0].GetInputVectorSize();

  m_weights = std::vector<double>(num_features);

  //initialize weight vector
  std::generate_n(m_weights.begin(),
                  num_features,
                  (zero_weight_initialization) ? utils::gen_rand(0) : utils::gen_rand());

  std::cout << "Starting weights:\t";
  for (auto m_weightselement : m_weights)
    std::cout << m_weightselement << "\t";
  std::cout << std::endl;

  for (int i = 0; i < m_max_iterations; i++) {
    int error_count = 0;
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      bool prediction = GetOutput(training_sample_with_bias.input_vector());
      bool correct_output = training_sample_with_bias.output_value();
      if (prediction != correct_output) {
        error_count++;
        double error = (correct_output ? 1 : 0) - (prediction ? 1 : 0);
        UpdateWeight(training_sample_with_bias.input_vector(),
                     error);
      }
    }
    if (error_count == 0) break;
  }

  std::cout << "Final weights:\t\t";
  for (auto m_weightselement : m_weights)
    std::cout << m_weightselement << "\t";
  std::cout << std::endl;
};


