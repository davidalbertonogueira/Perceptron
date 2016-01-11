//============================================================================
// Name : ffPerceptron.cpp
// Author : David Nogueira
//============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <numeric>
#include <vector>
#include <algorithm>

#define USE_SIGMOID 1
struct gen_rand {
  double factor;
public:
  gen_rand(double r = 1.0) : factor(r / RAND_MAX) {}
  double operator()() {
    return rand() * factor;
  }
};

class Perceptron {
private:
  double m_alpha;
  int m_iterations;
  double m_threshold;
  std::vector<double>  m_w;

public:
  //Simple feed forward perceptron
  Perceptron(double alpha, int iterations, double threshold) {
    m_alpha = alpha;
    m_iterations = iterations;
    m_threshold = threshold;
    //initialize weights
    m_w.clear();
  };

  ~Perceptron() {
    m_w.clear();
  };

  double sigmoid(double x) {
    //Typical sigmoid function created from input x
    //param x: input value
    //return: sigmoided value
    return 1 / (1 + exp(-x));
  }

  // Derivative of sigmoid function
  double deriv_sigmoid(double x) {
    return sigmoid(x)*(1 - sigmoid(x));
  };

  int response(const std::vector<double> &x) {
    //perceptron response
    //param X: input vector
    //return: perceptron out
    double inner_prod = std::inner_product(begin(x),
                                           end(x),
                                           begin(m_w),
                                           0.0);

#if USE_SIGMOID == 1
    double y = sigmoid(inner_prod);
    // In this super simple example perceptron will return
    // 1 for positive guess and 0 otherwise
    if (y > 0.5) {
      return 1;
    } else {
      return 0;
    }
#else
    if (inner_prod > m_threshold) {
      return 1;
    } else {
      return 0;
    }
#endif
  };

  void updateWeight(const std::vector<double> &x,
                    double error) {
    //update the vector of input weights
    //param X: input data
    //param error: prediction != true
    //return: updated weight vector
    for (int i = 0; i < m_w.size(); i++)
      m_w[i] += x[i] * m_alpha *  error;
  };

  void train(const std::vector<std::vector<double>> &x,
             const std::vector<int>  y,
             double bias_value) {
    //trains perceptron on vector data by looping each row
    //and updating the weight vector
    //param X: input data
    //param y: correct y value
    //return: updated parameters
    int num_examples = x.size();
    int num_features = x[0].size();

    m_w = std::vector<double>(num_features + 1);

    //set up bias
    std::vector<std::vector<double>> bias =
      std::vector<std::vector<double>>(num_examples,
                                       std::vector<double>(1, bias_value));
    std::vector<std::vector<double>> x_and_bias =
      std::vector<std::vector<double>>(num_examples,
                                       std::vector<double>(num_features + 1, 0.0));

    //[x_and_bias] = concatenation of [bias] [X]

    for (int i = 0; i < num_examples; i++) {
      x_and_bias[i][0] = bias[i][0];
      for (int j = 0; j < num_features; j++) {
        x_and_bias[i][j + 1] = x[i][j];
      }
    }

    //initialize weight vector
    std::generate_n(m_w.begin(),
                    num_features + 1,
                    gen_rand());

    std::cout << "starting weights:";
    for (auto m_welement : m_w)
      std::cout << m_welement << "\t";
    std::cout << std::endl;

    for (int i = 0; i < m_iterations; i++) {
      for (int j = 0; j < num_examples; j++) {
        int prediction = response(x_and_bias[j]);
        if (prediction != y[j]) {
          double error = y[j] - prediction;
          updateWeight(x_and_bias[j],
                       error);
        }
      }
    }

    std::cout << "final weights:";
    for (auto m_welement : m_w)
      std::cout << m_welement << "\t";
    std::cout << std::endl;
  };
};

int main() {
  Perceptron my_perceptron(0.1, 1000, 0.5);
  //Trying to learn a binary NAND function on inputs x1 and x2.
  int num_examples = 26;
  int num_features = 5;
  double bias_value = 1;
  std::vector< std::vector<double>> x =
    std::vector<std::vector<double>>(num_examples,
                                     std::vector<double>(num_features, bias_value));
  std::vector<int> y = std::vector<int>(num_examples, 0);

  x.push_back({ 0,0,0,0,0 });  y.push_back(1);
  x.push_back({ 0,0,1,0,0 });  y.push_back(1);
  x.push_back({ 0,1,0,0,0 });  y.push_back(1);
  x.push_back({ 0,1,1,0,0 });  y.push_back(0);
  x.push_back({ 1,0,0,0,0 });  y.push_back(1);
  //x.push_back({ 1,0,1,0,0 });  y.push_back(1);
  x.push_back({ 1,1,0,0,0 });  y.push_back(1);
  x.push_back({ 1,1,1,0,0 });  y.push_back(0);

  x.push_back({ 0,0,0,0,1 });  y.push_back(1);
  x.push_back({ 0,0,1,0,1 });  y.push_back(1);
  //x.push_back({ 0,1,0,0,1 });  y.push_back(1);
  x.push_back({ 0,1,1,0,1 });  y.push_back(0);
  x.push_back({ 1,0,0,0,1 });  y.push_back(1);
  x.push_back({ 1,0,1,0,1 });  y.push_back(1);
  x.push_back({ 1,1,0,0,1 });  y.push_back(1);
  x.push_back({ 1,1,1,0,1 });  y.push_back(0);

  x.push_back({ 0,0,0,1,0 });  y.push_back(1);
  x.push_back({ 0,0,1,1,0 });  y.push_back(1);
  x.push_back({ 0,1,0,1,0 });  y.push_back(1);
  x.push_back({ 0,1,1,1,0 });  y.push_back(0);
  //x.push_back({ 1,0,0,1,0 });  y.push_back(1);
  x.push_back({ 1,0,1,1,0 });  y.push_back(1);
  //x.push_back({ 1,1,0,1,0 });  y.push_back(1);
  x.push_back({ 1,1,1,1,0 });  y.push_back(0);

  x.push_back({ 0,0,0,1,1 });  y.push_back(1);
  x.push_back({ 0,0,1,1,1 });  y.push_back(1);
  x.push_back({ 0,1,0,1,1 });  y.push_back(1);
  //x.push_back({ 0,1,1,1,1 });  y.push_back(0);
  x.push_back({ 1,0,0,1,1 });  y.push_back(1);
  x.push_back({ 1,0,1,1,1 });  y.push_back(1);
  x.push_back({ 1,1,0,1,1 });  y.push_back(1);
  //x.push_back({ 1,1,1,1,1 });  y.push_back(0);

  my_perceptron.train(x, y, bias_value);

  std::cout << "f( { 0, 0, 0, 0, 0} ) = " << my_perceptron.response({ bias_value, 0, 0, 0, 0, 0 }) << std::endl;
  std::cout << "f( { 0, 0, 1, 0, 0} ) = " << my_perceptron.response({ bias_value, 0, 0, 1, 0, 0 }) << std::endl;
  std::cout << "f( { 0, 1, 0, 0, 0} ) = " << my_perceptron.response({ bias_value, 0, 1, 0, 0, 0 }) << std::endl;
  std::cout << "f( { 0, 1, 1, 0, 0} ) = " << my_perceptron.response({ bias_value, 0, 1, 1, 0, 0 }) << std::endl;
  std::cout << "f( { 1, 0, 0, 0, 0} ) = " << my_perceptron.response({ bias_value, 1, 0, 0, 0, 0 }) << std::endl;
  std::cout << "f( { 1, 0, 1, 0, 0} ) = " << my_perceptron.response({ bias_value, 1, 0, 1, 0, 0 }) << std::endl;
  std::cout << "f( { 1, 1, 0, 0, 0} ) = " << my_perceptron.response({ bias_value, 1, 1, 0, 0, 0 }) << std::endl;
  std::cout << "f( { 1, 1, 1, 0, 0} ) = " << my_perceptron.response({ bias_value, 1, 1, 1, 0, 0 }) << std::endl;

  std::cout << "f( { 0, 0, 0, 0, 1} ) = " << my_perceptron.response({ bias_value, 0, 0, 0, 0, 1 }) << std::endl;
  std::cout << "f( { 0, 0, 1, 0, 1} ) = " << my_perceptron.response({ bias_value, 0, 0, 1, 0, 1 }) << std::endl;
  std::cout << "f( { 0, 1, 0, 0, 1} ) = " << my_perceptron.response({ bias_value, 0, 1, 0, 0, 1 }) << std::endl;
  std::cout << "f( { 0, 1, 1, 0, 1} ) = " << my_perceptron.response({ bias_value, 0, 1, 1, 0, 1 }) << std::endl;
  std::cout << "f( { 1, 0, 0, 0, 1} ) = " << my_perceptron.response({ bias_value, 1, 0, 0, 0, 1 }) << std::endl;
  std::cout << "f( { 1, 0, 1, 0, 1} ) = " << my_perceptron.response({ bias_value, 1, 0, 1, 0, 1 }) << std::endl;
  std::cout << "f( { 1, 1, 0, 0, 1} ) = " << my_perceptron.response({ bias_value, 1, 1, 0, 0, 1 }) << std::endl;
  std::cout << "f( { 1, 1, 1, 0, 1} ) = " << my_perceptron.response({ bias_value, 1, 1, 1, 0, 1 }) << std::endl;

  std::cout << "f( { 0, 0, 0, 1, 0} ) = " << my_perceptron.response({ bias_value, 0, 0, 0, 1, 0 }) << std::endl;
  std::cout << "f( { 0, 0, 1, 1, 0} ) = " << my_perceptron.response({ bias_value, 0, 0, 1, 1, 0 }) << std::endl;
  std::cout << "f( { 0, 1, 0, 1, 0} ) = " << my_perceptron.response({ bias_value, 0, 1, 0, 1, 0 }) << std::endl;
  std::cout << "f( { 0, 1, 1, 1, 0} ) = " << my_perceptron.response({ bias_value, 0, 1, 1, 1, 0 }) << std::endl;
  std::cout << "f( { 1, 0, 0, 1, 0} ) = " << my_perceptron.response({ bias_value, 1, 0, 0, 1, 0 }) << std::endl;
  std::cout << "f( { 1, 0, 1, 1, 0} ) = " << my_perceptron.response({ bias_value, 1, 0, 1, 1, 0 }) << std::endl;
  std::cout << "f( { 1, 1, 0, 1, 0} ) = " << my_perceptron.response({ bias_value, 1, 1, 0, 1, 0 }) << std::endl;
  std::cout << "f( { 1, 1, 1, 1, 0} ) = " << my_perceptron.response({ bias_value, 1, 1, 1, 1, 0 }) << std::endl;

  std::cout << "f( { 0, 0, 0, 1, 1} ) = " << my_perceptron.response({ bias_value, 0, 0, 0, 1, 1 }) << std::endl;
  std::cout << "f( { 0, 0, 1, 1, 1} ) = " << my_perceptron.response({ bias_value, 0, 0, 1, 1, 1 }) << std::endl;
  std::cout << "f( { 0, 1, 0, 1, 1} ) = " << my_perceptron.response({ bias_value, 0, 1, 0, 1, 1 }) << std::endl;
  std::cout << "f( { 0, 1, 1, 1, 1} ) = " << my_perceptron.response({ bias_value, 0, 1, 1, 1, 1 }) << std::endl;
  std::cout << "f( { 1, 0, 0, 1, 1} ) = " << my_perceptron.response({ bias_value, 1, 0, 0, 1, 1 }) << std::endl;
  std::cout << "f( { 1, 0, 1, 1, 1} ) = " << my_perceptron.response({ bias_value, 1, 0, 1, 1, 1 }) << std::endl;
  std::cout << "f( { 1, 1, 0, 1, 1} ) = " << my_perceptron.response({ bias_value, 1, 1, 0, 1, 1 }) << std::endl;
  std::cout << "f( { 1, 1, 1, 1, 1} ) = " << my_perceptron.response({ bias_value, 1, 1, 1, 1, 1 }) << std::endl;

  return 0;
}
