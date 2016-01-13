//============================================================================
// Name : Main.cpp
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
#include <cassert>

void LearnAND() {
  std::cout << "Train AND function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },false},
    {{ 1, 0, 1 },false},
    {{ 1, 1, 0 },false},
    {{ 1, 1, 1 },true}
  };

  Perceptron my_perceptron(0.1, 100, 0.5);
  my_perceptron.Train(training_set, true, true);

  assert(my_perceptron.GetOutput({ 1, 0, 0 }) == false);
  assert(my_perceptron.GetOutput({ 1, 0, 1 }) == false);
  assert(my_perceptron.GetOutput({ 1, 1, 0 }) == false);
  assert(my_perceptron.GetOutput({ 1, 1, 1 }) == true);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnNAND() {
  std::cout << "Train NAND function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },true},
    {{ 1, 0, 1 },true},
    {{ 1, 1, 0 },true},
    {{ 1, 1, 1 },false}
  };

  Perceptron my_perceptron(0.1, 100, 0.5);
  my_perceptron.Train(training_set, true, true);

  assert(my_perceptron.GetOutput({ 1, 0, 0 }) == true);
  assert(my_perceptron.GetOutput({ 1, 0, 1 }) == true);
  assert(my_perceptron.GetOutput({ 1, 1, 0 }) == true);
  assert(my_perceptron.GetOutput({ 1, 1, 1 }) == false);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnOR() {
  std::cout << "Train OR function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },false},
    {{ 1, 0, 1 },true},
    {{ 1, 1, 0 },true},
    {{ 1, 1, 1 },true}
  };

  Perceptron my_perceptron(0.1, 100, 0.5);
  my_perceptron.Train(training_set, true, true);

  assert(my_perceptron.GetOutput({ 1, 0, 0 }) == false);
  assert(my_perceptron.GetOutput({ 1, 0, 1 }) == true);
  assert(my_perceptron.GetOutput({ 1, 1, 0 }) == true);
  assert(my_perceptron.GetOutput({ 1, 1, 1 }) == true);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnNOR() {
  std::cout << "Train NOR function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0, 0 },true },
    {{ 1, 0, 1 },false},
    {{ 1, 1, 0 },false},
    {{ 1, 1, 1 },false }
  };

  Perceptron my_perceptron(0.1, 100, 0.5);
  my_perceptron.Train(training_set, true, true);

  assert(my_perceptron.GetOutput({ 1, 0, 0 }) == true);
  assert(my_perceptron.GetOutput({ 1, 0, 1 }) == false);
  assert(my_perceptron.GetOutput({ 1, 1, 0 }) == false);
  assert(my_perceptron.GetOutput({ 1, 1, 1 }) == false);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

void LearnNOT() {
  std::cout << "Train NOT function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 1, 0},true},
    {{ 1, 1},false}
  };

  Perceptron my_perceptron(0.1, 100, 0.5);
  my_perceptron.Train(training_set, true, true);

  assert(my_perceptron.GetOutput({ 1, 0 }) == true);
  assert(my_perceptron.GetOutput({ 1, 1 }) == false);
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

int main() {
  LearnAND();
  LearnNAND();
  LearnOR();
  LearnNOR();
  LearnNOT();

  return 0;
}