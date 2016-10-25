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
#include "microunit.h"

UNIT(LearnAND) {
  std::cout << "Train AND function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0, 0 },false},
    {{ 0, 1 },false},
    {{ 1, 0 },false},
    {{ 1, 1 },true}
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  Perceptron my_perceptron;
  my_perceptron.Train(training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    class_id = my_perceptron.GetOutput(training_sample.input_vector());
    ASSERT_TRUE(class_id == training_sample.output_value());
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNAND) {
  std::cout << "Train NAND function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0, 0 },true},
    {{ 0, 1 },true},
    {{ 1, 0 },true},
    {{ 1, 1 },false}
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  Perceptron my_perceptron;
  my_perceptron.Train(training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    class_id = my_perceptron.GetOutput(training_sample.input_vector());
    ASSERT_TRUE(class_id == training_sample.output_value());
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnOR) {
  std::cout << "Train OR function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{  0, 0 },false},
    {{  0, 1 },true},
    {{  1, 0 },true},
    {{  1, 1 },true}
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  Perceptron my_perceptron;
  my_perceptron.Train(training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    class_id = my_perceptron.GetOutput(training_sample.input_vector());
    ASSERT_TRUE(class_id == training_sample.output_value());
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOR) {
  std::cout << "Train NOR function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0, 0 },true },
    {{ 0, 1 },false},
    {{ 1, 0 },false},
    {{ 1, 1 },false }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  Perceptron my_perceptron;
  my_perceptron.Train(training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    class_id = my_perceptron.GetOutput(training_sample.input_vector());
    ASSERT_TRUE(class_id == training_sample.output_value());
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnNOT) {
  std::cout << "Train NOT function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    {{ 0},true},
    {{ 1},false}
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  Perceptron my_perceptron;
  my_perceptron.Train(training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    class_id = my_perceptron.GetOutput(training_sample.input_vector());
    ASSERT_TRUE(class_id == training_sample.output_value());
  }
  std::cout << "Trained with success." << std::endl;
  std::cout << std::endl;
}

UNIT(LearnXOR) {
  std::cout << "Train XOR function with perceptron." << std::endl;

  std::vector<TrainingSample> training_set =
  {
    { { 0, 0 },false },
    { { 0, 1 },true },
    { { 1, 0 },true },
    { { 1, 1 },false }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  Perceptron my_perceptron;
  my_perceptron.Train(training_sample_set_with_bias,  0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    class_id = my_perceptron.GetOutput(training_sample.input_vector());
    if(class_id != training_sample.output_value()) {
      std::cout << "Failed to train. " <<
        " A simple perceptron cannot learn the XOR function." << std::endl;
      FAIL();
    }
  }
}

int main() {
  microunit::UnitTester::Run();
  return 0;
}