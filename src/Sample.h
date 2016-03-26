#ifndef TRAININGSAMPLE_H
#define TRAININGSAMPLE_H

#include <stdlib.h>
#include <vector>

class Sample {
public:
  Sample(const std::vector<double> & input_vector) {

    m_input_vector = input_vector;
  }
  std::vector<double> & input_vector() {
    return m_input_vector;
  }
  size_t GetInputVectorSize() const {
    return m_input_vector.size();
  }
  void AddBiasValue(double bias_value) {
    m_input_vector.insert(m_input_vector.begin(), bias_value);
  }
protected:
  std::vector<double> m_input_vector;
};


class TrainingSample : public Sample {
public:
  TrainingSample(const std::vector<double> & input_vector,
                 bool output_value) :
    Sample(input_vector) {
    m_output = output_value;
  }
  bool output_value() const { return m_output; }
protected:
  bool m_output;
};
#endif // TRAININGSAMPLE_H