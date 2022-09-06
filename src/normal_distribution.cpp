#include "normal_distribution.h"

// normal_distribution
#include <iostream>
#include <string>
#include <random>

std::default_random_engine generator;
std::normal_distribution<double> *distribution;

void init_normal_distribution(double comp, double noise) {
  double std = (comp * noise) / 100.0;
  double mean = comp;
  distribution = new std::normal_distribution<double>(mean, std);
}

double get_random_normal_number()
{
  double num = (*distribution)(generator);
  return num ;
}
