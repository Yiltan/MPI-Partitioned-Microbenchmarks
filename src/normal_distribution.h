#ifndef NORMAL_DISTRIBUTION_H
#define NORMAL_DISTRIBUTION_H

#ifdef __cplusplus
extern "C" {
#endif
  void init_normal_distribution(double mean, double std);
  double get_random_normal_number();
#ifdef __cplusplus
}
#endif

#endif /* NORMAL_DISTRIBUTION_H */
