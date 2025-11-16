#include <fftw3.h>
#include "defs.hpp"

double estimate_rpm_from_events(const vector<FrameEvent> &events, int num_blades) {
  const size_t N = events.size();
  if (N < 16) return numeric_limits<double>::quiet_NaN(); // not enough events

  // Extract times in seconds, sorted
  vector<double> t;
  t.reserve(N);
  for (const auto &e : events) t.push_back(static_cast<double>(e.t) * 1e-6);
  sort(t.begin(), t.end());

  const double t0 = t.front();
  const double t_last = t.back();
  const double T_total = t_last - t0;
  if (T_total <= 0.0) return numeric_limits<double>::quiet_NaN();

  // Inter-event gaps to estimate a reasonable bin width
  vector<double> dt;
  dt.reserve(N - 1);
  for (size_t i = 0; i + 1 < N; ++i) {
    double d = t[i + 1] - t[i];
    if (d > 0.0) dt.push_back(d);
  }
  if (dt.size() < 4) return numeric_limits<double>::quiet_NaN();

  // Median of dt
  nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
  double median_dt = dt[dt.size() / 2];

  // Bin width: several times median inter-event gap, but not too small
  double bin_width = max(5.0 * median_dt, 1e-5); // >= 10 µs

  int nbins = static_cast<int>(ceil(T_total / bin_width));
  if (nbins < 16) nbins = 16;

  // Allocate FFTW input/output
  double *in = (double*)fftw_malloc(sizeof(double) * nbins);
  fftw_complex *out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (nbins / 2 + 1));
  if (!in || !out) {
    if (in) fftw_free(in);
    if (out) fftw_free(out);
    return numeric_limits<double>::quiet_NaN();
  }

  // Zero-initialize bins
  fill(in, in + nbins, 0.0);

  // Histogram: event counts per bin
  for (double ti : t) {
    double rel = ti - t0;  // relative time
    int idx = static_cast<int>(rel / bin_width);
    if (idx >= 0 && idx < nbins) in[idx] += 1.0;
  }

  // Remove DC (mean) component
  double sum = 0.0;
  for (int i = 0; i < nbins; ++i) sum += in[i];
  double mean = sum / nbins;
  for (int i = 0; i < nbins; ++i) in[i] -= mean;

  // Plan & execute FFT
  fftw_plan plan = fftw_plan_dft_r2c_1d(nbins, in, out, FFTW_ESTIMATE);
  fftw_execute(plan);

  // Frequency axis info
  double fs = 1.0 / bin_width;          // sampling frequency (Hz)
  int nfreq = nbins / 2 + 1;

  // Ignore DC (k=0); search within reasonable band
  double f_min = 5.0;      // Hz
  double f_max = 5000.0;   // Hz

  double best_mag = 0.0;
  double best_freq = 0.0;

  for (int k = 1; k < nfreq; ++k) {
    double f = (static_cast<double>(k) * fs) / nbins;
    if (f < f_min || f > f_max) continue;

    double re = out[k][0];
    double im = out[k][1];
    double mag = hypot(re, im);

    if (mag > best_mag) {
      best_mag = mag;
      best_freq = f;
    }
  }

  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  if (best_freq <= 0.0) return numeric_limits<double>::quiet_NaN();

  // best_freq is blade-pass frequency (Hz)
  if (num_blades <= 0) return numeric_limits<double>::quiet_NaN();
  double f_rot = best_freq / static_cast<double>(num_blades); // rotor frequency
  double rpm = 60.0 * f_rot;
  return rpm;
}
