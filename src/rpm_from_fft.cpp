#include <fftw3.h>
#include "defs.hpp"
#include <cmath>
#include <algorithm>

double rpm_from_fft(const vector<FrameEvent> &events, int num_blades)
{
  const size_t N = events.size();
  if (N < 16)
    return numeric_limits<double>::quiet_NaN(); // not enough events

  // Extract times in seconds, sorted
  vector<double> t;
  t.reserve(N);
  for (const auto &e : events)
    t.push_back(static_cast<double>(e.t) * 1e-6);
  sort(t.begin(), t.end());

  const double t0 = t.front();
  const double t_last = t.back();
  const double T_total = t_last - t0;
  if (T_total <= 0.0)
    return numeric_limits<double>::quiet_NaN();

  // Inter-event gaps to estimate a reasonable bin width
  vector<double> dt;
  dt.reserve(N - 1);
  for (size_t i = 0; i + 1 < N; ++i)
  {
    double d = t[i + 1] - t[i];
    if (d > 0.0)
      dt.push_back(d);
  }
  if (dt.size() < 4)
    return numeric_limits<double>::quiet_NaN();

  // Median of dt
  nth_element(dt.begin(), dt.begin() + dt.size() / 2, dt.end());
  double median_dt = dt[dt.size() / 2];

  // Bin width: several times median inter-event gap, but not too small
  double bin_width = max(5.0 * median_dt, 1e-5); // >= 10 µs

  // Choose a stable FFT size: round up to next power of two, with sane bounds.
  int nbins = static_cast<int>(ceil(T_total / bin_width));
  if (nbins < 64)
    nbins = 64;
  // round up to next power-of-two
  {
    int p = 1;
    while (p < nbins && p < 4096)
      p <<= 1;
    nbins = p;
  }

  // Allocate FFTW input/output
  double *in = (double *)fftw_malloc(sizeof(double) * nbins);
  fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (nbins / 2 + 1));
  if (!in || !out)
  {
    if (in)
      fftw_free(in);
    if (out)
      fftw_free(out);
    return numeric_limits<double>::quiet_NaN();
  }

  // Zero-initialize bins
  fill(in, in + nbins, 0.0);

  // Histogram: event counts per bin
  for (double ti : t)
  {
    double rel = ti - t0; // relative time
    int idx = static_cast<int>(rel / bin_width);
    if (idx >= 0 && idx < nbins)
      in[idx] += 1.0;
  }

  // Remove DC (mean) component
  double sum = 0.0;
  for (int i = 0; i < nbins; ++i)
    sum += in[i];
  double mean = sum / nbins;
  for (int i = 0; i < nbins; ++i)
    in[i] -= mean;

  // Apply Hann window to reduce spectral leakage
  const double pi = 3.14159265358979323846;
  if (nbins > 1)
  {
    for (int i = 0; i < nbins; ++i)
    {
      double w = 0.5 * (1.0 - cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(nbins - 1)));
      in[i] *= w;
    }
  }

  // Plan & execute FFT
  fftw_plan plan = fftw_plan_dft_r2c_1d(nbins, in, out, FFTW_ESTIMATE);
  fftw_execute(plan);

  // Frequency axis info
  double fs = 1.0 / bin_width; // sampling frequency (Hz)
  int nfreq = nbins / 2 + 1;

  // Ignore DC (k=0); search within reasonable band
  double f_min = 5.0;    // Hz
  double f_max = 5000.0; // Hz

  // Build magnitude spectrum and use a 3-bin neighborhood sum to score peaks
  vector<double> mag(nfreq, 0.0);
  for (int k = 0; k < nfreq; ++k)
  {
    double re = out[k][0];
    double im = out[k][1];
    mag[k] = hypot(re, im);
  }

  auto band_mag = [&](int k)
  {
    double s = 0.0;
    if (k > 0)
      s += mag[k - 1];
    s += mag[k];
    if (k + 1 < nfreq)
      s += mag[k + 1];
    return s;
  };

  double best_score = 0.0;
  int best_k = -1;
  for (int k = 1; k < nfreq; ++k)
  {
    double f = (static_cast<double>(k) * fs) / nbins;
    if (f < f_min || f > f_max)
      continue;
    double score = band_mag(k);
    if (score > best_score)
    {
      best_score = score;
      best_k = k;
    }
  }

  if (best_k < 0)
  {

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return numeric_limits<double>::quiet_NaN();
  }

  // Prefer a lower harmonic (fundamental) if it's close in energy
  auto score_at = [&](int k)
  {
    if (k <= 0 || k >= nfreq)
      return 0.0;
    double f = (static_cast<double>(k) * fs) / nbins;
    if (f < f_min || f > f_max)
      return 0.0;
    return band_mag(k);
  };

  const double harmonic_bias_ratio = 0.6; // within ~ -4.4 dB
  int chosen_k = best_k;
  int half_k = static_cast<int>(llround(best_k / 2.0));
  int third_k = static_cast<int>(llround(best_k / 3.0));
  double best_s = band_mag(best_k);
  double s_half = score_at(half_k);
  double s_third = score_at(third_k);
  if (s_half >= harmonic_bias_ratio * best_s)
  {
    chosen_k = half_k;
  }
  else if (s_third >= harmonic_bias_ratio * best_s)
  {
    chosen_k = third_k;
  }

  double best_freq = (static_cast<double>(chosen_k) * fs) / nbins;

  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  if (best_freq <= 0.0)
    return numeric_limits<double>::quiet_NaN();

  // best_freq is blade-pass frequency (Hz)
  if (num_blades <= 0)
    return numeric_limits<double>::quiet_NaN();
  double f_rot = best_freq / static_cast<double>(num_blades); // rotor frequency
  double rpm = 60.0 * f_rot;
  return rpm;
}
