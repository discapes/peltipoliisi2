#pragma once

#include <vector>
#include "event_reader.hpp"

// Estimate RPM from timestamps (µs) of events in a small region using FFTW.
// - events: vector of Event; only the timestamp field (t, in microseconds) is used
// - num_blades: number of blades on the propeller (defaults to 2)
// Returns NaN if not enough data or no clear peak.
double estimate_rpm_from_events(const std::vector<Event> &events, int num_blades = 2);
