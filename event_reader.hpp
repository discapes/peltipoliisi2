#pragma once
#include <cstdint>
#include <string>
#include <functional>

using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;

// Decoded event structure from 8-byte DAT record.
// t: timestamp in microseconds
// x, y: coordinates (14-bit each, stored in lower bits)
// polarity: 0 or 1
struct Event { u32 t; uint16_t x; uint16_t y; uint8_t polarity; };

struct DatHeaderInfo {
    int width=-1, height=-1, version=-1;
    string date;
    int event_type=-1, event_size=-1;
};

// Reads a DAT file and streams events paced to real time.
// Additionally implements a sliding time window (default 50ms):
//  - At each event arrival time, invokes callback(e, +1) to indicate an increment.
//  - When an event leaves the window (t + window_us), invokes callback(e, -1) to indicate a decrement.
// The function sleeps until the earlier of the next arrival or next expiry to maintain pacing.
// Returns true on success; false otherwise.
bool stream_dat_events(const string &path,
                       const function<void(const Event &, int delta)> &callback,
                       DatHeaderInfo *out_header=nullptr,
                       u32 window_us = 50'000);
