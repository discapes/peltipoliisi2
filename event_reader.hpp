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

// Reads a DAT file in the described format. Parses header lines starting with '%'
// until a line exactly "% end". Then streams binary 8-byte event records without
// loading the whole file. For each decoded Event, invokes the callback.
// Returns true on success (file opened and header parsed); false otherwise.
// The callback may be empty; in that case events are simply discarded.
bool stream_dat_events(const string &path,
                       const function<void(const Event &)> &callback,
                       DatHeaderInfo *out_header=nullptr);
