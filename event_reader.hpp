#pragma once
#include <cstdint>
#include <string>
#include <functional>

// Decoded event structure from 8-byte DAT record.
// t: timestamp in microseconds
// x, y: coordinates (14-bit each, stored in lower bits)
// polarity: 0 or 1
struct Event {
    uint32_t t;       // microsecond timestamp
    uint16_t x;       // 0..16383
    uint16_t y;       // 0..16383
    uint8_t polarity; // 0 or 1
};

struct DatHeaderInfo {
    int width = -1;
    int height = -1;
    int version = -1;
    std::string date;
};

// Reads a DAT file in the described format. Parses header lines starting with '%'
// until a line exactly "% end". Then streams binary 8-byte event records without
// loading the whole file. For each decoded Event, invokes the callback.
// Returns true on success (file opened and header parsed); false otherwise.
// The callback may be empty; in that case events are simply discarded.
// Extended optional outputs:
// out_event_count: total number of events decoded.
// out_first_ts: timestamp of first event (microseconds) or 0 if none.
// out_last_ts: timestamp of last event (microseconds) or 0 if none.
// out_wall_seconds: wall-clock seconds spent streaming (decode + IO).
// out_data_span_us: (last_ts - first_ts) span in microseconds (0 if <2 events).
// realtime: if true, events are emitted paced to their timestamp differences
//           relative to the first event (microseconds). This causes streaming
//           to take roughly the original capture duration rather than finishing
//           as fast as possible. When false, events are emitted immediately.
// NOTE: When realtime is true, wall_clock time reported includes the sleeps.
bool stream_dat_events(const std::string &path,
                       const std::function<void(const Event &)> &callback,
                       DatHeaderInfo *out_header = nullptr,
                       std::uint64_t *out_event_count = nullptr,
                       std::uint32_t *out_first_ts = nullptr,
                       std::uint32_t *out_last_ts = nullptr,
                       double *out_wall_seconds = nullptr,
                       std::uint64_t *out_data_span_us = nullptr,
                       bool realtime = false);
