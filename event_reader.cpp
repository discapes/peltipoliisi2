#include "event_reader.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>

namespace {
#pragma pack(push,1)
struct RawRecord { u32 t32; u32 w32; }; // 8-byte raw event
#pragma pack(pop)

Event decode(const RawRecord &r) {
    Event e;
    e.t = r.t32; // already LE on LE hosts
    u32 w = r.w32;
    u32 x = w & 0x3FFFu;          // bits 13..0
    u32 y = (w >> 14) & 0x3FFFu;  // bits 27..14
    u32 raw_pol = (w >> 28) & 0xFu; // bits 31..28
    e.x = static_cast<uint16_t>(x);
    e.y = static_cast<uint16_t>(y);
    e.polarity = static_cast<uint8_t>(raw_pol ? 1 : 0);
    return e;
}
}

static bool parse_header(ifstream &ifs, DatHeaderInfo *info) {
    // Read line by line until "% end" encountered.
    string line;
    while (getline(ifs, line)) {
        if (line.rfind("%", 0) != 0) {
            // Line doesn't start with %, treat as start of binary (rewind position?)
            // Move stream position back to beginning of this line's content for binary.
            auto back = static_cast<streamoff>(line.size() + 1); // +1 for newline consumed
            ifs.seekg(-back, ios::cur);
            break; // header ended implicitly
        }
    if (line == "% end") {
            break; // finished header; binary starts after this line's newline
        }
        // Parse key-value pairs of form: % Key value(s)
        if (info) {
            string content = line.substr(1); // drop '%'
            while (!content.empty() && content[0] == ' ') content.erase(0,1);
            auto pos = content.find(' ');
            string key = content.substr(0, pos);
            string value = (pos == string::npos) ? string() : content.substr(pos+1);
            if (key == "Width") info->width = stoi(value);
            else if (key == "Height") info->height = stoi(value);
            else if (key == "Version") info->version = stoi(value);
            else if (key == "date") info->date = value;
        }
    }
    // After header: read two bytes (event_type, event_size)
    char type_byte = 0, size_byte = 0;
    ifs.read(&type_byte, 1);
    ifs.read(&size_byte, 1);
    if (ifs.gcount() < 1) {
        return false; // missing type byte
    }
    if (!ifs) return false; // missing size byte
    if (info) {
        info->event_type = static_cast<unsigned char>(type_byte);
        info->event_size = static_cast<unsigned char>(size_byte);
    }
    return true;
}

bool stream_dat_events(const string &path,
                       const function<void(const Event &)> &callback,
                       DatHeaderInfo *out_header,
                       u64 *out_event_count,
                       u32 *out_first_ts,
                       u32 *out_last_ts,
                       double *out_wall_seconds,
                       u64 *out_data_span_us,
                       bool realtime) {
    ifstream ifs(path, ios::binary);
    if (!ifs) {
        cerr << "Failed to open DAT file: " << path << "\n";
        return false;
    }

    DatHeaderInfo header; // local header info
    if (!parse_header(ifs, &header)) {
        cerr << "Failed to parse DAT header or missing type/size bytes." << "\n";
        return false;
    }
    if (header.event_size != 8) {
        cerr << "Unsupported event size " << header.event_size << " (expected 8)." << endl;
        return false;
    }

    if (out_header) *out_header = header;

    constexpr size_t CHUNK_BYTES = 1 << 20; // 1 MB chunk
    auto wall_start = chrono::steady_clock::now();
    vector<char> buffer(CHUNK_BYTES);
    u64 count = 0;
    bool have_first = false;
    u32 first_ts = 0;
    u32 last_ts = 0;

    // For realtime pacing
    chrono::steady_clock::time_point wall_base;
    u32 ts_base = 0;

    while (ifs) {
        ifs.read(buffer.data(), buffer.size());
        streamsize got = ifs.gcount();
        if (got <= 0) break;
        // Iterate full RawRecord-sized slices
        size_t offset = 0;
        while (offset + sizeof(RawRecord) <= static_cast<size_t>(got)) {
            RawRecord r;
            memcpy(&r, buffer.data() + offset, sizeof(RawRecord));
            offset += sizeof(RawRecord);
            Event e = decode(r);
            // Initialize bases on first event (for both stats and realtime pacing)
            if (callback) callback(e);
            ++count;
            if (!have_first) { first_ts = e.t; have_first = true; }
            last_ts = e.t; // overwrite each time

            if (realtime) {
                if (count == 1) {
                    wall_base = chrono::steady_clock::now();
                    ts_base = e.t;
                } else {
                    u32 delta_us = e.t - ts_base; // microseconds since first event
                    auto target = wall_base + chrono::microseconds(delta_us);
                    auto now = chrono::steady_clock::now();
                    if (target > now) {
                        this_thread::sleep_until(target);
                    }
                }
            }
        }
        // Any trailing bytes (< sizeof(RawRecord)) are carried implicitly by overwriting buffer next loop.
    }

    auto wall_end = chrono::steady_clock::now();
    if (out_event_count) *out_event_count = count;
    if (out_first_ts) *out_first_ts = have_first ? first_ts : 0u;
    if (out_last_ts) *out_last_ts = have_first ? last_ts : 0u;
    if (out_wall_seconds) *out_wall_seconds = chrono::duration<double>(wall_end - wall_start).count();
    if (out_data_span_us) *out_data_span_us = (have_first && count > 1) ? static_cast<u64>(last_ts - first_ts) : 0ull;

    return true;
}
