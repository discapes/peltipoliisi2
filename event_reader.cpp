#include "event_reader.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <thread>

namespace {
#pragma pack(push,1)
struct RawRecord {
    std::uint32_t t32;  // little-endian timestamp
    std::uint32_t w32;  // packed polarity + y + x
};
#pragma pack(pop)

Event decode(const RawRecord &r) {
    Event e;
    e.t = r.t32; // already little-endian on little-endian host; if not, add byte swap
    std::uint32_t w = r.w32;
    std::uint32_t x = w & 0x3FFFu;              // bits 13..0
    std::uint32_t y = (w >> 14) & 0x3FFFu;      // bits 27..14
    std::uint32_t raw_pol = (w >> 28) & 0xFu;   // bits 31..28
    e.x = static_cast<uint16_t>(x);
    e.y = static_cast<uint16_t>(y);
    e.polarity = static_cast<uint8_t>(raw_pol > 0 ? 1 : 0);
    return e;
}
}

static bool parse_header(std::ifstream &ifs, DatHeaderInfo *info) {
    // Read line by line until "% end" encountered.
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("%", 0) != 0) {
            // Line doesn't start with %, treat as start of binary (rewind position?)
            // Move stream position back to beginning of this line's content for binary.
            auto back = static_cast<std::streamoff>(line.size() + 1); // +1 for newline consumed
            ifs.seekg(-back, std::ios::cur);
            return true; // header ended implicitly
        }
        if (line == "% end") {
            return true; // finished header; binary starts after this line's newline
        }
        // Parse key-value pairs of form: % Key value(s)
        if (info) {
            // Remove leading '%' and spaces
            std::string content = line.substr(1); // drop '%'
            while (!content.empty() && content[0] == ' ') content.erase(0,1);
            // Split first token
            auto pos = content.find(' ');
            std::string key = content.substr(0, pos);
            std::string value = (pos == std::string::npos) ? std::string() : content.substr(pos+1);
            if (key == "Width") info->width = std::stoi(value);
            else if (key == "Height") info->height = std::stoi(value);
            else if (key == "Version") info->version = std::stoi(value);
            else if (key == "date") info->date = value;
        }
    }
    return false; // EOF before end marker
}

bool stream_dat_events(const std::string &path,
                       const std::function<void(const Event &)> &callback,
                       DatHeaderInfo *out_header,
                       std::uint64_t *out_event_count,
                       std::uint32_t *out_first_ts,
                       std::uint32_t *out_last_ts,
                       double *out_wall_seconds,
                       std::uint64_t *out_data_span_us,
                       bool realtime) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open DAT file: " << path << "\n";
        return false;
    }

    DatHeaderInfo header; // local header info
    if (!parse_header(ifs, &header)) {
        std::cerr << "Failed to parse DAT header or missing % end marker." << "\n";
        return false;
    }

    if (out_header) *out_header = header;

    constexpr std::size_t CHUNK_BYTES = 1 << 20; // 1 MB chunk
    auto wall_start = std::chrono::steady_clock::now();
    std::vector<char> buffer(CHUNK_BYTES);
    std::uint64_t count = 0;
    bool have_first = false;
    std::uint32_t first_ts = 0;
    std::uint32_t last_ts = 0;

    // For realtime pacing
    std::chrono::steady_clock::time_point wall_base;
    std::uint32_t ts_base = 0;

    while (ifs) {
        ifs.read(buffer.data(), buffer.size());
        std::streamsize got = ifs.gcount();
        if (got <= 0) break;
        // Iterate full RawRecord-sized slices
        std::size_t offset = 0;
        while (offset + sizeof(RawRecord) <= static_cast<std::size_t>(got)) {
            RawRecord r;
            std::memcpy(&r, buffer.data() + offset, sizeof(RawRecord));
            offset += sizeof(RawRecord);
            Event e = decode(r);
            // Initialize bases on first event (for both stats and realtime pacing)
            if (callback) callback(e);
            ++count;
            if (!have_first) { first_ts = e.t; have_first = true; }
            last_ts = e.t; // overwrite each time

            if (realtime) {
                if (count == 1) {
                    wall_base = std::chrono::steady_clock::now();
                    ts_base = e.t;
                } else {
                    std::uint32_t delta_us = e.t - ts_base; // microseconds since first event
                    auto target = wall_base + std::chrono::microseconds(delta_us);
                    auto now = std::chrono::steady_clock::now();
                    if (target > now) {
                        std::this_thread::sleep_until(target);
                    }
                }
            }
        }
        // Any trailing bytes (< sizeof(RawRecord)) are carried implicitly by overwriting buffer next loop.
    }

    auto wall_end = std::chrono::steady_clock::now();
    if (out_event_count) *out_event_count = count;
    if (out_first_ts) *out_first_ts = have_first ? first_ts : 0u;
    if (out_last_ts) *out_last_ts = have_first ? last_ts : 0u;
    if (out_wall_seconds) *out_wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    if (out_data_span_us) *out_data_span_us = (have_first && count > 1) ? static_cast<std::uint64_t>(last_ts - first_ts) : 0ull;

    return true;
}
