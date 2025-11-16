#include <deque>
#include <fstream>
#include <thread>
#include "defs.hpp"

// Shared active-event deque (events within the sliding window), protected by a mutex.
static std::deque<FrameEvent> g_active;
static std::mutex g_active_mtx;

namespace
{
#pragma pack(push, 1)
    struct RawRecord
    {
        u32 t32;
        u32 w32;
    }; // 8-byte raw event
#pragma pack(pop)

    FrameEvent decode(const RawRecord &r)
    {
        FrameEvent e;
        e.t = r.t32; // already LE on LE hosts
        u32 w = r.w32;
        u32 x = w & 0x3FFFu;            // bits 13..0
        u32 y = (w >> 14) & 0x3FFFu;    // bits 27..14
        u32 raw_pol = (w >> 28) & 0xFu; // bits 31..28
        e.x = static_cast<uint16_t>(x);
        e.y = static_cast<uint16_t>(y);
        e.polarity = static_cast<uint8_t>(raw_pol ? 1 : 0);
        return e;
    }
}

static bool parse_header(ifstream &ifs, DatHeaderInfo *info)
{
    // Read line by line until "% end" encountered.
    string line;
    while (getline(ifs, line))
    {
        if (line.rfind("%", 0) != 0)
        {
            // Line doesn't start with %, treat as start of binary (rewind position?)
            // Move stream position back to beginning of this line's content for binary.
            auto back = static_cast<streamoff>(line.size() + 1); // +1 for newline consumed
            ifs.seekg(-back, ios::cur);
            break; // header ended implicitly
        }
        if (line == "% end")
        {
            break; // finished header; binary starts after this line's newline
        }
        // Parse key-value pairs of form: % Key value(s)
        if (info)
        {
            string content = line.substr(1); // drop '%'
            while (!content.empty() && content[0] == ' ')
                content.erase(0, 1);
            auto pos = content.find(' ');
            string key = content.substr(0, pos);
            string value = (pos == string::npos) ? string() : content.substr(pos + 1);
            if (key == "Width")
                info->width = stoi(value);
            else if (key == "Height")
                info->height = stoi(value);
            else if (key == "Version")
                info->version = stoi(value);
            else if (key == "date")
                info->date = value;
        }
    }
    // After header: read two bytes (event_type, event_size)
    char type_byte = 0, size_byte = 0;
    ifs.read(&type_byte, 1);
    ifs.read(&size_byte, 1);
    if (ifs.gcount() < 1)
    {
        return false; // missing type byte
    }
    if (!ifs)
        return false; // missing size byte
    if (info)
    {
        info->event_type = static_cast<unsigned char>(type_byte);
        info->event_size = static_cast<unsigned char>(size_byte);
    }
    return true;
}

bool stream_dat_events(const string &path,
                       const function<void(const FrameEvent &, int)> &callback,
                       DatHeaderInfo *out_header,
                       u32 window_us,
                       const atomic<bool> &stop_flag)
{
    ifstream ifs(path, ios::binary);
    if (!ifs)
    {
        cerr << "Failed to open DAT file: " << path << "\n";
        return false;
    }

    DatHeaderInfo header; // local header info
    if (!parse_header(ifs, &header))
    {
        cerr << "Failed to parse DAT header or missing type/size bytes." << "\n";
        return false;
    }
    if (header.event_size != 8)
    {
        cerr << "Unsupported event size " << header.event_size << " (expected 8)." << endl;
        return false;
    }

    if (out_header)
        *out_header = header;

    // We'll stream one event at a time and interleave expirations.
    // Since events are time-ordered, expirations (t + window) are also time-ordered.
    // Maintain a FIFO deque of active events; the front is the next to expire.
    // This is shared via g_active so other threads can snapshot it.

    // Realtime pacing anchors
    chrono::steady_clock::time_point wall_base;
    bool have_base = false;
    u32 ts_base = 0;

    // Helper to convert a DAT timestamp to target wall time using the first event as t0
    auto to_wall_time = [&](u32 ts)
    {
        return wall_base + chrono::microseconds(static_cast<int64_t>(ts - ts_base));
    };

    // Read the first event (to initialize pacing)
    RawRecord rr;
    FrameEvent next_ev{};
    bool has_next_ev = false;
    ifs.read(reinterpret_cast<char *>(&rr), sizeof(rr));
    if (ifs.gcount() == sizeof(rr))
    {
        next_ev = decode(rr);
        has_next_ev = true;
        wall_base = chrono::steady_clock::now();
        have_base = true;
        ts_base = next_ev.t;
    }

    while ((has_next_ev || !g_active.empty()) && stop_flag.load())
    {
        // Determine next arrival and next expiry timestamps
        u32 next_arrival_ts = has_next_ev ? next_ev.t : numeric_limits<u32>::max();
        u32 next_expiry_ts;
        {
            lock_guard<mutex> lk(g_active_mtx);
            next_expiry_ts = !g_active.empty() ? static_cast<u32>(g_active.front().t + window_us) : numeric_limits<u32>::max();
        }
        bool do_expiry = next_expiry_ts <= next_arrival_ts;

        u32 schedule_ts = do_expiry ? next_expiry_ts : next_arrival_ts;
        if (have_base)
        {
            auto target = to_wall_time(schedule_ts);
            auto now = chrono::steady_clock::now();
            if (target > now)
                this_thread::sleep_until(target);
        }

        if (do_expiry)
        {
            // Expire all events whose expiry time <= schedule_ts
            while (true)
            {
                FrameEvent ev;
                bool have = false;
                {
                    lock_guard<mutex> lk(g_active_mtx);
                    if (!g_active.empty())
                    {
                        u32 exp_ts = static_cast<u32>(g_active.front().t + window_us);
                        if (exp_ts <= schedule_ts)
                        {
                            ev = g_active.front();
                            g_active.pop_front();
                            have = true;
                        }
                    }
                }
                if (have)
                {
                    if (callback)
                        callback(ev, -1);
                }
                else
                {
                    break;
                }
            }
        }
        else
        {
            // Emit arrival (+1)
            if (callback)
                callback(next_ev, +1);
            // Track active event for future expiry
            {
                lock_guard<mutex> lk(g_active_mtx);
                g_active.push_back(next_ev);
            }

            // Load next event from file
            ifs.read(reinterpret_cast<char *>(&rr), sizeof(rr));
            if (ifs.gcount() == sizeof(rr))
            {
                next_ev = decode(rr);
                has_next_ev = true;
                // If we somehow didn't have base (shouldn't happen), initialize
                if (!have_base)
                {
                    wall_base = chrono::steady_clock::now();
                    have_base = true;
                    ts_base = next_ev.t;
                }
            }
            else
            {
                has_next_ev = false;
            }
        }
    }

    return true;
}

void snapshot_active_events(vector<FrameEvent> &out)
{
    lock_guard<mutex> lk(g_active_mtx);
    out.assign(g_active.begin(), g_active.end());
}
