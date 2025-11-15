#!/usr/bin/env -S bash -euo pipefail

meson setup build && meson compile -C build && ./build/peltipoliisi2
