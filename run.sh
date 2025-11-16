#!/usr/bin/env -S bash -euo pipefail

meson setup build --reconfigure ${PKG_CONFIG_PATH:+--pkg-config-path $PKG_CONFIG_PATH} --buildtype release -Doptimization=3 -Db_lto=true && meson compile -vC build && time ./build/peltipoliisi2 ./data/drone_idle.dat
