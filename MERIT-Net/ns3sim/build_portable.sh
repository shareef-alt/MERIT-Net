#!/bin/bash
# usage: ./build_portable.sh meritnet-scenario.cc meritnet-scenario
NS=$(python3 -c "import importlib.util as u; print(list(u.find_spec('ns3').submodule_search_locations)[0])")
LIB=$NS/lib64; [ -d "$LIB" ] || LIB=$NS/lib
g++ -std=c++20 -O2 -I$NS/include "$1" -o "$2" -L$LIB \
  $(ls $LIB | grep -E '^libns3[.0-9]*-.*\.so$' | sed 's/^lib/-l/; s/\.so$//' | tr '\n' ' ') -Wl,-rpath,$LIB
