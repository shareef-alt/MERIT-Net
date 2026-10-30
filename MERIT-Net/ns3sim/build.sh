#!/bin/bash
# usage: ./build.sh src.cc out
NS=/usr/local/lib/python3.13/dist-packages/ns3
g++ -std=c++20 -O2 -I$NS/include "$1" -o "$2" -L$NS/lib64 $(ls $NS/lib64 | grep -E '^libns3.44-.*\.so$' | sed 's/^lib/-l/; s/\.so$//' | tr '\n' ' ') -Wl,-rpath,$NS/lib64
