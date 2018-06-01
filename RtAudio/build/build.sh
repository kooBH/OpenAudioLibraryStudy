#!/bin/sh
echo 필수 라이브러리를 설치합니다.
sudo apt-get update
sudo apt-get install cmake
sudo apt-get install libasound2-dev

cmake ..
make
./setting.sh
