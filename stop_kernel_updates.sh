#!/bin/bash

kernel=$(uname -r)
sudo apt-mark hold linux-headers-$kernel linux-headers-generic linux-image-$kernel linux-image-generic
# to unhold, simply use 'apt-mark unhold'