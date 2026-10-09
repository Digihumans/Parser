#!/bin/bash

arg=${1:?Usage ./stop_kernel_updates.sh <hold>/<unhold> (To hold/unhold kernel updates)}
kernel=$(uname -r)
sudo apt-mark $arg linux-headers-$kernel linux-headers-generic linux-image-$kernel linux-image-generic
# to unhold, simply use 'apt-mark unhold'