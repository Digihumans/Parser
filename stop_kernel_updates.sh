#!/bin/bash

sudo apt-mark hold linux-image-generic linux-headers-generic linux-image-$(uname -r) linux-headers-$(uname -r)
# to unhold, simply use 'apt-mark unhold'
