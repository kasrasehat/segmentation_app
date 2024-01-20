#!/bin/bash

python3 app.py -s 0.0.0.0 -p $LISTEN_PORT \
 --checkpoint_oneformer $CHECKPOINT_ONEFORMER \
 $EXTRA_OPTIONS
 
