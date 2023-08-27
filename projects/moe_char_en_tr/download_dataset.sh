#!/bin/bash

# Assert that the current directory is moe-transformer
if [[ "$(pwd)" != *"/moe-transformer" ]]; then
    echo "Error: Current working directory is not moe-transformer! I highly recommend switching to it so that this script runs properly."
    exit 1
fi

# Continue with the script if the assertion passes
wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-de-en-v1.0.tar.gz
tar -xvf opus-100-corpus-en-tr-v1.0.tar.gz
python projects/moe_char_en_tr/construct_encoder.py
