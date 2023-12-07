#!/bin/bash

# Set default values for environment variables required by notebook compose
# configuration file.

# Container name
: "${NAME:=micrograd}"
export NAME

# Exposed container port
: "${PORT:=10000}"
export PORT

# Container work volume name
: "${WORK_VOLUME:=${NAME}-work}"
export WORK_VOLUME

# Container secrets volume name
: "${SECRETS_VOLUME:=${NAME}-secrets}"
export SECRETS_VOLUME

# TOKEN for login
: "${TOKEN:=letmein}"
export TOKEN