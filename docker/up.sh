#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CONFIG=docker-compose.yml

# Setup environment
# shellcheck disable=SC1091
source "${DIR}/env.sh"

# Bring up a notebook container, using container name as project name
echo "Bringing up notebook '${NAME}'"
docker-compose -f "${DIR}/${CONFIG}" -p "${NAME}" up --build -d

IP_PORT=$(docker inspect --format='{{.NetworkSettings.Ports}}' ${NAME})
echo "Notebook ${NAME} listening on ${IP_PORT}"