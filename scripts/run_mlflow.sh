#!/bin/bash

FILE_DIR=$(cd $(dirname $0);pwd)
ENV_PATH=$(cd ${FILE_DIR}/../; pwd)/.env

source $ENV_PATH
LOGS_ROOT=${LOGS_ROOT:-${FILE_DIR}/../logs}

TRACKING_DIR="${LOGS_ROOT}/mlflow/mlruns/"
if [ -d $TRACKING_DIR ]; then
  MLFLOW_UI="poetry run mlflow ui"
  $MLFLOW_UI "--backend-store-uri=${TRACKING_DIR}"
else
  echo -e "NOT Exists: \n\t${TRACKING_DIR}"
fi
