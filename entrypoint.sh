#!/usr/bin/env bash
set -euo pipefail

# entrypoint.sh
# Purpose: Ensure AWS profile 'equipo56' is used even when credentials are provided
# via environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION).
# Then, pull DVC artifacts (S3 remote) at runtime (no-scm) and start the API server.

# 1) Resolve region and capture env credentials
REGION="${AWS_DEFAULT_REGION:-us-east-2}"
ACCESS_KEY="${AWS_ACCESS_KEY_ID:-}"
SECRET_KEY="${AWS_SECRET_ACCESS_KEY:-}"
SESSION_TOKEN="${AWS_SESSION_TOKEN:-}"

# 2) Write AWS shared credentials/config using profile 'equipo56'
mkdir -p /root/.aws
{
  echo "[equipo56]"
  echo "aws_access_key_id=${ACCESS_KEY}"
  echo "aws_secret_access_key=${SECRET_KEY}"
  # Only write session token line if present
  if [ -n "$SESSION_TOKEN" ]; then
    echo "aws_session_token=${SESSION_TOKEN}"
  fi
} > /root/.aws/credentials

{
  echo "[profile equipo56]"
  echo "region=${REGION}"
  echo "output=json"
} > /root/.aws/config

# 3) Force SDKs to use the profile instead of direct env credentials
export AWS_PROFILE=equipo56
export AWS_DEFAULT_REGION="${REGION}"
export AWS_SDK_LOAD_CONFIG=1
# Unset direct env credentials to ensure boto3 uses the profile
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN || true

# Debug: show sanitized AWS config being used
echo "AWS profile in use: ${AWS_PROFILE}"
echo "AWS region: ${REGION}"
if [ -f /root/.aws/credentials ]; then
  echo ": ~/.aws/credentials"
  # mask secret entirely, and mask access key id leaving first 4 chars
  sed -E 's/(aws_secret_access_key=).*/\1********/; s/(aws_access_key_id=)([A-Z0-9]{4}).*/\1\2********/' /root/.aws/credentials || true
fi
if [ -f /root/.aws/config ]; then
  echo ": ~/.aws/config"
  cat /root/.aws/config || true
fi

# Derive S3 endpoint for the configured region (helps avoid region mismatch 400s)
if [ "${REGION}" = "us-east-1" ]; then
  S3_ENDPOINT="https://s3.amazonaws.com"
else
  S3_ENDPOINT="https://s3.${REGION}.amazonaws.com"
fi
export S3_ENDPOINT

# Debug: verify identity and bucket reachability via botocore
python3 - <<'PY'
import os, json
try:
    from botocore.session import Session
    profile = os.environ.get("AWS_PROFILE", "")
    region = os.environ.get("AWS_DEFAULT_REGION", "")
    sess = Session(profile=profile)
    sts = sess.create_client("sts", region_name=region)
    ident = sts.get_caller_identity()
    print("STS identity:", json.dumps({"Account": ident.get("Account"), "Arn": ident.get("Arn")}))
    s3 = sess.create_client("s3", region_name=region, endpoint_url=os.environ.get("S3_ENDPOINT"))
    s3.head_bucket(Bucket="itesm-mna")
    print("S3 head_bucket ok for 'itesm-mna'")
except Exception as e:
    print("Credential/S3 check failed:", repr(e))
PY

# 4) Enable no-scm mode in repo config and pull DVC artifacts if missing
MODEL_PATH="models/best_gridsearch_amplio.joblib"
if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH. Enabling DVC no-scm mode and pulling from remote using profile 'equipo56'..."
  # Ensure DVC operates without a Git repository
  dvc config core.no_scm true
  # Ensure remote uses expected profile & region & explicit endpoint
  dvc remote modify team_remote profile equipo56 || true
  dvc remote modify team_remote region "${REGION}" || true
  dvc remote modify team_remote endpointurl "${S3_ENDPOINT}" || true
  # Optional: set as default remote if not already
  dvc remote default team_remote || true
  # Show effective DVC config (safe: contains url/region/profile)
  dvc config -l || true
  # Create parent dir just in case
  mkdir -p "$(dirname "$MODEL_PATH")"
  if ! dvc pull -v "$MODEL_PATH"; then
    echo "WARNING: dvc pull failed. The application may fallback to MLflow if configured."
  fi
fi

# 5) Default MLflow tracking to local file backend if not provided
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  export MLFLOW_TRACKING_URI="file:/app/mlruns"
  echo "MLFLOW_TRACKING_URI not set. Defaulting to ${MLFLOW_TRACKING_URI}"
fi

# 6) Start the API server
exec uvicorn app_api:app --host 0.0.0.0 --port 8000
