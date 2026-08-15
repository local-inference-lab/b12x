#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  capture_sglang_native_profile.sh start [options]
  capture_sglang_native_profile.sh stop [options]

Attach-only helper for SGLang's native profiling endpoints.
It does not send inference traffic; use it against an already-busy server.

Options:
  --base-url URL         Server base URL. Default: http://127.0.0.1:30000
  --mode MODE            one of: decode, mtp, all. Default: decode
  --num-steps N          Auto-stop after N scheduler steps
  --output-dir DIR       Server-side output directory for trace files
  --merge-profiles       Ask SGLang to merge rank traces
  --profile-prefix NAME  Prefix for emitted trace filenames
  -h, --help             Show this message

Notes:
  mode=decode -> profile_by_stage=true, profile_stages=["decode"]
  mode=mtp    -> profile_by_stage=true, profile_stages=["prefill"]
                 This is the closest native bucket for speculative
                 extend/verify-family kernels after TTFT.
  mode=all    -> plain /start_profile without stage filtering

Examples:
  capture_sglang_native_profile.sh start --mode decode --num-steps 16
  capture_sglang_native_profile.sh start --mode mtp --num-steps 12 --output-dir /tmp/sglang-prof
  capture_sglang_native_profile.sh stop
EOF
}

action="${1:-}"
if [[ -z "${action}" ]]; then
  usage
  exit 1
fi
shift || true

base_url="http://127.0.0.1:30000"
mode="decode"
num_steps=""
output_dir=""
merge_profiles="false"
profile_prefix=""

# Configurable hard maxima to prevent a malicious/compromised endpoint from
# exhausting disk or hanging the helper. Override via environment variables.
: "${B12X_PROFILE_MAX_FILESIZE:=67108864}"   # 64 MiB ceiling on a single response body
: "${B12X_PROFILE_MAX_TIME:=3600}"           # absolute wall-clock deadline (seconds)

# Validate that env values are positive integers.
_validate_positive_int() {
  local val="$1" name="$2"
  if ! [[ "${val}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${name} must be a positive integer, got '${val}'" >&2
    exit 1
  fi
}
_validate_positive_int "${B12X_PROFILE_MAX_FILESIZE}" "B12X_PROFILE_MAX_FILESIZE"
_validate_positive_int "${B12X_PROFILE_MAX_TIME}" "B12X_PROFILE_MAX_TIME"
readonly B12X_PROFILE_MAX_FILESIZE B12X_PROFILE_MAX_TIME

# Require curl >= 8.4.0 for reliable --max-filesize on chunked/lengthless
# responses.  Older versions only enforce --max-filesize when Content-Length
# is known, leaving chunked bodies unbounded.
_curl_version="$(curl --version 2>/dev/null | head -1 | awk '{print $2}')"
if [[ -z "${_curl_version}" ]]; then
  echo "Error: curl is required but was not found" >&2
  exit 1
fi
_curl_major="${_curl_version%%.*}"
_curl_rest="${_curl_version#*.}"
_curl_minor="${_curl_rest%%.*}"
if [[ "${_curl_major}" -lt 8 ]] || { [[ "${_curl_major}" -eq 8 ]] && [[ "${_curl_minor}" -lt 4 ]]; }; then
  echo "Error: curl >= 8.4.0 is required for reliable max-filesize enforcement, found ${_curl_version}" >&2
  exit 1
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      base_url="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --num-steps)
      num_steps="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --merge-profiles)
      merge_profiles="true"
      shift
      ;;
    --profile-prefix)
      profile_prefix="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${action}" in
  start|stop)
    ;;
  *)
    echo "Unknown action: ${action}" >&2
    usage
    exit 1
    ;;
esac

case "${mode}" in
  decode|mtp|all)
    ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    usage
    exit 1
    ;;
esac

_RESPONSE_FILE=""

_cleanup_response_file() {
  [[ -n "${_RESPONSE_FILE}" ]] && rm -f "${_RESPONSE_FILE}"
  return 0
}

curl_post() {
  local url="$1"
  local body="$2"
  local http_code curl_exit actual_size

  trap _cleanup_response_file EXIT
  _RESPONSE_FILE="$(mktemp "${TMPDIR:-/tmp}/sglang_prof_resp.XXXXXXXX")"
  chmod 600 "${_RESPONSE_FILE}"

  http_code=$(curl -q -sS \
      --no-location \
      --proto '=http,https' \
      --max-time "${B12X_PROFILE_MAX_TIME}" \
      --max-filesize "${B12X_PROFILE_MAX_FILESIZE}" \
      -o "${_RESPONSE_FILE}" \
      -w '%{http_code}' \
      -X POST \
      -H 'Content-Type: application/json' \
      --data "${body}" \
      --url "${url}" 2>/dev/null) && curl_exit=0 || curl_exit=$?

  if [[ ${curl_exit} -ne 0 ]]; then
    if [[ ${curl_exit} -eq 63 ]]; then
      echo "Error: response exceeded max filesize (${B12X_PROFILE_MAX_FILESIZE} bytes)" >&2
    else
      echo "Error: curl failed with exit code ${curl_exit}" >&2
    fi
    return 1
  fi

  actual_size=$(wc -c < "${_RESPONSE_FILE}" | tr -d ' ')
  if [[ "${actual_size}" -gt "${B12X_PROFILE_MAX_FILESIZE}" ]]; then
    echo "Error: response body (${actual_size} bytes) exceeded max filesize (${B12X_PROFILE_MAX_FILESIZE} bytes)" >&2
    return 1
  fi

  cat "${_RESPONSE_FILE}"
  rm -f "${_RESPONSE_FILE}"
  _RESPONSE_FILE=""

  if [[ "${http_code}" != 2* ]]; then
    echo >&2
    echo "Request failed with HTTP ${http_code}" >&2
    return 1
  fi

  echo
  return 0
}

# Validate URL scheme.
_validate_url() {
  local url="$1"
  if ! [[ "${url}" =~ ^https?:// ]]; then
    echo "Error: base-url must start with http:// or https://" >&2
    exit 1
  fi
}
_validate_url "${base_url}"

if [[ "${action}" == "stop" ]]; then
  curl_post "${base_url%/}/stop_profile" '{}' || exit 1
  exit 0
fi

profile_by_stage="false"
profile_stages_json='null'

if [[ "${mode}" == "decode" ]]; then
  profile_by_stage="true"
  profile_stages_json='["decode"]'
elif [[ "${mode}" == "mtp" ]]; then
  profile_by_stage="true"
  profile_stages_json='["prefill"]'
fi

json='{'
json+="\"profile_by_stage\":${profile_by_stage}"
json+=",\"merge_profiles\":${merge_profiles}"
json+=",\"profile_stages\":${profile_stages_json}"

if [[ -n "${num_steps}" ]]; then
  json+=",\"num_steps\":${num_steps}"
fi

if [[ -n "${output_dir}" ]]; then
  json+=",\"output_dir\":\"${output_dir}\""
fi

if [[ -n "${profile_prefix}" ]]; then
  json+=",\"profile_prefix\":\"${profile_prefix}\""
fi

json+='}'

# Attempt start; on failure, best-effort bounded stop cleanup.
if ! curl_post "${base_url%/}/start_profile" "${json}"; then
  curl_post "${base_url%/}/stop_profile" '{}' 2>/dev/null || true
  exit 1
fi
