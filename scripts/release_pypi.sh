#!/bin/bash -p
#
# release_pypi.sh — thin wrapper that delegates to the Python launcher.
#
# A shell body cannot safely bootstrap credential handling: Bash startup
# hooks (BASH_ENV), imported functions (BASH_FUNC_*), and spoofed
# SHELLOPTS=privileged all execute before the script body and can
# exfiltrate an inherited TWINE_PASSWORD.  This wrapper therefore does
# nothing but exec the Python launcher, which owns all credential
# handling, environment scrubbing, and subprocess management.
#
exec /usr/bin/python3 "$(dirname "${BASH_SOURCE[0]}")/release_pypi.py" "$@"
