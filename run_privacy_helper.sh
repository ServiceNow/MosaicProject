#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JOB_SPEC=${JOB_SPEC:-"${SCRIPT_DIR}/privacy_helper_1gpu.yaml"}
DEFAULT_SUFFIX=$(date +%s)
JOB_NAME=${JOB_NAME:-"privacy_helper_1gpu_${DEFAULT_SUFFIX}"}
DNS_NAME=${DNS_NAME:-"dns-privacy-helper"}
ACCOUNT=${ACCOUNT:-"snow.research.rlar"}
TMP_JOB_SPEC=""

if [[ ! -f "${JOB_SPEC}" ]]; then
    echo "Job spec not found: ${JOB_SPEC}" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${TMP_JOB_SPEC}" && -f "${TMP_JOB_SPEC}" ]]; then
        rm -f "${TMP_JOB_SPEC}"
    fi
}

trap cleanup EXIT

TMP_JOB_SPEC=$(mktemp /tmp/privacy_helper_job.XXXXXX.yaml)

python - "${JOB_SPEC}" "${TMP_JOB_SPEC}" "${JOB_NAME}" "${DNS_NAME}" <<'PY'
import sys

src_path, dst_path, job_name, dns_name = sys.argv[1:5]

with open(src_path) as f:
    lines = f.readlines()

rewritten = []
replaced_job_name = False
in_internal_dns = False
replaced_dns_name = False

for line in lines:
    if not replaced_job_name and line.startswith("name: "):
        rewritten.append(f"name: {job_name}\n")
        replaced_job_name = True
        continue

    if line.startswith("options:"):
        in_internal_dns = False
        rewritten.append(line)
        continue

    if line.startswith("  internal-dns:"):
        in_internal_dns = True
        rewritten.append(line)
        continue

    if in_internal_dns and not replaced_dns_name and line.startswith("    name: "):
        rewritten.append(f"    name: {dns_name}\n")
        replaced_dns_name = True
        continue

    if in_internal_dns and line.startswith("    ports:"):
        in_internal_dns = False

    rewritten.append(line)

if not replaced_job_name:
    raise SystemExit("Could not find top-level job name in helper spec")

if not replaced_dns_name:
    raise SystemExit("Could not find internal DNS name in helper spec")

with open(dst_path, "w") as f:
    f.writelines(rewritten)
PY

helper_url="http://${DNS_NAME}:8012"

# The checked-in YAML is the source of truth for the helper job. This wrapper
# only fills in the runtime name so repeated submits do not collide.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "eai job submit --account ${ACCOUNT} -f ${TMP_JOB_SPEC}"
    echo "helper_url=${helper_url}"
    echo
    cat "${TMP_JOB_SPEC}"
    exit 0
fi

eai job submit --account "${ACCOUNT}" -f "${TMP_JOB_SPEC}"
echo
echo "helper_url=${helper_url}"
