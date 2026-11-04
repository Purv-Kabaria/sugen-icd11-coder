#!/usr/bin/env bash
# Crawl the MMS linearization in each WHO language, one container at a time.
# Usage: bash tools/crawl_who_languages.sh fr es pt
set -euo pipefail

release="2026-01"
port=8383
image="whoicd/icd-api:latest"

for language in "$@"; do
    name="icdapi-${language}"
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run -d --name "${name}" -p "${port}:80" \
        -e acceptLicense=true -e "include=${release}_${language}" "${image}" >/dev/null

    for _ in $(seq 1 60); do
        status=$(curl -s -o /dev/null -w '%{http_code}' \
            -H 'API-Version: v2' -H "Accept-Language: ${language}" -H 'Accept: application/json' \
            "http://localhost:${port}/icd/release/11/${release}/mms" || true)
        [ "${status}" = "200" ] && break
        sleep 10
    done

    uv run python tools/crawl_who_mms.py --base-url "http://localhost:${port}" \
        --release "${release}" --language "${language}" \
        --out "cache/who/mms-${release}.${language}.jsonl.gz"
    docker rm -f "${name}" >/dev/null
done
