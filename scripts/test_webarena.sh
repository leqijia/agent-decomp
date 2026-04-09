#!/usr/bin/env bash
#
# Quick health check for the 4 WebArena services.
#
# Usage:
#   ./test_webarena.sh <public-hostname-or-ip>
#
# Example:
#   ./test_webarena.sh 172.185.52.29

set -euo pipefail

HOSTNAME_ARG="${1:-}"
if [[ -z "$HOSTNAME_ARG" ]]; then
  echo "Usage: $0 <public-hostname-or-ip>"
  exit 1
fi
HOSTNAME="$HOSTNAME_ARG"

check_url() {
  local label="$1" url="$2"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$url" || echo "---")
  local status="OK"
  if [[ "$code" != "200" && "$code" != "301" && "$code" != "302" ]]; then
    status="FAIL"
  fi
  printf "  %-6s %-15s %-50s  HTTP %s\n" "$status" "$label" "$url" "$code"
}

echo "WebArena health check against $HOSTNAME"
echo ""
check_url "Shopping"       "http://${HOSTNAME}:7770"
check_url "Admin"          "http://${HOSTNAME}:7780"
check_url "Forum"          "http://${HOSTNAME}:9999"
check_url "GitLab"         "http://${HOSTNAME}:8023"

echo ""
echo "Docker container status:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | \
  grep -E "shopping|forum|gitlab|NAMES" || echo "  (no containers running)"
