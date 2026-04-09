#!/usr/bin/env bash
#
# WebArena Docker setup script.
#
# Runs on a fresh Ubuntu 24.04 VM and brings up 4 WebArena services:
#   - Shopping (Magento storefront)       :7770
#   - Shopping Admin (Magento CMS)        :7780
#   - GitLab                              :8023
#   - Forum (Postmill)                    :9999
#
# Skipped: Wikipedia (~80GB) and Map (~180GB), since our task categories
# (Shopping, Reddit, GitLab, Content Management) don't require them.
#
# Usage:
#   ./setup_webarena.sh <public-hostname-or-ip>
#
# Example:
#   ./setup_webarena.sh 172.185.52.29
#
# Prereqs:
#   - Ubuntu 24.04 LTS, x86_64
#   - At least 100 GB free disk (128 GB total VM disk is the minimum with our
#     stream-loading trick for GitLab; the naive "download tar then load" path
#     would need 256 GB because GitLab alone is ~50 GB and would peak at
#     ~100 GB during `docker load`)
#   - At least 8 GB RAM (16 recommended)
#   - Ports 7770, 7780, 8023, 9999 open in the VM's firewall/NSG
#
# The script is idempotent where practical: rerunning after a partial
# failure will skip already-completed steps (installed Docker, loaded images,
# running containers).

set -euo pipefail

# ---------- config ----------
HOSTNAME_ARG="${1:-}"
if [[ -z "$HOSTNAME_ARG" ]]; then
  echo "ERROR: pass the public hostname/IP as the first argument."
  echo "Usage: $0 <public-hostname-or-ip>"
  exit 1
fi
HOSTNAME="$HOSTNAME_ARG"

IMAGES_DIR="$HOME/webarena-images"
LOG_FILE="$HOME/webarena-setup.log"

MIRROR="http://metis.lti.cs.cmu.edu/webarena-images"

# tarball name -> expected image tag after `docker load`
# shopping / shopping_admin / forum are small enough (~10 GB each) to download
# to disk then load. GitLab is ~50 GB and would need ~100 GB peak disk if we
# did the same (tar + loaded image coexisting), so it's handled separately
# by streaming the download directly into `docker load` (see below).
declare -A IMAGES=(
  ["shopping_final_0712.tar"]="shopping_final_0712"
  ["shopping_admin_final_0719.tar"]="shopping_admin_final_0719"
  ["postmill-populated-exposed-withimg.tar"]="postmill-populated-exposed-withimg"
)

# Streamed image: (image_tag, tarball_url)
GITLAB_IMAGE="gitlab-populated-final-port8023"
GITLAB_TAR="gitlab-populated-final-port8023.tar"

# ---------- logging ----------
exec > >(tee -a "$LOG_FILE") 2>&1
log()  { echo -e "\n[$(date +'%H:%M:%S')] >>> $*"; }
step() { echo -e "\n================================================================"; log "$*"; echo "================================================================"; }

step "WebArena setup starting (host=$HOSTNAME)"
log "Log file: $LOG_FILE"

# ---------- 1. apt updates + basic tools ----------
step "1/7  Installing base packages"
sudo apt-get update -y
sudo apt-get install -y curl wget ca-certificates gnupg lsb-release jq

# ---------- 2. install Docker ----------
step "2/7  Installing Docker"
if command -v docker >/dev/null 2>&1; then
  log "Docker already installed ($(docker --version)), skipping."
else
  # Official Docker install per https://docs.docker.com/engine/install/ubuntu/
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

  sudo usermod -aG docker "$USER"
  log "Docker installed. NOTE: group 'docker' added — new SSH sessions will use it"
  log "without sudo. We'll use 'sudo docker' in this script for safety."
fi

DOCKER="sudo docker"
$DOCKER --version

# ---------- 3. download image tarballs ----------
step "3/7  Downloading image tarballs to $IMAGES_DIR"
mkdir -p "$IMAGES_DIR"
cd "$IMAGES_DIR"

for tarball in "${!IMAGES[@]}"; do
  image="${IMAGES[$tarball]}"

  # Skip if image already loaded into Docker
  if $DOCKER image inspect "$image" >/dev/null 2>&1; then
    log "Image '$image' already loaded, skipping download of $tarball."
    rm -f "$tarball"
    continue
  fi

  if [[ -f "$tarball" ]]; then
    log "Tarball $tarball already present, skipping download."
  else
    log "Downloading $tarball ..."
    wget --show-progress -q -c "$MIRROR/$tarball" -O "$tarball.partial"
    mv "$tarball.partial" "$tarball"
  fi
done

# ---------- 4a. load small images into Docker, delete tarballs to reclaim disk ----------
step "4a/7  Loading small images (shopping / shopping_admin / forum)"
for tarball in "${!IMAGES[@]}"; do
  image="${IMAGES[$tarball]}"
  if $DOCKER image inspect "$image" >/dev/null 2>&1; then
    log "Image '$image' already loaded, skipping."
    rm -f "$tarball"
    continue
  fi
  log "Loading $tarball ..."
  $DOCKER load --input "$tarball"
  log "Removing $tarball to reclaim disk."
  rm -f "$tarball"
done

# ---------- 4b. stream-load GitLab (~50 GB, doesn't fit on disk as tar + loaded) ----------
step "4b/7  Stream-loading GitLab directly into Docker (no tar on disk)"
if $DOCKER image inspect "$GITLAB_IMAGE" >/dev/null 2>&1; then
  log "GitLab image '$GITLAB_IMAGE' already loaded, skipping."
else
  log "Streaming $GITLAB_TAR from $MIRROR directly into 'docker load'"
  log "This is ~50 GB and takes ~15-25 min. No resume if it fails — we retry."
  # wget streams to stdout, docker load reads from stdin.
  # --tries=3 for a bit of resilience against transient CMU hiccups.
  wget --tries=3 --timeout=60 -q -O - "$MIRROR/$GITLAB_TAR" \
    | sudo docker load
  log "GitLab streamed in successfully."
fi

log "Images currently loaded:"
$DOCKER image ls | grep -E "shopping|gitlab|postmill" || true

# ---------- 5. start containers ----------
step "5/7  Starting containers"

# Helper: (re)start a container only if not already running
ensure_container() {
  local name="$1"; shift
  if $DOCKER ps --format '{{.Names}}' | grep -qx "$name"; then
    log "Container '$name' already running, skipping."
    return
  fi
  if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$name"; then
    log "Container '$name' exists but is stopped, starting it."
    $DOCKER start "$name"
    return
  fi
  log "Creating and starting container '$name'."
  $DOCKER run -d --name "$name" "$@"
}

ensure_container shopping        -p 7770:80 shopping_final_0712
ensure_container shopping_admin  -p 7780:80 shopping_admin_final_0719
ensure_container forum           -p 9999:80 postmill-populated-exposed-withimg
ensure_container gitlab          -p 8023:8023 \
                 gitlab-populated-final-port8023 \
                 /opt/gitlab/embedded/bin/runsvdir-start

log "Waiting 60s for Shopping/Admin/Forum to warm up ..."
sleep 60

# ---------- 6. configure services with the public hostname ----------
step "6/7  Configuring services for host=$HOSTNAME"

log "Configuring Shopping (Magento base URL)"
$DOCKER exec shopping /var/www/magento2/bin/magento setup:store-config:set \
  --base-url="http://${HOSTNAME}:7770"
$DOCKER exec shopping mysql -u magentouser -pMyPassword magentodb \
  -e "UPDATE core_config_data SET value='http://${HOSTNAME}:7770/' WHERE path = 'web/secure/base_url';"
$DOCKER exec shopping /var/www/magento2/bin/magento cache:flush

log "Configuring Shopping Admin (Magento base URL + disable password rotation)"
$DOCKER exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
$DOCKER exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
$DOCKER exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set \
  --base-url="http://${HOSTNAME}:7780"
$DOCKER exec shopping_admin mysql -u magentouser -pMyPassword magentodb \
  -e "UPDATE core_config_data SET value='http://${HOSTNAME}:7780/' WHERE path = 'web/secure/base_url';"
$DOCKER exec shopping_admin /var/www/magento2/bin/magento cache:flush

log "Configuring GitLab — this one takes ~5 min the first time"
# Wait for GitLab to be reachable inside the container before reconfiguring.
# gitlab-ctl status returns 0 only once runit has brought the services up.
for i in {1..60}; do
  if $DOCKER exec gitlab gitlab-ctl status >/dev/null 2>&1; then
    log "GitLab runit is up (after ${i}0s)"
    break
  fi
  echo -n "."
  sleep 10
done
echo ""

$DOCKER exec gitlab update-permissions || true
$DOCKER exec gitlab sed -i \
  "s|^external_url.*|external_url 'http://${HOSTNAME}:8023'|" /etc/gitlab/gitlab.rb
$DOCKER exec gitlab gitlab-ctl reconfigure

# ---------- 7. health check ----------
step "7/7  Health check"
sleep 5

check_url() {
  local label="$1" url="$2"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$url" || echo "---")
  printf "  %-25s %-50s  HTTP %s\n" "$label" "$url" "$code"
}

echo ""
echo "Service status:"
check_url "Shopping"       "http://${HOSTNAME}:7770"
check_url "Shopping Admin" "http://${HOSTNAME}:7780"
check_url "Forum"          "http://${HOSTNAME}:9999"
check_url "GitLab"         "http://${HOSTNAME}:8023"

echo ""
log "Setup complete. Docker containers:"
$DOCKER ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "================================================================"
echo "  Done. Full log: $LOG_FILE"
echo ""
echo "  Next steps:"
echo "    1. Open each URL in your browser to verify the sites render:"
echo "         http://${HOSTNAME}:7770"
echo "         http://${HOSTNAME}:7780"
echo "         http://${HOSTNAME}:9999"
echo "         http://${HOSTNAME}:8023"
echo "    2. If GitLab returns 502, wait 2 min and try again (reconfigure"
echo "       is slow). If it persists, see 'GitLab 502' in mds/DOCKER_WEBARENA.md."
echo "    3. Log out and log back in (or run 'newgrp docker') so your user"
echo "       can run docker commands without sudo."
echo "================================================================"
