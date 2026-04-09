# scripts/

Operational scripts for the WebArena Docker environment on the project VM.

## setup_webarena.sh

One-time setup. Installs Docker, downloads the 4 WebArena image tarballs
from CMU's mirror, loads them, starts the containers, configures them for
the given public hostname, and runs a final health check.

Usage (on the VM):

```bash
cd ~/agent-decomp
./scripts/setup_webarena.sh 172.185.52.29
```

Takes roughly 30–60 minutes end-to-end (most of it downloading ~45 GB of
images from CMU). Logs everything to `~/webarena-setup.log`.

Services brought up:

| Service         | Port |
|-----------------|------|
| Shopping        | 7770 |
| Shopping Admin  | 7780 |
| GitLab          | 8023 |
| Forum (Reddit)  | 9999 |

Wikipedia and Map are skipped because our task categories don't need them.

The script is re-runnable: if a previous run died mid-way, rerunning will
skip downloads/loads/container creation that already completed.

## test_webarena.sh

Health check. `curl`s each service and prints HTTP status codes plus the
running Docker container list. Use this to verify the environment is up
after a reboot or after starting containers.

```bash
./scripts/test_webarena.sh 172.185.52.29
```
