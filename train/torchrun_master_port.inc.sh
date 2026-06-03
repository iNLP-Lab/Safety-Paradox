# Shared by train/*.sh: set MASTER_PORT for torchrun rendezvous.
# If MASTER_PORT is already set (e.g. multi-node or Slurm), it is left unchanged.
# Otherwise bind to port 0 and use the OS-assigned free port to avoid EADDRINUSE.
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi
