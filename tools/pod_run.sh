#!/usr/bin/env bash
# Run a command on a RunPod pod reached through the SSH proxy.
#
# The proxy at ssh.runpod.io does not honour a remote command argument. Given
# `ssh pod "ls"` it prints its banner, opens an interactive shell, and waits,
# so the call hangs forever and looks like an authentication or network fault.
# It is neither: the shell is up, it just never receives the command.
#
# So commands go over stdin instead, which the interactive shell does read. The
# session echoes everything back, and the terminal decorates it with bracketed
# paste and title escapes, so output is fenced between markers and stripped.
#
# Usage:  tools/pod_run.sh <pod-ssh-address> <key> <<'EOF'
#           whatever commands
#         EOF
set -uo pipefail

ADDRESS="${1:?pod ssh address}"
KEY="${2:?path to ssh key}"
BEGIN="__POD_RUN_BEGIN__"
END="__POD_RUN_END__"

payload=$(cat)
transcript=$(mktemp)
trap 'rm -f "$transcript"' EXIT

{
  printf 'echo %s\n' "$BEGIN"
  printf '%s\n' "$payload"
  printf 'echo %s:$?\n' "$END"
  printf 'exit\n'
} | ssh -tt \
      -o StrictHostKeyChecking=no \
      -o IdentitiesOnly=yes \
      -o ConnectTimeout=20 \
      -o ServerAliveInterval=30 \
      "$ADDRESS" -i "$KEY" > "$transcript" 2>&1

# Strip carriage returns and ANSI/OSC decoration, then keep only what the
# remote shell produced between the markers. The first marker line is the
# echoed command itself, so drop everything up to and including the *last*
# bare occurrence.
tr -d '\r' < "$transcript" \
  | sed -e 's/\x1b\][0-9]*;[^\x07]*\x07//g' -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
  | awk -v b="$BEGIN" -v e="$END" '
      $0 ~ "^" b "$" { collecting = 1; buffer = ""; next }
      collecting && $0 ~ "^" e ":" { split($0, parts, ":"); code = parts[2]; collecting = 0; done = 1; next }
      collecting { buffer = buffer $0 "\n" }
      END { printf "%s", buffer; exit (done ? code + 0 : 1) }
    '
