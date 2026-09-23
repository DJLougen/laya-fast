"""Serialize GPU timing runs across concurrent agents/processes.

Usage:  .venv/bin/python gpulock.py -- <command> [args...]
Holds an exclusive fcntl lock on /tmp/laya_gpu.lock for the lifetime of the
command, so two benchmarks never share the GPU and corrupt each other's timings.
"""
import fcntl
import subprocess
import sys

LOCK = "/tmp/laya_gpu.lock"


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main())
