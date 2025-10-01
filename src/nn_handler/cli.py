import argparse
import os
import re
import subprocess
import sys


def _env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _parse_gpus_per_node(raw: str) -> str:
    """
    SLURM_GPUS_PER_NODE can be '4', '2(gpuA100)', or '2(xgpus)'. We take the first integer.
    """
    if not raw:
        return "1"
    m = re.search(r"\d+", str(raw))
    return m.group(0) if m else "1"


def main():
    parser = argparse.ArgumentParser(
        prog="nn_handler_run",
        description="Thin wrapper around torchrun with sane SLURM defaults."
    )

    # Training script + its args (anything after -- goes to the script)
    parser.add_argument("script", help="Path to the Python training script.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="Arguments for the training script (use `--` to separate).")

    # Overridable torchrun knobs (default from SLURM/env)
    parser.add_argument("--nnodes", type=str,
                        default=_env_default("SLURM_JOB_NUM_NODES", "1"),
                        help="Number of nodes (default: SLURM_JOB_NUM_NODES or 1).")
    parser.add_argument("--nproc-per-node", type=str,
                        default=_parse_gpus_per_node(_env_default("SLURM_GPUS_PER_NODE", "1")),
                        help="Processes per node (default: parsed from SLURM_GPUS_PER_NODE or 1).")
    parser.add_argument("--node-rank", type=str,
                        default=_env_default("SLURM_NODEID", "0"),
                        help="Rank of the node (default: SLURM_NODEID or 0).")
    parser.add_argument("--rdzv-backend", type=str, default="c10d",
                        help="Rendezvous backend (default: c10d).")
    parser.add_argument("--rdzv-endpoint", type=str,
                        default=_env_default("RDZV_ENDPOINT", f"{_env_default('HOSTNAME', 'localhost')}:29500"),
                        help="host:port for rendezvous (default: RDZV_ENDPOINT or HOSTNAME:29500).")
    parser.add_argument("--rdzv-id", type=str,
                        default=_env_default("SLURM_JOB_ID", "0"),
                        help="Rendezvous ID (default: SLURM_JOB_ID or 0).")

    # Common optional torchrun extras you may want to override
    parser.add_argument("--max-restarts", type=str, default=None,
                        help="Max restarts for each worker (torchrun --max_restarts).")
    parser.add_argument("--monitor-interval", type=str, default=None,
                        help="Monitoring interval seconds (torchrun --monitor_interval).")
    parser.add_argument("--tee", type=str, choices=["0", "1", "2"], default=None,
                        help="Tee worker stdout/stderr: 0=disable,1=stdout,2=stderr.")
    parser.add_argument("--standalone", action="store_true",
                        help="Use torchrun --standalone (single node, auto master).")

    # Raw passthrough to torchrun if you need something not modeled above
    parser.add_argument("--torchrun-extra", nargs="*", default=[],
                        help="Extra tokens appended verbatim to torchrun (advanced).")

    args = parser.parse_args()

    # Strip the separator if present
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    cmd = ["torchrun"]

    if args.standalone:
        cmd.append("--standalone")
    else:
        cmd += [
            f"--nnodes={args.nnodes}",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--node_rank={args.node_rank}",
            f"--rdzv_backend={args.rdzv_backend}",
            f"--rdzv_endpoint={args.rdzv_endpoint}",
            f"--rdzv_id={args.rdzv_id}",
        ]

    # Optional extras if provided
    if args.max_restarts is not None:
        cmd.append(f"--max_restarts={args.max_restarts}")
    if args.monitor_interval is not None:
        cmd.append(f"--monitor_interval={args.monitor_interval}")
    if args.tee is not None:
        cmd.append(f"--tee={args.tee}")

    # Any raw extras for torchrun
    if args.torchrun_extra:
        cmd += list(args.torchrun_extra)

    # Finally the training script + its args
    cmd += [args.script, *script_args]

    # print("[nn_handler_run] Executing:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
