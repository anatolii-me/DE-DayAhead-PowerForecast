from pathlib import Path
import os

# Function to determine the number of available threads for computation, taking into account environment variables and system limits
def get_available_threads() -> int:
    # Allow an explicit limit, for example on a shared server.
    override = os.getenv("MODEL_N_THREADS")
    if override is not None:
        if not override.isdigit() or int(override) < 1:
            raise ValueError("MODEL_N_THREADS must be a positive integer.")
        return int(override)

    # Python 3.13 reports CPUs available to this process and respects CPU affinity.
    process_count = getattr(os, "process_cpu_count", lambda: None)()
    limits = [process_count or os.cpu_count() or 1]

    # Respect common limits set by schedulers and numerical runtimes.
    allocation_variables = ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS", "OMP_NUM_THREADS")
    limits.extend(
        int(value)
        for name in allocation_variables
        if (value := os.getenv(name, "")).isdigit() and int(value) > 0
    )

    # A container may expose every host CPU while limiting its CPU quota.
    cgroup_v2 = Path("/sys/fs/cgroup/cpu.max")
    cgroup_v1_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    cgroup_v1_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if cgroup_v2.exists():
            quota_text, period_text = cgroup_v2.read_text().split()
            if quota_text != "max":
                limits.append(max(1, int(quota_text) // int(period_text)))
        elif cgroup_v1_quota.exists() and cgroup_v1_period.exists():
            quota = int(cgroup_v1_quota.read_text())
            period = int(cgroup_v1_period.read_text())
            if quota > 0:
                limits.append(max(1, quota // period))
    except (OSError, ValueError):
        pass

    return max(1, min(limits))