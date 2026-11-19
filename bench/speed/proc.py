"""Process facts for the supervisor: resident memory of a process tree, machine load.

On macOS the memory comes from libproc (proc_pidinfo, proc_listchildpids) through
ctypes, so a check every few seconds costs microseconds and starts no process that
would compete with a measurement; elsewhere it falls back to `ps`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys

_PROC_PIDTASKINFO = 4


class _TaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32),
        ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


_libproc = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib", use_errno=True)
    except OSError:
        _libproc = None


def _children(pid: int) -> list[int]:
    buf = (ctypes.c_int * 1024)()
    n = _libproc.proc_listchildpids(pid, buf, ctypes.sizeof(buf))
    if n <= 0:
        return []
    # proc_listchildpids returns a count of pids on current macOS (bytes on old ones)
    count = n if n <= 1024 else n // ctypes.sizeof(ctypes.c_int)
    return [buf[i] for i in range(min(count, 1024)) if buf[i] > 0]


def _rss(pid: int) -> int:
    info = _TaskInfo()
    n = _libproc.proc_pidinfo(pid, _PROC_PIDTASKINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    return info.resident_size if n == ctypes.sizeof(info) else 0


def tree_pids(pid: int) -> list[int]:
    """pid and all its descendants."""
    if _libproc is None:
        return _ps_tree(pid)[0]
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(_children(p))
    return out


def tree_rss(pid: int) -> int:
    """Resident bytes of pid and all its descendants."""
    if _libproc is None:
        return _ps_tree(pid)[1]
    return sum(_rss(p) for p in tree_pids(pid))


def _ps_tree(pid: int) -> tuple[list[int], int]:
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return [pid], 0
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3:
            p, pp, r = map(int, parts)
            kids.setdefault(pp, []).append(p)
            rss[p] = r * 1024
    pids, todo = [], [pid]
    while todo:
        p = todo.pop()
        pids.append(p)
        todo.extend(kids.get(p, []))
    return pids, sum(rss.get(p, 0) for p in pids)


def load1() -> float:
    """The 1-minute load average."""
    return os.getloadavg()[0]


def busy_processes(limit: int = 5, min_cpu: float = 20.0) -> list[str]:
    """The processes using the most CPU right now (for a warning before a run)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "%cpu=,pid=,comm="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            try:
                cpu = float(parts[0])
            except ValueError:
                continue
            if cpu >= min_cpu and int(parts[1]) != os.getpid():
                rows.append((cpu, f"{parts[2].rsplit('/', 1)[-1]} (pid {parts[1]}, {cpu:.0f}%)"))
    return [r for _, r in sorted(rows, reverse=True)[:limit]]


# ---------------------------------------------------------------- CPU use of other processes
#
# The machine's busy CPU time comes from host_processor_info (per-CPU tick counters of the
# kernel, 1/100 s); the CPU time of a process from proc_pidinfo (total_user + total_system,
# in mach time units on Apple silicon). "Other" CPU over an interval is the machine's busy
# time minus the run's own process tree, divided by the wall time: the number of cores
# other processes kept busy. Reading the counters costs microseconds and starts nothing.

_CPU_STATE_USER, _CPU_STATE_SYSTEM, _CPU_STATE_IDLE, _CPU_STATE_NICE = 0, 1, 2, 3
_CLK_TCK = 100.0
_libc = None
_tb = 1.0
if _libproc is not None:
    try:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

        class _Timebase(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

        _t = _Timebase()
        _libc.mach_timebase_info(ctypes.byref(_t))
        _tb = _t.numer / _t.denom  # mach units -> ns
        _libc.mach_host_self.restype = ctypes.c_uint32
        _libc.mach_task_self.restype = ctypes.c_uint32
        _libc.host_processor_info.argtypes = [
            ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.POINTER(ctypes.c_uint32)]
        _libc.vm_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_size_t, ctypes.c_size_t]
        _HOST = _libc.mach_host_self()
    except (OSError, AttributeError):
        _libc = None


def machine_busy_s() -> float | None:
    """Busy CPU seconds of all CPUs since boot (user + system + nice ticks)."""
    if _libc is None:
        return None
    ncpu = ctypes.c_uint32()
    info = ctypes.POINTER(ctypes.c_int32)()
    count = ctypes.c_uint32()
    if _libc.host_processor_info(_HOST, 2, ctypes.byref(ncpu), ctypes.byref(info), ctypes.byref(count)) != 0:
        return None
    try:
        busy = 0
        for c in range(ncpu.value):
            base = c * 4
            busy += (info[base + _CPU_STATE_USER] & 0xFFFFFFFF) + (info[base + _CPU_STATE_SYSTEM] & 0xFFFFFFFF) \
                + (info[base + _CPU_STATE_NICE] & 0xFFFFFFFF)
        return busy / _CLK_TCK
    finally:
        _libc.vm_deallocate(_libc.mach_task_self(), ctypes.cast(info, ctypes.c_void_p).value,
                            count.value * ctypes.sizeof(ctypes.c_int32))


def proc_cpu_s(pid: int) -> float:
    """CPU seconds (user + system) a live process has used; 0 if it cannot be read."""
    if _libproc is None:
        return 0.0
    info = _TaskInfo()
    n = _libproc.proc_pidinfo(pid, _PROC_PIDTASKINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    if n != ctypes.sizeof(info):
        return 0.0
    return (info.total_user + info.total_system) * _tb * 1e-9


def tree_cpu_s(pid: int) -> float:
    return sum(proc_cpu_s(p) for p in tree_pids(pid))


class CpuMeter:
    """CPU cores other processes kept busy between start() and stop(), not counting the
    process tree of `own_pid` (the supervisor: its worker, R and caffeinate). The own
    tree's time is read at both ends, so a worker killed in between makes the interval
    unreliable: stop() then returns None."""

    def __init__(self, own_pid: int | None = None):
        self.own = own_pid or os.getpid()

    def start(self):
        import time

        self.t0 = time.monotonic()
        self.busy0 = machine_busy_s()
        self.pids0 = {p: proc_cpu_s(p) for p in tree_pids(self.own)}
        return self

    def stop(self) -> float | None:
        import time

        busy1 = machine_busy_s()
        if busy1 is None or self.busy0 is None:
            return None
        wall = time.monotonic() - self.t0
        if wall <= 0:
            return None
        pids1 = {p: proc_cpu_s(p) for p in tree_pids(self.own)}
        if any(p not in pids1 for p in self.pids0):  # a process of ours ended: its time is lost
            return None
        own = sum(pids1[p] - self.pids0.get(p, 0.0) for p in pids1)
        return max(0.0, (busy1 - self.busy0 - own) / wall)


def _all_pids() -> list[int]:
    n = _libproc.proc_listallpids(None, 0)
    if n <= 0:
        return []
    buf = (ctypes.c_int * (n + 64))()
    n = _libproc.proc_listallpids(buf, ctypes.sizeof(buf))
    return [buf[i] for i in range(max(0, n)) if buf[i] > 0]


def _name(pid: int) -> str:
    buf = ctypes.create_string_buffer(256)
    if _libproc.proc_name(pid, buf, ctypes.sizeof(buf)) > 0:
        return buf.value.decode(errors="replace")
    return "?"


def idle_check(window_s: float = 3.0, own_pid: int | None = None, top: int = 5) -> tuple[float | None, list[str]]:
    """(cores other processes kept busy over the next window_s seconds, the busiest of
    them as "name (pid, 1.2 cores)"). Processes of other users may be missing from the
    list (their CPU time cannot be read) but are in the total."""
    import time

    if _libc is None:
        return None, busy_processes(limit=top)
    own = set(tree_pids(own_pid or os.getpid()))
    meter = CpuMeter(own_pid).start()
    before = {p: proc_cpu_s(p) for p in _all_pids() if p not in own}
    time.sleep(window_s)
    after = {p: proc_cpu_s(p) for p in _all_pids() if p not in own}
    other = meter.stop()
    rows = []
    for p, t in after.items():
        d = (t - before.get(p, t)) / window_s
        if d >= 0.1:
            rows.append((d, f"{_name(p)} (pid {p}, {d:.1f}코어)"))
    return other, [r for _, r in sorted(rows, reverse=True)[:top]]
