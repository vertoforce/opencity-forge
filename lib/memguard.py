"""Bounded-memory helpers: check headroom before heavy work, sample peak RSS."""
import os, time, threading

RESERVE_GB = 12.0

def mem_available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1048576.0
    return 0.0

def free_disk_gb(path='/'):
    s = os.statvfs(path)
    return s.f_bavail*s.f_frsize/1e9

def rss_gb(pid=None):
    pid = pid or os.getpid()
    for line in open(f'/proc/{pid}/status'):
        if line.startswith('VmRSS:'):
            return int(line.split()[1])/1048576.0
    return 0.0

def require(stage, need_gb, need_disk_gb=2.0, wait_s=900):
    """Wait for headroom; fail loudly rather than walking into an OOM kill."""
    t0 = time.time()
    while True:
        av = mem_available_gb(); dk = free_disk_gb()
        if dk < need_disk_gb:
            raise MemoryError(f"{stage}: only {dk:.1f} GB disk free, "
                              f"need {need_disk_gb:.1f} GB - refusing")
        if av - need_gb >= RESERVE_GB:
            return av
        if time.time()-t0 > wait_s:
            raise MemoryError(
                f"{stage}: need {need_gb:.1f} GB with a {RESERVE_GB:.0f} GB "
                f"host reserve, only {av:.1f} GB available after "
                f"{wait_s}s - refusing to start rather than run out of memory")
        time.sleep(15)

class PeakRSS:
    def __init__(self, interval=0.5, cap_gb=None, label=""):
        self.interval=interval; self.peak=0.0; self.stop=False
        self.cap_gb=cap_gb; self.label=label; self.tripped=False
    def _run(self):
        while not self.stop:
            r = rss_gb()
            if r > self.peak: self.peak = r
            if self.cap_gb and r > self.cap_gb:
                self.tripped = True
            time.sleep(self.interval)
    def __enter__(self):
        self.t=threading.Thread(target=self._run,daemon=True); self.t.start()
        return self
    def __exit__(self,*a):
        self.stop=True; self.t.join(timeout=2)
