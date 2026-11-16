"""Evicts files over 1 MB under the given directories from the Linux page cache (posix_fadvise DONTNEED; no root). WSL holds
its page cache as Windows memory, so this gives Windows memory back without stopping a job (win_guard.sh, 14 Nov).

    python scripts/evict_cache.py <dir> [<dir> ...]
"""
import os, sys
n = 0
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            try:
                if os.path.islink(p) or os.path.getsize(p) < 1 << 20:
                    continue
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
print("evicted", n, "files")
