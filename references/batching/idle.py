"""Check the timing prerequisites, then run the supplied command under the caller's lock."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections import deque


def main() -> None:
    command = sys.argv[1:]
    interleaved = bool(command and command[0] == "--ab")
    if interleaved:
        command = command[1:]
    power = subprocess.check_output(["pmset", "-g"], text=True)
    source = subprocess.check_output(["pmset", "-g", "batt"], text=True)
    if not re.search(r"powermode\s+2", power) or "AC Power" not in source:
        sys.exit("Timing requires High Power and AC.")
    consecutive = 0
    window: deque[tuple[float, int]] = deque(maxlen=11)
    for _ in range(90):
        status = subprocess.check_output(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], text=True)
        values = [int(v) for v in re.findall(r'"Device Utilization %"=(\d+)', status)]
        processes = subprocess.check_output(["ps", "-A", "-o", "pcpu=,comm="], text=True)
        busy = [line.strip() for line in processes.splitlines() if float(line.split()[0]) > 50]
        print(f"GPU utilization before timing: {values}; busy CPU processes: {len(busy)}", flush=True)
        if interleaved:
            window.append((time.monotonic(), max(values, default=100)))
            readings = [v for _, v in window]
            if (
                len(window) == 11
                and window[-1][0] - window[0][0] >= 20
                and max(readings) < 20
                and max(readings) - min(readings) <= 3
            ):
                break
            time.sleep(2)
            continue
        consecutive = consecutive + 1 if values and max(values) <= 2 and not busy else 0
        if consecutive == 3:
            break
        time.sleep(2)
    else:
        sys.exit("GPU is not idle; no timing was started.")
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
