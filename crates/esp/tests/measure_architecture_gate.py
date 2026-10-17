# Copyright 2026 Mark Alan Boykin
# This Source Code Form is subject to the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Run one native architecture gate and retain compact memory measurements.

Usage: python measure_architecture_gate.py OUTPUT_STEM EXECUTABLE [ARG ...]
Requires psutil. NVIDIA readings cover the whole GPU, including other apps.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    stem = Path(sys.argv[1])
    command = sys.argv[2:]
    log = stem.with_suffix('.log')
    receipt = stem.with_suffix('.json')
    if log.exists() or receipt.exists():
        raise RuntimeError('use a fresh receipt stem; existing evidence is immutable')
    peak_rss = peak_os_working_set = peak_gpu_global_mib = 0
    gpu_name = None
    gpu_total_mib = None
    samples = 0
    start = time.monotonic()
    with log.open('wb') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        owner = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                memory = owner.memory_info()
                peak_rss = max(peak_rss, memory.rss)
                peak_os_working_set = max(peak_os_working_set, getattr(memory, 'peak_wset', memory.rss))
            except psutil.NoSuchProcess:
                break
            try:
                values = subprocess.check_output(['nvidia-smi', '--id=0',
                    '--query-gpu=name,memory.total,memory.used',
                    '--format=csv,noheader,nounits'], text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0).strip().split(',')
                gpu_name = values[0].strip()
                gpu_total_mib = int(values[1])
                peak_gpu_global_mib = max(peak_gpu_global_mib, int(values[2]))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            samples += 1
            time.sleep(2)
        code = process.wait()
    receipt.write_text(json.dumps({
        'command':command, 'cwd':str(Path.cwd()), 'exit_code':code,
        'seconds':time.monotonic()-start, 'physical_ram_bytes':psutil.virtual_memory().total,
        'peak_sampled_process_rss_bytes':peak_rss,
        'peak_os_process_working_set_bytes':peak_os_working_set,
        'gpu':gpu_name, 'gpu_total_mib':gpu_total_mib,
        'peak_sampled_whole_gpu_used_mib':peak_gpu_global_mib,
        'gpu_scope':'whole GPU including unrelated applications; not per-process allocation',
        'samples':samples, 'sample_interval_seconds':2,
        'executable_sha256':hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
        'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
    },indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':str(receipt), 'exit_code':code}),flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
