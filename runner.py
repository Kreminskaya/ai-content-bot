#!/usr/bin/env py -3
# -*- coding: utf-8 -*-
import subprocess
import sys
import os
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

log_file = f"deploy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

print(f"Starting deploy, logging to {log_file}")

with open(log_file, 'w', encoding='utf-8') as f:
    process = subprocess.Popen(
        [sys.executable, 'deploy_to_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    for line in process.stdout:
        f.write(line)
        f.flush()
        print(line, end='')
    
    process.wait()
    exit_code = process.returncode

print(f"\nDeploy finished with exit code {exit_code}")
sys.exit(exit_code)
