import re
import matplotlib.pyplot as plt
from datetime import datetime

# Parse logs (pipe from stdin or read file)
logs = open('autoscaling_logs.txt').read()

times, rps, replicas = [], [], []
for line in logs.split('\n'):
    if 'Requests per second:' in line and 'Target number of replicas:' in line:
        time_str = re.search(r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line).group(1)
        rps_val = float(re.search(r'Requests per second: (\d+\.?\d*)', line).group(1))
        rep_val = int(re.search(r'Target number of replicas: (\d+)', line).group(1))
        
        times.append(datetime.strptime(time_str, '%m-%d %H:%M:%S'))
        rps.append(rps_val)
        replicas.append(rep_val)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax1.plot(times, rps, 'b-', linewidth=1)
ax1.set_ylabel('Requests/sec', fontsize=12)
ax1.grid(True, alpha=0.3)

ax2.plot(times, replicas, 'r-', linewidth=1.5)
ax2.fill_between(times, replicas, alpha=0.3, color='red')
ax2.set_ylabel('Target Replicas', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoscaler_timeline.png', dpi=150)
print("Saved to autoscaler_timeline.png")
