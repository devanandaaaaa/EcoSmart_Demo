import time, os

LOGFILE = "action_logs.csv"

def log_action(device_id, action, pred, threshold):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    header = "timestamp,device_id,action,prediction,threshold\n"
    line = f"{ts},{device_id},{action},{pred:.1f},{threshold:.1f}\n"
    
    if not os.path.exists(LOGFILE):
        with open(LOGFILE, "w") as f:
            f.write(header)
    with open(LOGFILE, "a") as f:
        f.write(line)
    
    print(f"[LOG] {line.strip()}")

def control_plug(device_id, action, pred, threshold):
    print(f"[SIMULATION] {device_id} -> {action} (pred={pred:.1f}, thr={threshold:.1f})")
    log_action(device_id, action, pred, threshold)
    return True
