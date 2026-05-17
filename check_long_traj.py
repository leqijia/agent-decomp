import json, os
for f in os.listdir("trajectories/oracle_external"):
    if f.endswith(".json"):
        d = json.load(open(f"trajectories/oracle_external/{f}"))
        ts = d.get("total_steps", 0)
        if 41 <= ts <= 80:
            print(f"{f}: total_steps={ts}, success={d.get('success')}, stop_reason={d.get('stop_reason')}")
