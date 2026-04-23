import json
import os

for annotator in ['annotator_1', 'annotator_2']:
    d = f'oracle/outputs/tstar/{annotator}'
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        if not f.endswith('.json'):
            continue
        path = os.path.join(d, f)
        data = json.load(open(path))
        task_id = str(data['task_id'])
        t_star = data['t_star']
        oracle_state = data['oracle_state']
        if isinstance(oracle_state, str):
            parsed = json.loads(oracle_state)
        else:
            parsed = oracle_state
        fixed = {
            'trajectory_id': task_id,
            'step': t_star,
            'prompt_version': 'v3',
            'input_tokens': 0,
            'output_tokens': 0,
            'cost_usd': data.get('cost_usd', 0),
            'raw_response': data.get('oracle_state', ''),
            'parsed': parsed,
        }
        json.dump(fixed, open(path, 'w'), indent=2)
        print(f'Fixed {path}')
