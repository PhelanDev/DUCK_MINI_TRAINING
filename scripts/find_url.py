import json

path = r'C:\Users\LEGION\.gemini\antigravity\brain\eac089d5-7756-4b70-896e-d09247ae73c7\.system_generated\logs\transcript.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        c = str(data.get('content', ''))
        tc = str(data.get('tool_calls', ''))
        if 'pollen' in c.lower() or 'pollen' in tc.lower() or 'huggingface.co' in c.lower() or 'huggingface.co' in tc.lower():
            idx = data.get('step_index')
            for line_c in (c + '\n' + tc).splitlines():
                if any(k in line_c.lower() for k in ['http', 'github', 'huggingface', 'onshape']):
                    print(f'step {idx}: {line_c.strip()[:180]}')
