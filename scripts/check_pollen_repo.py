import urllib.request
import json

req = urllib.request.Request('https://api.github.com/repos/pollen-robotics/microduck/contents', headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as resp:
        items = json.loads(resp.read().decode())
        for it in items:
            print(f"{it['type']}: {it['path']}")
except Exception as e:
    print('Error:', e)
