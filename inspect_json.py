import json

data = json.load(open('instruction-data.json', 'r', encoding='utf-8'))
print(f"Total entries: {len(data)}")
print()

for i, e in enumerate(data[:5]):
    print(f"--- Entry {i+1} ---")
    print(f"  instruction: {e['instruction']}")
    print(f"  input: {e['input'][:120]}")
    print(f"  output: {e['output'][:200]}")
    print()
