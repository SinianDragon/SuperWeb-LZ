import pathlib
import re

files = list(pathlib.Path('compute_node/performance_metrics').rglob('*.py'))
files += list(pathlib.Path('compute_node/compute_methods').rglob('*.py'))
files += [pathlib.Path('compute_node/task_executor.py')]
files += [pathlib.Path('tests').rglob('*.py')]

count = 0
for f in files:
    try:
        content = f.read_text('utf-8')
        new_content = re.sub(
            r'text=True(,|\s)', 
            r'text=True, encoding="utf-8", errors="replace"\1', 
            content
        )
        if new_content != content:
            f.write_text(new_content, 'utf-8')
            count += 1
    except Exception as e:
        pass

print(f"Patched {count} files.")
