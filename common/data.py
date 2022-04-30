import os
import re

def find_shelves(path, suffix="", prepend_path=False):
    files = set([os.path.splitext(f)[0] for f in os.listdir(path) if re.match(r'.*\.(?:db|dat)$', f)])
    if prepend_path:
        return sorted([os.path.join(path, os.path.splitext(f)[0]) for f in files])
    return sorted(list(files))