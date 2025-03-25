import sys
import os
sys.path.append(os.path.abspath('..'))  # add root project dir
print("Python sys.path:")
for p in sys.path:
    print("  →", p)

print("Working dir:", os.getcwd())