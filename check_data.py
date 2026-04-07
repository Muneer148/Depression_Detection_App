import glob
import os

data_root = "data/depression-dataset"

print("Current working directory:", os.getcwd())
print("Looking for wav files in:", os.path.abspath(data_root))

for folder in ["depression1", "depression2", "normal1", "normal2"]:
    path = os.path.join(data_root, folder, "*.wav")
    files = glob.glob(path)
    print(f"{folder}: {len(files)} files found")
    if files:
        print("   Example:", files[0])