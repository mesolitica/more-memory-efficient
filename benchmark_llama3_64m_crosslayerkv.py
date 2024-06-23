import subprocess
from tqdm import tqdm

lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

for l in tqdm(lengths):
    subprocess.call(f'python3 benchmark/llama3_64m_crosslayerkv.py {l}', shell=True)