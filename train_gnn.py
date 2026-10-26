"""LRR command-line entry point; see README for arguments."""
from pathlib import Path
import runpy,sys
scripts=Path(__file__).resolve().parent/'scripts'
sys.path.insert(0,str(scripts))
if __name__=='__main__':runpy.run_path(str(scripts/'pretrain.py'),run_name='__main__')
