import argparse
from src.evaluation.evaluate import evaluate
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); evaluate(a.config,True)
