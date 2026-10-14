"""Обёртка: scripts/causal.py эксперимента loss_terms с вариантами bypass_only (common.py рядом).
Аргументы — те же, что у оригинала."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # import common -> common.py этого эксперимента
runpy.run_path(os.path.join(HERE, "..", "..", "loss_terms", "scripts", "causal.py"), run_name="__main__")
