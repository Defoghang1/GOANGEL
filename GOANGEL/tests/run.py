import sys, traceback, importlib
sys.path.insert(0, '.')
mod = importlib.import_module(sys.argv[1])
fails = 0
for n in sorted(dir(mod)):
    if n.startswith('test_'):
        try:
            getattr(mod, n)(); print('PASS', n)
        except Exception:
            fails += 1; print('FAIL', n); traceback.print_exc()
sys.exit(1 if fails else 0)
