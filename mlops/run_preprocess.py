from mlops.preprocess import run_all

if __name__ == "__main__":
    clean, ready = run_all()
    print(f"[OK] interim clean: {clean}")
    print(f"[OK] preprocessed:  {ready}")
