import json, os, sys

def load(path):
    if not os.path.isfile(path):
        print(f"missing: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

b = load("results/base.json")
imp = load("results/improved.json")

print("\n" + "="*65)
print("  COMPARISON: baseline vs improved")
print("="*65)

header = f"{'metric':<15} {'baseline':>12} {'improved':>12} {'delta':>12}"
print(header)
print("-"*65)

for key in ["map50", "map50_95", "prec", "rec"]:
    bv = b[key]
    iv = imp[key]
    d = iv - bv
    sign = "+" if d >= 0 else ""
    print(f"{key:<15} {bv:>12.4f} {iv:>12.4f} {sign}{d:>11.4f}")

print(f"\n{'train time(s)':<15} {b['time_sec']:>12.1f} {imp['time_sec']:>12.1f}")
print("="*65)

# dump the comparison too
combo = {"baseline": b, "improved": imp}
with open("results/comparison.json", "w") as f:
    json.dump(combo, f, indent=2)
print("\nsaved to results/comparison.json")
