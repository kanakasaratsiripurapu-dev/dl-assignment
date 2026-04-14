#!/bin/bash
set -e

echo "installing deps..."
pip install -q -r requirements.txt

echo ""
echo "=== PART 1: TRAINING ==="
python part1/baseline.py
python part1/improved.py
python part1/compare.py

echo ""
echo "=== PART 2: EXPORT + BENCH ==="
python part2/export.py
python part2/bench.py

echo ""
echo "done. check results/ for output files."
echo ""
echo "next steps:"
echo "  1. start api:  cd part2/server && uvicorn app:app --port 8000"
echo "  2. start ui:   cd part2/ui && npm i && npm run dev"
echo "  3. annotate:   python tools/annotate.py --vid YOUR_VIDEO.mp4"
echo "  4. eval:       python tools/eval_custom.py"
echo "  5. fill in the tables in README.md with actual numbers"
