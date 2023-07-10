
#################
# Computing and plotting IRR from paper
#################
# Analysis
echo "Running 0-shot analysis"
python3 scripts/run_irr.py  --expt=bin_zeroshot > results/0shot_IRR.txt

echo "Running k-shot analysis"
python3 scripts/run_irr.py  --expt=bin_kshot > results/kshot_IRR.txt

echo "Running k-shot with reasoning analysis"
python3 scripts/run_irr.py  --expt=bin_kshot_reasoning > results/kshot_reasoning_IRR.txt