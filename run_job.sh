#!/bin/bash

#SBATCH -J pacmath_pilot
#SBATCH -o /project/sajja/asami4/pac-math/logs/full.o%j
#SBATCH -e /project/sajja/asami4/pac-math/logs/full.e%j
#SBATCH --mail-user=asami4@cougarnet.uh.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

# ? Activate env
source /project/sajja/asami4/pac-math/venv/bin/activate

# ? Go to project
cd /project/sajja/asami4/pac-math

# ? Set Ollama models path
export OLLAMA_MODELS=/project/sajja/asami4/ollama_cache/models

# ? Start Ollama inside job
nohup /project/sajja/asami4/bin/ollama serve > ollama.log 2>&1 &

# ? wait for server
sleep 5

# ? Run your script
rm -f outputs/full/summary_methods.csv
rm -f outputs/full/mcnemar_primary_comparisons.csv
rm -f outputs/full/summary_by_topic.csv
rm -f outputs/full/audit_segments.csv
rm -f outputs/full/diagnostic_*.csv
rm -f outputs/full/main_table_methods.csv outputs/full/main_table_methods.tex
rm -f outputs/full/appendix_table_methods.csv outputs/full/appendix_table_methods.tex

python -u scripts/summarize_results.py
python -u scripts/diagnose_experiment.py
python -u scripts/audit_experiment.py
python -u scripts/export_main_tables.py