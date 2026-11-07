#!/bin/bash
set -euo pipefail

# Current directory: House_Edge_Lab_Starter_Pack
SOURCE="$(cd ../.. && pwd)"
DEST="$(pwd)/research_environment"

echo "Source: $SOURCE"
echo "Destination: $DEST"

mkdir -p "$DEST"

# 1. Training and validation data ONLY
mkdir -p "$DEST/public"
cp -R "$SOURCE/public/train" "$DEST/public/"
cp -R "$SOURCE/public/validation" "$DEST/public/"

# 2. Starter Python package
cp -R "$SOURCE/slot_forecasting" "$DEST/"

# 3. Experiment, prediction, and submission utilities
for file in \
    run_starter.py \
    run_experiments.py \
    check_submission.py \
    build_entry.py
do
    cp "$SOURCE/$file" "$DEST/"
done

# 4. Competition documentation and metadata
for file in \
    README.md \
    DATA.md \
    RULES.md \
    EXPERIMENTS.md \
    SUBMISSION_README.md \
    SHA256.json
do
    cp "$SOURCE/$file" "$DEST/"
done

cp "$SOURCE/public/metadata.json" "$DEST/public/"

# 5. Optional starter notebook
cp "$SOURCE/starter.ipynb" "$DEST/"

echo ""
echo "Research environment created successfully."
echo ""
find "$DEST" -maxdepth 3 -type f | sort

echo ""
echo "IMPORTANT: public/test was intentionally excluded."