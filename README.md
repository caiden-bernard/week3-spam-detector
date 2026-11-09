# Week 3: Spam Detector

A dependency-free SMS spam classification exercise using a small, illustrative
dataset and a multinomial Naive Bayes model.

## Run the project

From this directory, prepare the reproducible train/test split and evaluate the
classifier:

```powershell
python prepare_data.py
python spam_detector.py
```

To display the exact messages and labels in each split, run:

```powershell
python spam_detector.py --show-data
```

The generated files are written to `data/processed/`. The source messages are
in `data/messages.csv`; the default split uses 75% of each label for training
and a fixed random seed of 42. You can override the input and split settings,
for example:

```powershell
python prepare_data.py --test-size 0.2 --seed 7
```

The analysis and limitations are documented in [analysis_report.md](analysis_report.md).