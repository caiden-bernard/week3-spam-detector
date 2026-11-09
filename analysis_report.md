# Spam Detector Analysis Report

## Project overview

This project demonstrates a text-classification workflow using a small,
hand-authored collection of example SMS messages. `prepare_data.py` validates
the CSV and creates a reproducible, stratified training/test split.
`spam_detector.py` trains a multinomial Naive Bayes classifier with Laplace
smoothing and evaluates it on the held-out test split. Both scripts use only
the Python standard library.

## Dataset and method

- Source: `data/messages.csv`, with 40 ham and 40 spam examples.
- Split: 75% training and 25% testing within each class, using seed `42`.
- Features: lowercase alphanumeric word tokens, the exact number of detected
  URLs, and each URL's top-level domain (`.com`, `.org`, `.net`, `.edu`, `.gov`,
  or `other`). Messages without URLs receive a `none` domain feature. URL
  detection recognizes links beginning with `http://`, `https://`, or `www.`.
- Model: multinomial Naive Bayes with add-one smoothing.
- Positive class: `spam`.

The source examples are small and deliberately illustrative. They are not a
representative sample of real messages and should not be used to screen actual
communications.

## Why split the data?

The split keeps two different jobs separate. The model learns word and feature
frequencies from the 60 training messages. The 20 test messages are held aside
until evaluation and are not used to fit the model. This estimates how well the
trained model handles examples it has not seen, which is more useful than
scoring it on its training messages: a model can memorize those examples and
appear successful without generalizing to new messages.

The split is stratified so both ham and spam appear in each set, and seed `42`
makes the split repeatable. Keep the test set out of model fitting and feature
tuning; repeatedly adjusting the model based on test results can make the test
set part of the training process in practice.

## Held-out results

Run `python prepare_data.py` followed by `python spam_detector.py` to reproduce
the metrics printed below.

| Metric | Result |
|---|---:|
| Training examples | 60 |
| Test examples | 20 |
| Accuracy | 90.0% |
| Spam precision | 83.3% |
| Spam recall | 100.0% |
| Spam F1 | 90.9% |

### Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual \ Predicted | Spam | Legitimate |
|---|---:|---:|
| Spam | 10 (spam correctly caught) | 0 (spam wrongly let through) |
| Legitimate | 2 (legitimate wrongly marked spam) | 8 (legitimate correctly passed) |

The four counts sum to all 20 test messages. The 2 legitimate messages marked
as spam are false positives; the 0 spam messages let through are false
negatives.

The bundled examples do not contain URLs, so this split does not measure
whether the new link features improve classification. These scores describe
only this seeded split of a tiny, curated dataset. The
phrasing in the training and test examples is highly similar, so this result
does not establish performance on new senders, changing spam campaigns, or
ordinary user messages. The two false positives were normal messages about
going for a walk after work and picking up milk; their words overlap with
phrases in the spam training examples.

## What the model learned

The program prints the ten features with the largest absolute spam-vs-ham
log-likelihood ratio after training. A positive weight points toward spam; a
negative weight points toward legitimate mail. On this training split, the
largest absolute weight is `i` at `-2.761` (toward legitimate), while `now` is
the strongest spam-associated feature at `+2.443`. `claim` also points toward
spam at `+2.076`.

This is a useful check against intuition: one might expect a URL count or
domain type to be important, but the training messages contain no URLs. The
model therefore has no varied URL examples from which to learn that signal.
Instead, it learned associations from repeated words in this small sample.
The weights are associations in the training data, not evidence that a word
causes a message to be spam; more representative labeled data could change
the ranking substantially.

## Limitations and next steps

- Collect a larger, representative, appropriately licensed dataset.
- Use multiple stratified folds or a separate external test set.
- Inspect false positives and false negatives, especially before blocking
  messages automatically.
- Add features for spelling variation, punctuation, and sender behavior only
  after measuring their impact on validation data.
- Treat predictions as suggestions; this demo has no confidence calibration,
  privacy protections, or production safeguards.




## My additions

I added a feature to count hyperlinks and check for domains like .org or .edu, as these are more likely to be legitimate links.

## Training

The model was trained on 60 training messages and tested 20 test messages. It had an accuracy of 90% on the test messages. You split the training and test sets so that the model doesn't 'memorize' the data, but takes what it learned and applied it to other data. This is why study guides differ from actual test questions: so that students actually need to learn and not just memorize answers.

## Confusion Matrix

Of the actual spam, all 10 were predicted as such.

Of the actual legitimate, 2 were predicted spam and 8 were predicted legitimate.

## Precision and Recall

Precision is the measure of spam that was caught divided by all messages marked as spam. 

Recall is spam that was caught divided by all actual spam messages. 

This model's precision was 83.3% and its recall was 100%, meaning that 83.3% of messages marked as spam were actually spam, and 100% of spam messages were caught.

## 2 errors

In this scenario, a job offer being sent to spam is much worse than a spam email making it through the filter. A legitimate email going to spam would be a false negative and a spam email making it through would be a false positive. A way to combat this possibility is to give false negatives increased weight, telling the algorithm that false negatives are more detrimental than false positives.

## Feature weights

This algorithm weighs words higher than other variables. Words such as 'I', 'will', and 'please' carry legitimate weights, and words such as 'claim' and 'now' carry spam weights.

## Break it

A spam email that reads like a normal email was let through. This email read "Hi, thanks for helping me with the homework. Please send me the address when you can. Your free prize is ready, claim now". Both 'please' and 'thanks' were on the list of words weighted as legitimate.

A legitimate email that has wording associated with spam was predicted as spam. That email read "The school fundraiser prize table is open today. Please claim your free gift now". Once again, words like 'claim' are heavily weighted as spam.

These findings make sense. The algorithm seems to value wording over other possible indicators, which I assumed after seeing how it weighted specific words.

