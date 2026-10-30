### Sequence decoding (CTC), test set

| setting | model | seeds | CER % greedy | WER % greedy | CER % grammar | WER % grammar | exact % grammar |
|---|---|---|---|---|---|---|---|
| closed vocab, random split | conv + transformer, shuffled labels (control) | 3 | 83.4 ± 7.5 | 103.5 ± 3.1 | 74.9 ± 1.7 | 96.2 ± 4.6 | 0.3 ± 0.6 |
| closed vocab, random split | linear on TD features (Jou et al.) | 3 | 78.0 ± 2.6 | 110.0 ± 7.2 | 68.1 ± 2.7 | 92.2 ± 3.2 | 0.3 ± 0.6 |
| closed vocab, random split | conv + transformer | 5 | 60.6 ± 3.0 | 99.5 ± 0.8 | 46.2 ± 4.4 | 70.0 ± 5.8 | 1.2 ± 1.1 |
| closed vocab, last 100 utterances held out | conv + transformer | 3 | 62.9 ± 1.6 | 100.5 ± 1.3 | 47.1 ± 4.2 | 72.7 ± 6.6 | 0.7 ± 1.2 |
| train vocalised, test silent | conv + transformer | 3 | 81.7 ± 7.7 | 100.3 ± 0.6 | 69.8 ± 2.9 | 90.0 ± 2.2 | 0.3 ± 0.6 |
| closed vocab, without throat electrode | conv + transformer | 3 | 55.1 ± 8.4 | 94.8 ± 4.5 | 44.6 ± 3.7 | 68.7 ± 5.3 | 0.3 ± 0.6 |
| closed vocab, 6 face electrodes | conv + transformer | 3 | 55.3 ± 2.5 | 97.7 ± 1.5 | 47.0 ± 2.8 | 72.5 ± 3.4 | 1.3 ± 1.2 |
| closed vocab, 2 neck electrodes | conv + transformer | 3 | 71.4 ± 5.2 | 100.9 ± 2.0 | 60.2 ± 1.7 | 84.4 ± 2.5 | 0.0 |

### Slot classification, balanced accuracy %

| task | chance | Hudgins + logreg (5x5-fold CV) | Hudgins + GBM (CV) | logreg, contiguous folds | logreg, shuffled labels | duration only | logreg, train vocalised, test silent |
|---|---|---|---|---|---|---|---|
| template | 33.3 | 96.9 ± 0.3 | 94.5 ± 0.2 | 97.2 | 35.4 ± 1.1 | 61.6 ± 0.8 | 75.1 |
| ampm | 50.0 | 85.5 ± 0.6 | 82.3 ± 0.4 | 84.6 | 48.4 ± 2.8 | 50.1 ± 1.5 | 65.4 |
| weekday | 14.3 | 55.6 ± 1.7 | 46.8 ± 1.8 | 52.0 | 14.6 ± 2.3 | 13.1 ± 1.7 | 20.0 |
| month | 8.3 | 28.0 ± 1.7 | 24.2 ± 1.8 | 25.2 | 8.1 ± 0.5 | 11.0 ± 0.7 | 13.0 |

Same 100 test utterances as the CTC models (E1 split):

| task | logreg | GBM (5 seeds) | conv + transformer, greedy | conv + transformer, grammar |
|---|---|---|---|---|
| template | 99.0 | 95.0 | 33.8 ± 0.9 | 91.7 ± 5.1 |
| ampm | 85.6 | 79.3 | 12.2 ± 15.0 | 68.8 ± 6.7 |
| weekday | 43.6 | 51.9 | 6.0 ± 9.2 | 58.9 ± 10.5 |
| month | 19.7 | 24.0 | 0.8 ± 1.9 | 25.7 ± 3.9 |

### Electrode ablation, Hudgins + logreg, balanced accuracy % (5x5-fold CV)

| electrodes | template | ampm | weekday | month |
|---|---|---|---|---|
| all | 96.9 ± 0.3 | 85.5 ± 0.6 | 55.6 ± 1.7 | 28.0 ± 1.7 |
| no_throat | 97.1 ± 0.3 | 84.1 ± 1.4 | 55.4 ± 2.5 | 26.6 ± 0.5 |
| face_only | 96.1 ± 0.3 | 82.3 ± 1.5 | 56.7 ± 1.4 | 24.5 ± 0.2 |
| neck_only | 89.7 ± 0.9 | 77.1 ± 2.0 | 28.5 ± 2.3 | 16.6 ± 2.1 |
| throat_only | 77.9 ± 1.2 | 59.6 ± 1.7 | 23.1 ± 2.7 | 15.7 ± 1.7 |
| drop_0 | 97.0 ± 0.3 | 84.8 ± 0.9 | 52.9 ± 2.4 | 27.4 ± 1.5 |
| drop_1 | 96.2 ± 0.3 | 83.7 ± 1.1 | 52.8 ± 3.2 | 26.5 ± 1.0 |
| drop_2 | 95.4 ± 0.3 | 83.3 ± 1.0 | 54.0 ± 1.9 | 26.4 ± 1.6 |
| drop_3 | 97.1 ± 0.3 | 84.1 ± 1.4 | 55.4 ± 2.5 | 26.6 ± 0.5 |
| drop_4 | 97.0 ± 0.2 | 83.0 ± 1.2 | 54.9 ± 1.3 | 27.2 ± 1.2 |
| drop_5 | 97.2 ± 0.5 | 83.6 ± 1.1 | 48.0 ± 1.7 | 26.3 ± 1.6 |
| drop_6 | 97.0 ± 0.3 | 84.0 ± 0.7 | 56.4 ± 0.4 | 29.1 ± 0.6 |
| drop_7 | 96.8 ± 0.3 | 84.5 ± 2.0 | 57.8 ± 3.5 | 27.6 ± 0.7 |
