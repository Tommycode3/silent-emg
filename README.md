# silent-emg: reading silently mouthed dates and times from face and neck muscles

![hero](figures/hero.png)

@@INTRO@@

<!-- TODO(Tommaso): your own words on why -->

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. data: 3.9 GB archive from Zenodo (CC BY 4.0), checksum md5:7f97d2182b896652999b1b2d0c69fd7b
mkdir -p data/raw && cd data/raw
curl -L -o emg_data.tar.gz "https://zenodo.org/records/4064409/files/emg_data.tar.gz?download=1"
tar xzf emg_data.tar.gz --exclude='*.flac' emg_data/closed_vocab emg_data/silent_parallel_data
cd ../..

# 2. filter once and cache as memory-mapped float32 (about 480 MB)
python scripts/prepare_data.py

# 3. experiments
python scripts/run_classical.py        # Hudgins features + logreg / GBM, a few minutes on CPU
python scripts/run_experiments.py      # CTC models, one run at a time, resumable (hours on an M-series Mac)
python scripts/grammar_decode.py       # re-decode saved CTC outputs with the date/time grammar
python scripts/make_figures.py         # results/summary.md and figures/
python scripts/plot_raw.py             # figures/raw_emg_example.png

# tests (synthetic data only, no download)
python -m pytest -q
```

`run_experiments.py` waits before each run until macOS reports at least 25% free memory, trains one model at a time, and appends each finished run to `results/runs.jsonl`, so it can be stopped and restarted.

## Results

@@RESULTS@@

## Method

### Data and targets

Eight monopolar surface EMG channels at 1000 Hz (OpenBCI Cyton) from one speaker. Electrode sites, from Table 3 of Gaddy & Klein (2020):

| ch | site | group |
|---|---|---|
| 1 | left cheek just above mouth | face |
| 2 | left corner of chin | face |
| 3 | below chin, back 3 cm | neck |
| 4 | throat, 3 cm left of Adam's apple | neck |
| 5 | mid-jaw right | face |
| 6 | right cheek just below mouth | face |
| 7 | right cheek 2 cm from nose | face |
| 8 | back of right cheek, 4 cm in front of ear | face |

The closed-vocabulary set is 500 prompts drawn from three templates: `08:11 PM`, `July 07 2018`, `Monday June 30`. The prompts are written, but the speaker articulated words, so targets are converted to a spoken form (`eight eleven pm`, `july seventh twenty eighteen`, `monday june thirtieth`) by `silent_emg/spoken.py`. Two readings are a guess: years 2000 to 2009 become `two thousand X`, and `:00` becomes `oclock`. My conversion yields 73 distinct words; the paper reports 67, so the speaker's readings differ from mine in a few places.

### Preprocessing

Per channel, zero-phase IIR notches at 60 Hz and each harmonic below Nyquist (quality factor 30), then a zero-phase 4th-order Butterworth band-pass from 2 to 450 Hz. The previous and next recording are concatenated before filtering so that filter transients fall outside the utterance. For the raw-signal model the signal is then decimated to 500 Hz with a zero-phase FIR anti-alias filter, and every channel is z-scored with mean and standard deviation from the training split only.

### Classical features

Utterance-level (slot classification). For a segment $x_1..x_N$ of one channel, with dead-zone $\varepsilon = 0.01\,\sigma_x$, the Hudgins et al. (1993) features are

$$
\mathrm{MAV} = \frac{1}{N}\sum_n |x_n|,\qquad
\mathrm{WL} = \frac{1}{N}\sum_n |x_{n+1}-x_n|,
$$

$$
\mathrm{ZC} = \frac{1}{N}\sum_n \mathbb{1}\left[x_n x_{n+1} < 0 \wedge |x_{n+1}-x_n| \ge \varepsilon\right],\qquad
\mathrm{SSC} = \frac{1}{N}\sum_n \mathbb{1}\left[(x_n - x_{n-1})(x_n - x_{n+1}) > 0\right]\ \text{(same dead-zone)},
$$

plus $\log \mathrm{RMS}$ ($\log$ is also applied to MAV and WL). They are computed on the whole utterance and on 4 equal-length quarters, for all 8 channels, plus $\log$ duration: $5 \times 8 \times 5 + 1 = 201$ features. Classifiers: L2 logistic regression on standardised features ($C = 0.1$), and histogram gradient boosting (200 iterations, learning rate 0.05, 15 leaves).

Frame-level (linear CTC baseline). Following Jou et al. (2006), each channel is split into a low-frequency part $w = \bar{\bar{x}}$ (two passes of a 9-point moving average) and a high-frequency part $p = x - w$, with $r = |p|$. On 27 ms frames with a 10 ms hop:

$$
f = \left[\tfrac{1}{n}\sum w^2,\ \tfrac{1}{n}\sum w,\ \tfrac{1}{n}\sum p^2,\ \mathrm{ZCR}(p),\ \tfrac{1}{n}\sum r,\ \mathrm{WL},\ \mathrm{SSC}\right],
$$

stacked over $\pm 7$ frames (840 dimensions), followed by one linear layer and a softmax per frame.

### Neural model

Raw 500 Hz EMG goes through three residual 1D convolution blocks (kernel 3, GELU, batch norm; strides 2, 2, 1; 32, 64, 128 channels), giving 125 frames per second, then sinusoidal positions and a 3-layer pre-norm Transformer encoder ($d = 128$, 4 heads, feed-forward 384, dropout 0.3), then a linear layer over 37 characters plus the CTC blank (0.6 M parameters). Training augmentation: per-channel gain $\sim \mathcal{N}(1, 0.1)$, additive noise $\sigma = 0.05$, and two zeroed time spans of up to 80 ms.

### Training objective and decoding

Both sequence models are trained with connectionist temporal classification (Graves et al., 2006). With $\mathcal{B}$ the map that removes repeats and then blanks,

$$
\mathcal{L} = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} p_\theta(\pi_t \mid x).
$$

AdamW (learning rate $10^{-3}$ for the network, $3 \cdot 10^{-3}$ for the linear model, weight decay 0.01), linear warm-up then cosine decay, batch 16, gradient clipping at 1. The checkpoint with the lowest validation CER is kept.

Two decoders are reported. Greedy decoding takes $\arg\max_k p(k \mid x_t)$ per frame and applies $\mathcal{B}$. Grammar decoding searches only over sentences the three templates can produce (hours 1 to 12, minutes, am/pm, months, ordinal days 1 to 31, weekdays, and every year in the range seen in training): the grammar is compiled into a graph of character states with the CTC blank topology, and Viterbi search returns

$$
\hat{y} = \arg\max_{y \in G} \max_{\pi \in \mathcal{B}^{-1}(y)} \sum_t \log p_\theta(\pi_t \mid x).
$$

### Metrics

$\mathrm{CER} = (S + D + I) / N$ over characters of the spoken-form text, $\mathrm{WER}$ the same over words. Slot accuracy is balanced accuracy (mean per-class recall); for the CTC models the slot is read off the decoded text.

### Evaluation protocol

- Random split: 370 train / 30 validation / 100 test utterances, the sizes used by Gaddy & Klein (2020) (their indices were not released for the closed set, so mine are a fresh random draw with seed 0). The split is fixed; seeds change initialisation, batch order and augmentation.
- Contiguous split: the last 100 recordings are the test set, the 30 before them validation. All 500 closed-vocabulary utterances come from one session, so this is the closest available proxy for a later recording.
- Held-out session (open vocabulary): train on 5 silent sessions, select on session 5-10, test on session 5-11. Electrodes were re-attached between sessions. A random-utterance split of the same sizes is the comparison.
- Shuffled-label control: training and validation transcripts are permuted across utterances; the test labels stay true. Whatever such a model scores comes from the label distribution alone.
- Electrode ablation: the model is retrained on the subset of channels (they are removed, not zeroed).
- Classical slot classifiers: 5-fold cross-validation repeated 5 times with different stratified folds, plus 5 contiguous folds; also trained on the 400 non-test utterances of the random split and scored on the same 100 test utterances as the CTC models.

## Data and licenses

- Dataset: Gaddy, D. (2020). Silent Speech EMG. Zenodo. [10.5281/zenodo.4064409](https://doi.org/10.5281/zenodo.4064409). License: Creative Commons Attribution 4.0 International (checked through the Zenodo and DataCite APIs). One file, `emg_data.tar.gz`, 3.92 GB. This repository uses `closed_vocab/` (500 silent and 500 vocalised utterances, one session) and `silent_parallel_data/` (1588 silent utterances, 7 sessions); audio files are not extracted.
- No data is committed. `data/` is git-ignored.
- The preprocessing follows the published description in Gaddy & Klein (2020) and the MIT-licensed reference code at [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech); no code is copied from it.
- Code in this repository: MIT.

## Limitations

@@LIMITATIONS@@

## References

- Gaddy, D., & Klein, D. (2020). Digital Voicing of Silent Speech. Proceedings of EMNLP 2020, 5521-5530. [10.18653/v1/2020.emnlp-main.445](https://doi.org/10.18653/v1/2020.emnlp-main.445)
- Gaddy, D. (2020). Silent Speech EMG [dataset]. Zenodo. [10.5281/zenodo.4064409](https://doi.org/10.5281/zenodo.4064409)
- Gaddy, D., & Klein, D. (2021). An Improved Model for Voicing Silent Speech. Proceedings of ACL-IJCNLP 2021 (short papers). [10.18653/v1/2021.acl-short.23](https://doi.org/10.18653/v1/2021.acl-short.23)
- Jou, S.-C., Schultz, T., Walliczek, M., Kraft, F., & Waibel, A. (2006). Towards continuous speech recognition using surface electromyography. Interspeech 2006. [10.21437/Interspeech.2006-212](https://doi.org/10.21437/Interspeech.2006-212)
- Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy for multifunction myoelectric control. IEEE Transactions on Biomedical Engineering, 40(1), 82-94. [10.1109/10.204774](https://doi.org/10.1109/10.204774)
- Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification. Proceedings of ICML 2006. [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891)
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017. [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
