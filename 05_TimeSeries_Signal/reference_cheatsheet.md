# Domain 5 — Sleep Stage Classification (Time-Series / Biosignal / EEG)

> 🔄 SOTA ล่าสุด (กลางปี 2026) + library เจ๋ง ๆ ดู `../ADVANCED_SOTA.md` และ `../LIBRARIES.md` — ไฟล์นี้คือความรู้พื้นฐาน/กับดักที่ยังใช้ได้

Competition: `super-ai-engineer-ss-6-sleep-stage-classification`

Source note: Kaggle page gated (redirects to login). Reconstructed from the Super AI franchise pattern (SS5 ran "IoT Sleep Stage Classification") + Sleep-EDF / AASM literature. Two data formats covered (raw EEG, wearable IoT) so you're robust either way. Confirm metric/label-set/FS/#channels on the live page first.

## A. TASK SUMMARY
- Task: multiclass — assign one sleep stage to each `30-second epoch`. Per-epoch classification, not forecasting.
- Classes — AASM 5-class (memorize): `W` Wake, `N1` light (rare ~5-8%), `N2` dominant (~40-50%), `N3` deep, `REM`. Variants: +`Unknown/Movement` (6th), or collapsed 3-class `W/N/R` (SS5 IoT). Check label set in `train.csv` first.
- Signal formats:
  - Format-1 (raw EEG, Sleep-EDF): single-channel EEG (`Fpz-Cz`/`Pz-Oz`), `100 Hz`, 30s = `3000` samples/epoch. `.edf` (read with `mne`) or pre-cut `.npz`/`.parquet`.
  - Format-2 (wearable IoT): `BVP, HR, IBI, EDA, TEMP, ACC_X/Y/Z` at ~16-64 Hz over 30s windows. CSV of features or raw blocks.
- Metric: likely `Accuracy`, `Macro-F1`, or `Cohen's kappa`. N1 rare + N2 dominant → accuracy misleading; optimize macro-F1 / kappa regardless.
- Submission: `id,label` CSV. Labels integer (`0=W..4=REM`) or string — match `sample_submission.csv`.

## B. DOMAIN CATEGORIZATION
- Primary: time-series / sequence classification (fixed window → discrete label).
- Sub-domain: biosignal / EEG / polysomnography (PSG); broader = BCI + wearable health.
- Identical structure to: HAR (accelerometer), ECG beat typing, EMG gesture — same toolbox (spectral features + GBDT, or 1D-CNN / CNN-RNN / transformer).
- Key property: epochs are temporally correlated (sleep follows transition rules) — context-aware models beat per-epoch models.

## C. PREDICTED EXAM VARIATIONS
1. ECG arrhythmia (MIT-BIH). Window = one heartbeat (around R-peak), add QRS detection + RR features. Imbalanced → macro-F1. Same 1D-CNN.
2. EEG seizure detection (CHB-MIT/TUH). Multi-channel, higher-band power, split by patient, focal loss.
3. HAR / accelerometer. Lower FS (20-50Hz), time-domain stats dominate, window ~2.5s. Split by subject.
4. Anomaly / apnea detection. Autoencoder reconstruction error / imbalanced binary. Metric AUPRC.
5. Time-series forecasting (regression!). Lag features + LightGBM or N-BEATS/PatchTST. Metric RMSE/MAE/SMAPE. Detect from the metric.
6. Audio / vibration event. Feature = log-mel spectrogram / MFCC (`librosa`) → 2D-CNN.

General rule: identify (window length, FS, #channels, #classes, metric) in first 10 min, then drop into the reusable pipeline.

## E. SOTA TABLE (Sleep-EDF, AASM 5-class, single-channel EEG)
| Model | Architecture | ACC / MF1 / kappa | When to use |
|---|---|---|---|
| SleePyCo (2022) | feature-pyramid CNN + supervised contrastive | 86.8 / 81.2 / 0.82 | Current top single-channel |
| XSleepNet (2020) | multi-view raw + spectrogram, seq | 86.4 / 80.9 / 0.81 | Both raw + time-freq |
| 1D-ResNet-SE-LSTM (2023) | ResNet+SE → BiLSTM | 86.4 / 82.0 / 0.81 | Great N1 recall, easy to reimplement |
| AttnSleep (2021) | multi-res CNN + self-attention | 84.4 / 78.1 / 0.79 | Good attention baseline |
| U-Sleep / U-Time | fully-conv U-Net, whole-night | ~80-87 MF1 | Best cross-dataset generalization |
| SeqSleepNet (2019) | hierarchical RNN | 85.2 / 79.6 / 0.79 | Classic seq2seq |
| DeepSleepNet (2017) | 2-branch CNN → BiLSTM | 82.0 / 76.9 / 0.76 | Canonical reference (= our CNN) |
| TinySleepNet (2020) | light CNN → 1 LSTM | 85.4 / 80.5 / 0.80 | Best perf-per-param ← recommended deep |
| spectral features + LightGBM | hand-crafted band-power/Hjorth | ~78-82 / ~73-76 | Fastest baseline ← <1h submission |
| LaBraM / BIOT (2024) | EEG foundation transformer | SOTA-competitive | Only if pretrained weights + GPU |

Hard truth: N1 F1 caps ~0.45-0.59 for everyone — the macro-F1 bottleneck. CNN → CNN+sequence (LSTM/attention) = +3-5 macro-F1, the single best upgrade.

## F. PRACTICAL GOTCHAS
1. Class imbalance (N1 rare). Optimize macro-F1/kappa, never raw accuracy. Use inverse-frequency class weights / oversample / focal loss.
2. Subject-wise splitting (the #1 mistake). Epochs from same recording are correlated; random split inflates CV 5-15 pts. Always `StratifiedGroupKFold` by `subject`. No subject col → split by contiguous time blocks.
3. Epoch context. Feed ±1-2 epochs or add LSTM/attention over per-epoch embeddings. Cheap, high-impact.
4. Per-recording normalization. z-score per recording/epoch; don't fit a global scaler then apply (leaks scale).
5. Filtering/resampling. Band-pass 0.3-35 Hz, resample to common FS (100 Hz) before windowing.
6. Wake-trimming (Sleep-EDF). Trim long W stretches before/after sleep or W dominates.
7. Label alignment. Verify signal/hypnogram line up (off-by-one destroys accuracy).
8. Metric mismatch. Early-stop / model-select on the actual competition metric (macro-F1), not log-loss/accuracy.

## G. QUICK-WIN BASELINE (<1 hour)
Feature LightGBM (band-power + Hjorth + stats per epoch) with class weights + subject-wise CV. CPU-only, minutes, leak-free. Expect ~0.73-0.78 macro-F1. Upgrade to 1D-CNN, then ensemble (average probs). Feature+LGBM first because zero GPU, fast, almost impossible to leak with GroupKFold.

## H. KEYWORDS TO STUDY
```
sleep stage classification, AASM 5-class W N1 N2 N3 REM, polysomnography PSG, hypnogram,
Sleep-EDF Sleep-EDFX PhysioNet, ISRUC MASS, single-channel EEG Fpz-Cz Pz-Oz, 30-second epoch,
100 Hz resampling, band-pass filter 0.3-35 Hz, Welch PSD power spectral density,
delta theta alpha sigma beta gamma bands, relative band power, Hjorth parameters mobility complexity,
spectral entropy, sleep spindles K-complex, EEG EOG EMG, MNE-Python read_raw_edf,
DeepSleepNet TinySleepNet U-Time U-Sleep SeqSleepNet XSleepNet AttnSleep SleepTransformer,
SleePyCo contrastive, 1D-ResNet-SE-LSTM, LaBraM BIOT EEG foundation model,
1D-CNN time-series, CNN-BiLSTM, epoch context window, sequence-to-sequence sleep scoring,
class imbalance N1, weighted cross-entropy, focal loss, oversampling, macro-F1, Cohen kappa,
StratifiedGroupKFold, subject-wise split, data leakage biosignal, per-recording normalization,
LightGBM multiclass, Welch periodogram scipy.signal, wearable Empatica E4 BVP HR EDA ACC IBI,
AutoGluon, ECG arrhythmia MIT-BIH, HAR accelerometer, seizure detection CHB-MIT,
log-mel spectrogram MFCC, CWT STFT time-frequency, transfer learning EEG
```
