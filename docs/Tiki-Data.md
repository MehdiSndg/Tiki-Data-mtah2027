# Tiki-Data — possession from automatic tracking and video

More Than A Hack 2027, Phase 1. Per-frame team possession (H / A / D / X) for the BAR-ATH match.
Submission file `Tiki-Data.csv`, built by `scripts/make_submission.py` (version **v28**) and checked with the
organisers' `evaluation/validate_submission.py`.

## 1. Summary

We built a pipeline that never sees a label of the provided game. Every model is trained on external, labelled
football data that we first **degrade to look like our match's automatic tracking**: the ball goes missing, is
smoothed, is placed wrongly and stays frozen at the rates we measured on BAR-ATH without ground truth. Every choice
(model, weight, decoder setting, member added or rejected) was made with the organisers' scorer on those external
matches, under a rule written down before the run. The public five minutes were only read as a check, never used
to select anything.

The chain, one stage per line:

1. **Periods.** Kick-offs from the tracking (players in their own halves, ball at the centre spot), half ends from
   players leaving the pitch plus the referee's whistle in the audio. Frames outside the two graded halves are X;
   graded frames with no play (before the kick-off, after the whistle) are D.
2. **Team probability per frame.** Six sequence models on tracking features (players, ball, distances, speeds,
   goalkeepers): a bidirectional GRU, a wide TCN and a player-set GRU, each with a single team head and with a
   two-head (team + boundary) variant, trained on degraded DFL and Metrica matches (6 seeds each). One further member
   from our teammate Mehmet Ali Topkara's pipeline (LightGBM + TCN-BiGRU, also trained only on external data) at
   log-odds weight 0.25.
3. **Video evidence.** A T-DEED video model (SoccerNet's released checkpoint) with our own dense per-frame head
   (left team / right team / dead), fine-tuned on SoccerTrack v2 only, run on a virtual camera that follows the
   tracked ball in the tactical video. The mean of seven such models, smoothed over 6 s, is added to the log-odds
   at weight 0.5.
4. **Restarts.** Smart Tagging's set-piece phases give the team that takes the restart a log-odds push at the first
   frame where the dead ball ends.
5. **Dead ball.** Four probabilities averaged: a gradient-boosting classifier and a GRU on tracking features (both
   trained on external data, the classifier also on copies with a realistic dead ball), the video model's dead output,
   and the teammate's dead-ball output. Two rules force D:
   not-played periods, and broadcast-replay stretches where the tactical ball is still.
6. **Decoding.** An explicit-duration semi-Markov decoder (H, A, D segments with duration priors; a new possession
   needs a segment change) takes the team and dead probabilities, plus a per-frame boundary probability from a
   separate boundary GRU as a bonus for starting a segment there.

## 2. Why this approach

- **The data on our match is the problem, not the model.** Measured on BAR-ATH without ground truth: the ball is
  missing on 40–42% of graded frames (10–12% during play), gaps are long (median 10.7 s, p90 34 s), a still ball is
  usually dropped, the ball is smoothed (equivalent to DFL's 25-frame smoothing), referees are absent, and 8.6% / 5.3%
  of frames have duplicated players. A model trained on clean tracking does not transfer. So the first thing we built
  was the degradation model (`mtah/degrade.py`) and the evaluation on degraded external matches (`mtah/harness.py`).
- **Where the score goes.** With oracle parts plugged in on the external judges, a perfect frame-level team call is
  worth about +0.22 spellF1 and a perfect dead ball about +0.025; half of all possessions are shorter than 5 s, and
  the short ones are the ones we miss. So most of the work went into the team call and into short possessions.
- **Learned sequence models beat rules.** A control rule (nearest player within 1.5 m for 0.4 s) and a gradient-boosted
  per-frame model were our first baselines; a bidirectional GRU beat the booster by +0.07 on the DFL full chain, and
  every later step (TCN, player-set GRU, two-head team + boundary nets, more seeds, more draws) was a measured gain.
- **Video where tracking is blind.** During ball gaps the tracking cannot see who has the ball; the video can.
  The ball-centred video model was the single largest gain on SoccerTrack (+0.05–0.07 with smoothing).
- **A segment decoder for a segment metric.** spellF1 counts possessions, not frames. The semi-Markov decoder
  scores whole segments with duration priors, which removed many spurious one-second flips (+0.018 on SoccerTrack's
  full chain, 8 of 8 matches).

## 3. Assets — used, tried, dropped

| asset | how we used it | status |
|---|---|---|
| tactical ATD tracking | input to every model | used |
| tactical video | input to the video model (inference only, via a ball-centred virtual camera); audio for the half-end whistle | used |
| homography | maps the tracked ball into the video for the virtual camera | used |
| broadcast ATD + video | synchronised to the tactical frames; replays with a still ball → forced dead; as a ball source it hurt the team call | partly used |
| Smart Tagging | restart team for set pieces (input, not a label; it describes game phases, not possession) | used |
| DFL / Sportec open data (7 matches) | training and evaluation, degraded | used |
| Metrica sample data (3 matches) | training and evaluation, degraded; ball moved off the player it is glued to | used |
| SoccerTrack v2 (10 matches) | training of the video head; a full-chain judge with video (8 matches with ball status) | used |
| SoccerNet T-DEED checkpoint | starting weights of the video model | used |
| PFF FC World Cup 2022 (64 matches) | judge (12 held out; real broadcast tracking, ball where seen) and dead-ball realism; training with it was tried several times and not adopted in this version | used for evaluation |
| WASB ball detector (pretrained on soccer, ISSIA-CNR) | run over every graded frame of the tactical video (tiles, its own tracker); agrees with the ATD (median 1.0 m) but finds a ball in only 12% of in-play ATD gaps, which are 3% of play: too little to change spellF1 | tried, not used |
| possessor-crop classifier | trained on self-labels from the BAR-ATH video; removed (v19c) because training on the provided video is not allowed | dropped |
| self-trained ball detector (YOLO on ATD pseudo-labels of BAR-ATH) | same reason | dropped |
| SoccerTrack possession labels for the team models | learnable but a different definition: DFL judge fell 0.745 → 0.730 | not used in this version |
| further ideas tested and rejected under their pre-registered rules | owner-of-ball head, short-possession expert, de-jittered ball, boundary model on the new recipe, more dead-ball and boundary seeds, ball-blind channel, others | dropped |

## 4. Training data

None of it comes from the provided game. No human made any label on BAR-ATH for us, and no model was trained, fitted,
tuned or calibrated on the provided video, its tracking, or its ground truth.

| dataset | source / licence | who labelled | amount | used for |
|---|---|---|---|---|
| DFL Bundesliga open data | Sportec Solutions / IDSSE, CC BY 4.0 | DFL's official per-frame `BallPossession` and `BallStatus` flags (TRACAB tracking) | 7 matches, ~1 M frames at 25 Hz | team, boundary and dead-ball models; judge |
| Metrica Sports sample data | Metrica's public sample repository | Metrica's event data; per-frame labels derived from events by our code (`mtah.external.metrica_event_labels`) | 3 matches | team, boundary and dead-ball models; judge |
| SoccerTrack v2 | Scott et al., CC BY 4.0 | the dataset authors' per-frame ball status (dead / team 1 / team 2 / loose) | 10 matches (8 with status) | video head fine-tuning; three team members; judge (cross-fold) |
| SoccerNet ball action spotting | SoccerNet (T-DEED checkpoint released by the challenge) | SoccerNet annotators | as released | starting weights only |
| PFF FC World Cup 2022 | PFF FC public release | PFF event data; labels derived from events by our code | 64 matches | 12 judge; 10 train one team member |

Training copies: each external match is degraded at our match's measured rates (one or three random draws), and
every match is also used with the teams swapped. Since 13 Nov each match gets one more copy with a realistic dead
ball (`mtah/deadsim.py`: the ball is lost and reappears near the restart spot before play resumes, as in our
tracking), and Metrica's ball is moved off the player it sits on before degradation (`mtah/unglue.py`).

## 5. The possession rules we implemented

The labels we train toward follow the organisers' convention as described in the challenge text and as far as it can
be read from the public window without fitting it:

- **Possession is control, not touch.** A spell shorter than 1.6 s with the opponent on both sides is given to that
  opponent (`mtah.external.merge_short`). DFL's flag changes on every touch (mean spell 6.6–7.8 s against ~14 s in the
  organisers' public labels); this rule closes most of that gap. 1.6 s was chosen on external data. We report every
  judge both at this and at a 3 s merge.
- **Loose balls and challenges.** The ball stays with the team that last had control until the opponent gains
  control. In the Metrica labels a "ball lost" event hands possession over only through the opponent's following
  recovery. A contested ball is therefore never a third state.
- **Dead ball.** D starts when the ball crosses the line or play is stopped (ball out, shot out or goal, foul) and ends
  at the restart. Not-played time inside a graded half (before the kick-off, after the whistle) is D. A D segment
  always separates two possessions, so H D H counts as two possessions, as the scorer counts them.
- **Restarts.** The team taking a set piece is given possession at the restart (Smart Tagging's team, pushed into the
  log-odds rather than forced).
- **X.** Frames outside the two graded halves.

## 6. Results

All numbers are spellF1 at the two referees (spells merged at 1.6 s / 3 s), full chain, on matches the models never
saw in training (DFL and Metrica: each match held out in turn, no video there).

| judge | v25 (semi-Markov decoder) | v26 (+ teammate's member) | **v28 (+ dead-ball classifier on realistic copies, seven video models)** |
|---|---|---|---|
| DFL open data, 7 matches | 0.7403 / 0.7119 | 0.7552 / 0.7238 | v26 + 0.019 on realistic dead-ball copies |
| Metrica sample, 3 matches | 0.7844 / 0.7793 | 0.8144 / 0.7818 | — |
| SoccerTrack v2, 8 matches, with video (cross-fold) | 0.7235 / 0.7216 | (teammate's model not run there) | **0.7299 / 0.7283** |
| public window, report only | 0.729 | 0.800 | **0.800** |

The public window holds 17 possessions; one spurious 1.3 s spell moves it by 0.025, so it cannot rank versions on
its own. Later versions (more members, a video-weighted dead ball) scored higher on SoccerTrack and PFF; the team
submitted v28.

## 7. Limitations

- **Short possessions.** Spells under 2 s are recalled about one time in three on the external judges; most of our
  remaining error is there. A touch-level model (pass and reception detection from video) would be the next step.
- **Ball gaps.** On 40% of graded frames our match has no ball; there the call rests on player configuration and on
  the video. The video model covers only what the tactical camera shows around the last known ball.
- **Domain shift of the video model.** It was fine-tuned on SoccerTrack v2 (university matches, a different camera).
  Its weight (0.5) and smoothing were set there; on our match we could only check it without labels.
- **Labelling convention.** Our 1.6 s control rule is an approximation of the organisers' convention; their exact
  handling of very short controlled touches is not known to us.
- **Dead-ball onsets.** The tracking keeps a moving ball during some dead stretches (up to 20 s); the dead-ball
  models miss part of those.
- **Training is not bit-reproducible across GPUs.** We release the weight files; rebuilding the CSV from them is
  byte-identical on our machine.

## 8. Reproduction

`scripts/reproduce.sh` lists every step from the organisers' bundle to the CSV:

1. caches of the provided match (tracking and frame tables, broadcast → tactical frame map; no model, no label);
2. the video channel: virtual-camera frames of the tactical video and inference with the seven video models;
3. the teammate's component (his code, his nets trained on external data only);
4. the trained weights in `models/` (team, boundary and dead-ball nets; video heads) — training commands are given
   beside each weight file (Kaggle / Colab scripts in `scripts/kaggle/` and `analysis/`);
5. `python scripts/make_submission.py v28 evaluation/results/Tiki-Data.csv` (about 7.4 GB RAM);
6. `python evaluation/validate_submission.py evaluation/results/Tiki-Data.csv` and the sha256 we record there.

Environment: Python 3.11 with PyTorch (CUDA 12.8), scikit-learn, numpy, pandas; the video stage needs the
`sn-teamspotting` code (T-DEED) and a GPU. Tests: `pytest tests/`.
