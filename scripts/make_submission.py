"""Write the submission CSV from scratch, no cached result reused.

    uv run python scripts/make_submission.py [v19c|v19|v18b|v18|v17|v16|v15|v14|v13|v12|v11b|v11|v10|v9|v8|v7|v6|v5|v4|v3|v2|v1] [OUT]    # default v19c -> evaluation/results/v19c_chain.csv

Both: periods (mtah.periods) -> dead ball from a HistGB classifier trained on degraded external data
(v1: analysis/m1_dead_classifier.py features; v2: plus mtah.deadball's, analysis/p4_common.dead_v2).
Team call:
  v2  the learned team model (mtah.teammodel: features -> HistGB trained on degraded DFL + Metrica ->
      two-state Viterbi, switch cost 16), the submission since 26 Oct (notes/BULGULAR.md §10)
  v3  v2 with the broadcast + stillness dead-ball rule (mtah.deadball), Smart Tagging's restart team
      (mtah.restarts) and the possessor-crop evidence (mtah.possessor; needs cache/possessor_prob.npz from
      analysis/p6d_possessor_crops.py infer with models/possessor_r18.pt), calibrated GT-free on ball_v1's
      validation blocks and added to the Viterbi at SCALE
  v4  v3 with the team model a bidirectional GRU (mtah.grumodel, 3 seeds averaged, switch cost 8;
      DFL LOMO full chain 0.676 vs the booster's 0.606, P6s); weights in models/team_gru_<hash>.pt
  v5  v4 with a per-frame switch cost from a boundary GRU (mtah.grumodel.switch_costs; P6x +0.007 on DFL)
  v11b v11 decoded with a flat switch cost ENSEMBLE_SWITCH instead of the boundary GRU's per-frame cost
      (P8i: nested choice 3-4 on DFL, 0.751 -> 0.755, 6 of 7; averaging three members softens the probabilities)
  v17 v16 with the GRU members 256 hidden / 3 layers (P9at full chain: DFL +0.004, Metrica +0.008 at the 3 s
      referee, no loss at 1.6 s), dead ball v8 (Metrica 1-3, X frames out; P9ar) and the strict restart evidence
  v18 v17 with the wide TCN (dilations 1-128) as both TCN members (V18_ARCHES; P9au full chain DFL +0.004 / +0.002 at
      3 s, Metrica +0.002 / +0.001), everything else as v17 (dead ball 3 seeds). Not adopted: P9ay three draws (Metrica
      3 s -0.003), P9bc tail windows (Metrica 1.6 s -0.004)
  v19 v18 with the single-head wide TCN trained on three degradation draws per match (V19_DRAWS; P9bg full chain DFL
      +0.0032 / +0.0002 at 3 s, Metrica +0.0030 / +0.0009), everything else as v18
  v28  v27a with the seven video models' mean (A, B, F, A3, F2, F3, F4) instead of the five's.
  v27  v26 with both dead-ball models (classifier and GRU) trained on the production copies plus copies with a realistic
      dead ball (mtah/deadsim.py; P9dt / P9dv / P9dw / P9dx: the production copies' ball reappears exactly at the restart,
      real tracking and our ATD do not do that). v27a: the classifier only.
  v26  v25 with Mehmet Ali Topkara's model as a further member (his delivery's probabilities for our match,
      cache/team_in/mehmet_ali_topkara/barath/components.npz; LightGBM + TCN-BiGRU trained on DFL + Metrica only): his
      P(home | alive) at log-odds weight 0.25 against our members' mean, his P(dead) as a fourth dead-ball probability
      (P9dj, through the semi-Markov decoder: DFL +0.015 / +0.012 (7/7), Metrica +0.030 / +0.003 (3/3); each model on
      its own degraded copy there, so an upper bound)
  v25  v24 decoded by the explicit-duration semi-Markov decoder (mtah/semimarkov.py, vendored from Mehmet Ali Topkara's
      delivery; configuration fixed on DFL + Metrica, P9dh: nested DFL +0.003 / +0.004, Metrica +0.010 / +0.003, the
      picked setting +0.006 / +0.008 (7/7) and +0.009 / +0.004 (3/3); P9di, SoccerTrack full chain with the video
      channels: 0.7051 -> 0.7235, 8/8) instead of the switch-cost Viterbi inside dead blocks: P(H, A, D) from the
      team probability and the mean of the three dead-ball probabilities, the boundary GRU's probability as its
      boundary bonus, the rule-based dead frames (not-played periods, broadcast + stillness) forced dead
  v24  v23 with the video evidence and the video dead-ball probability averaged over the five video models run on our
      match (A, B, F and the second copies A3, F2; cache/barath_dense_avg5.npz) instead of F alone (P9cy: two copies'
      average +0.056 / +0.072 against +0.050 / +0.066 and +0.053 / +0.077 single, 4 unseen SoccerTrack matches)
  v23  v22 with v21's video dead ball
  v22  v20 with the video evidence box-smoothed over 151 frames (6 s) before the fusion, w 0.5 kept (P9cv: SoccerTrack
      +0.052 / +0.068 against v20's +0.026 / +0.030, 8 / 8; P9cw: +0.071 against +0.022 with the spells reweighted to
      DFL's length distribution; label-free check on our match holds)
  v21  v20 with the video head's dead-ball probability as a third dead-ball model: to_blocks((p + q + v) / 3) in dead
      ball v8 (P9cs: SoccerTrack full chain +0.035 on top of v20's evidence, 8 / 8; P9cu: the check on our match)
  v20  v19c with dense video possession evidence: T-DEED (SoccerNet checkpoint) with a per-frame left / right / dead
      head, trained on SoccerTrack v2 only (P9ci), inference on our tactical video's virtual camera (P9cl, P9cm ->
      cache/barath_dense.npz); 0.5 x (log q_home - log q_away) added to the log-odds (P9cp: SoccerTrack +0.026 /
      +0.030 on 8 matches by cross-fold models; P9co: label-free check on our match)
  v19c v19 without the possessor-crop evidence (scale 0): the crop classifier was trained on self-labels of the BAR-ATH
      video, which organiser e-mails reportedly rule out ("no training on the provided video"); the compliant fallback
  v18b v17 plus the V18_DRAWS members as extra members (dropped as a decision path on 7 Nov; with V18_DRAWS empty it is v17)
  v16 v14 with each of the six team nets averaged over 6 seeds instead of 3 (P9ad: team call DFL +0.006 6 / 7,
      +0.006 7 / 7 at the 3 s referee; Metrica +0.002 2 / 3, +0.012 3 / 3)
  v15 v14 with the six team nets trained on every match twice, once with the harsh degradation (mixed=True; P9e:
      usual judge neutral, harsh judge DFL +0.013 / Metrica +0.029)
  v14 v13 with the team call the mean of six nets: v11's three members and v13's three two-head members (P9b on
      P9a's out-of-fold probabilities)
  v13 v11 with dead ball v7 (p4_common.close_short_gaps after the union) and each team member a two-head net (team +
      boundary target; mtah.grumodel.external_gru multitask),
      the team head averaged, the boundary GRU's switch cost as v11 (P8n: team call DFL +0.010 6 / 7, Metrica +0.005 2 / 3)
  v12 v11 with each team member pre-trained on SoccerTrack v2, then fine-tuned (P8d, P8j)
  v11 v10 with the player-set GRU (mtah.setgru) as a third member of the team ensemble (P8b)
  v10 v9 with the team call the mean of a GRU and a TCN, both trained also on Metrica 1-3 with short
      possessions weighted x2, and the boundary GRU also on Metrica (P7x)
  v9  v8 with dead ball v6: both dead-ball models trained on three degradation draws (p4_common.dead_v6; P7h)
  v8  v7 with dead ball v5: the classifier's probability averaged with a dead-ball GRU's
      (analysis/p4_common.dead_v5; P7c, P7f)
  v7  v6 with both GRUs trained also on every match with the teams swapped (mtah.grumodel.SWAP; P7e)
  v6  v5 with dead ball v4: the classifier trained on a degradation that keeps a carried dead ball
      (analysis/p4_common.dead_v4; P6z)
  v1  the control rule with the ATD ball (R 1.5 m, T 0.4 s)
Then the organisers' validator.
"""

import os
from functools import partial
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import p4_common  # noqa: E402
import possession_metrics as pm  # noqa: E402

OUT = os.path.join(ROOT, "evaluation", "results", "{}_chain.csv")
ENSEMBLE_SWITCH = 4.0


V17_ARCHES = ("big", "tcn", "set")     # P9at: the 256x3 GRU instead of the production GRU, in both heads
V17_TAIL = False                        # P9aq: tail windows, pending
V18_DRAWS = {}                          # (arch, two-head) -> degradation draws; P9ay (3 draws, single-head BigGRU) not adopted
V18_TAIL = set()                        # (arch, two-head) on tail windows; P9bc (single-head BigGRU) not adopted (Metrica 1.6 s -0.004)
V18_ARCHES = ("big", "tcn128", "set")   # P9au: the wide TCN (dilations 1-128) instead of the TCN, in both heads
V19_DRAWS = {("tcn128", False): 3}      # P9bg: the single-head wide TCN on three degradation draws (full chain DFL +0.003, Metrica +0.003)


def main(version="v26", out=None):
    out = out or OUT.format(version)
    st_extra = version in ("v32", "v34", "v36", "v37", "v38")  # v32: v30 + the SoccerTrack-trained tcn128 member as a seventh member (P9fg)
    st_arches = ("tcn128", "big", "set") if version in ("v37", "v38") else ("tcn128",)   # v37: v33 + P9fg's three SoccerTrack members
    pff_set = version in ("v38", "v39")           # v38: v37 + P9fo's PFF-trained set member; v39: v33 + that member only
    pff_extra = version in ("v35", "v36")         # v35: v33 + the PFF-trained tcn128 member (P9fo); v36: v34 + that member
    video_dead_half = version in ("v33", "v34", "v35", "v36", "v37", "v38", "v39")   # v33: v30 with the video's dead-ball probability at weight 1/2 (P9fh) and D's emission weight 1.0 (P9fi); v34: v32 + v33
    if st_extra or video_dead_half or pff_extra or pff_set:
        version = "v30"
    mix = version in ("v27", "v27a", "v28", "v29", "v30", "v31")     # v27a: the dead-ball classifier on mix copies (P9dw); v27: also the GRU (P9dx)
    mix_gru = version == "v27"
    dense = "barath_dense_avg7.npz" if version in ("v28", "v29", "v30", "v31") else "barath_dense_avg5.npz"   # v28: the seven video models' mean
    members_c = version in ("v29", "v30", "v31")              # v29: v28 with the six team members on recipe C (mtah.members_c; P9ef, P9ei, P9ep)
    seed_sets = tuple(tuple(range(a, a + 6)) for a in {"v30": (0, 6), "v31": (0, 6, 12, 18)}[version]) if version in ("v30", "v31") else None   # v30 / v31: 12 / 24 seeds per member (P9eu, P9ex)
    if mix:
        version = "v26"
    tag = version
    if version in ("v20", "v21", "v22", "v23", "v24", "v25", "v26"):                 # v19c's chain with the video evidence added at the two marked places
        version = "v19c"
    for c in (p4_common.CACHE, p4_common.CACHE_V2, p4_common.CACHE_V4, p4_common.CACHE_V5, p4_common.CACHE_V6, p4_common.CACHE_V8):
        if os.path.exists(c):
            os.remove(c)
    video = tag in ("v21", "v23", "v24", "v25", "v26")       # their dead ball and probabilities come from p9cu's build alone
    if not video:                                            # (ours() and dead_v8 were overwritten there; review 13 Nov)
        dead, codes = p4_common.ours()
    if version in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v11b", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c"):
        dead = None if video else {"v2": p4_common.dead_v2, "v6": p4_common.dead_v4, "v7": p4_common.dead_v4, "v8": p4_common.dead_v5, "v9": p4_common.dead_v6,
                "v10": p4_common.dead_v6, "v11": p4_common.dead_v6, "v11b": p4_common.dead_v6, "v12": p4_common.dead_v6, "v13": p4_common.dead_v7, "v14": p4_common.dead_v7, "v15": p4_common.dead_v7, "v16": p4_common.dead_v7, "v17": partial(p4_common.dead_v8, nseeds=3), "v18": partial(p4_common.dead_v8, nseeds=3), "v18b": partial(p4_common.dead_v8, nseeds=3), "v19": partial(p4_common.dead_v8, nseeds=3), "v19c": partial(p4_common.dead_v8, nseeds=3)}.get(version, p4_common.dead_v3)()
        if tag in ("v24", "v25", "v26"):
            os.environ["BARATH_DENSE"] = os.path.join(ROOT, "cache", dense)
        if video:                                                                # the video head's dead-ball probability as a third dead-ball model (P9cs, P9cu)
            from p9cu_barath_video_dead import build as video_dead
            _, dead, dp, dq, _ = video_dead(mix=mix, mix_gru=mix_gru)              # built once: deterministic, reused by the semi-Markov decoder
        from mtah.match import from_tactical
        from mtah.teammodel import SWITCH, decode, external_model, features
        m = from_tactical()
        g = m["half"] > 0
        if version in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v11b", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c"):
            from mtah.grumodel import SEEDS, SWITCH, external_gru, predict, switch_costs
            swap = version in ("v7", "v8", "v9", "v10", "v11", "v11b", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c")
            if version in ("v10", "v11", "v11b", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c"):
                arches = V18_ARCHES if version in ("v18", "v19", "v19c") else V17_ARCHES if version in ("v17", "v18b") else ("gru", "tcn", "set") if version in ("v11", "v11b", "v12", "v13", "v14", "v15", "v16") else ("gru", "tcn")
                pre = "st" if version == "v12" else None
                heads = {"v13": (True,), "v14": (False, True), "v15": (False, True), "v16": (False, True), "v17": (False, True), "v18": (False, True), "v18b": (False, True), "v19": (False, True), "v19c": (False, True)}.get(version, (False,))
                seeds = tuple(range(6)) if version in ("v16", "v17", "v18", "v18b", "v19", "v19c") else SEEDS
                tail = V17_TAIL and version == "v17"
                draws = (lambda a, mt: V18_DRAWS.get((a, mt), 1)) if version == "v18" else (lambda a, mt: V19_DRAWS.get((a, mt), 1)) if version in ("v19", "v19c") else (lambda a, mt: 1)
                tails = (lambda a, mt: (a, mt) in V18_TAIL) if version == "v18" else (lambda a, mt: tail)
                if members_c:
                    from mtah.members_c import cache_path, external_gru_c
                    sets = seed_sets or (seeds,)
                    missing = [cache_path("team", a, mt, ss, draws(a, mt)) for ss in sets for a in arches for mt in heads if not os.path.exists(cache_path("team", a, mt, ss, draws(a, mt)))]
                    assert not missing, "recipe C members are trained on Colab / Kaggle, not here: missing " + ", ".join(map(os.path.basename, missing))
                    member = lambda a, mt: [external_gru_c(arch=a, multitask=mt, seeds=ss, draws=draws(a, mt)) for ss in sets]
                else:
                    member = lambda a, mt: external_gru(swap=True, metrica=True, short_k=2.0, arch=a, pretrain=pre, multitask=mt, mixed=version == "v15", seeds=seeds,
                                                        tail=tails(a, mt), draws=draws(a, mt))
                if st_extra:
                    from mtah.members_c import ST_LABELLED
                    st_member = []
                    for a_ in st_arches:
                        st_path = cache_path("team", a_, False, tuple(range(6)), 1, st=ST_LABELLED)
                        assert os.path.exists(st_path), "the P9fg member is trained on Colab, not here: missing " + os.path.basename(st_path)
                        st_member.append(predict(external_gru_c(arch=a_, multitask=False, seeds=tuple(range(6)), draws=1, st=ST_LABELLED), m))
                else:
                    st_member = []
                if pff_set:
                    from mtah.pff import pff_ids
                    pff_g = tuple(pff_ids()[12:22])
                    pff_path = cache_path("team", "set", False, tuple(range(6)), 1, pff=pff_g)
                    assert os.path.exists(pff_path), "the P9fo set member is trained on Colab, not here: missing " + os.path.basename(pff_path)
                    st_member.append(predict(external_gru_c(arch="set", multitask=False, seeds=tuple(range(6)), draws=1, pff=pff_g), m))
                if pff_extra:
                    from mtah.pff import pff_ids
                    pff_g = tuple(pff_ids()[12:22])
                    pff_path = cache_path("team", "tcn128", False, tuple(range(6)), 1, pff=pff_g)
                    assert os.path.exists(pff_path), "the P9fo member is trained locally, not here: missing " + os.path.basename(pff_path)
                    st_member.append(predict(external_gru_c(arch="tcn128", multitask=False, seeds=tuple(range(6)), draws=1, pff=pff_g), m))
                p = np.mean([np.mean([predict(e, m) for e in member(a, mt)], axis=0) if members_c else predict(member(a, mt), m) for a in arches for mt in heads]
                            + st_member
                            + ([predict(external_gru(swap=True, metrica=True, short_k=2.0, arch=a, multitask=mt, seeds=seeds, draws=d), m)
                                for (a, mt), d in V18_DRAWS.items()] if version == "v18b" else []), axis=0)
            else:
                p = predict(external_gru(swap=swap), m)
            if version == "v11b":
                SWITCH = ENSEMBLE_SWITCH
            elif version in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c"):
                SWITCH = switch_costs(external_gru(target="boundary", swap=swap, metrica=version in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c")), m)
        else:
            p = external_model().predict_proba(features(m))[:, 1]
        if version in ("v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v11b", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v18b", "v19", "v19c"):
            from mtah.data import load_tactical
            from mtah.possessor import SCALE, calibrate, crop_evidence, evidence_logodds
            from p3e_ball_pseudolabels import BLOCK
            z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
            if version != "v19c":                    # v19c and later: no possessor-crop evidence (its model was trained on the BAR-ATH video),
                m["hom"] = load_tactical()["hom"]    # and its files are not even read
                t, q = crop_evidence(m)
                val = np.flatnonzero(g & ((np.arange(len(g)) // BLOCK) % 5 == 4))
                z = z + SCALE * evidence_logodds(t, q, calibrate(m, t, q, val))
            if tag == "v26":                         # Mali's team probability as a member at log-odds weight 0.25 (P9de, P9dj)
                mali = np.load(os.path.join(ROOT, "cache", "team_in", "mehmet_ali_topkara", "barath_rebuilt" if members_c else "barath", "components.npz"))   # v29: rebuilt by us (scripts/mali_components.py)
                mt = np.clip(mali["p_home_given_alive"].astype(np.float64), 1e-4, 1 - 1e-4)
                z = np.where(mali["valid"], 0.75 * z + 0.25 * np.log(mt / (1 - mt)), z)
            if tag in ("v20", "v21", "v22", "v23", "v24", "v25", "v26"):   # dense video possession evidence, w = 0.5 fixed on 10 Nov (P9ci, P9cp, P9co)
                e = np.load(os.path.join(ROOT, "cache", dense if tag in ("v24", "v25", "v26") else "barath_dense.npz"))["e"].astype(np.float64)
                z = z + 0.5 * (np.convolve(e, np.ones(151) / 151, "same") if tag in ("v22", "v23", "v24", "v25", "v26") else e)   # v22: smoothed (P9cv, P9cw)
            sys.path.insert(0, os.path.join(ROOT, "starter"))
            import load_smart_tagging
            from mtah.restarts import restart_evidence
            z = z + restart_evidence(load_smart_tagging.load(), g & ~dead, strict=version in ("v17", "v18", "v18b", "v19", "v19c"))
            p = 1 / (1 + np.exp(-z))
        codes = np.where(g, decode(p, g & ~dead, SWITCH), pm.X).astype(np.int8)
        if tag in ("v25", "v26"):                             # the semi-Markov decoder on the same probabilities (P9dh, P9di)
            from mtah import semimarkov
            from mtah.deadball import broadcast_still_dead
            from mtah.frames import load_frames
            from mtah.grumodel import SWITCH_HI, SWITCH_LO
            from mtah.periods import NOT_PLAYED
            dv = np.where(g, np.load(os.environ["BARATH_DENSE"])["q_dead"].astype(np.float64), 0)
            fr = load_frames()
            rule = (np.isin(fr["period"].values, NOT_PLAYED) & g) | (broadcast_still_dead(fr, os.path.join(ROOT, "cache", "broadcast_on_tactical.npz")) & g)
            b = np.clip((SWITCH_HI - np.broadcast_to(np.asarray(SWITCH, float), p.shape)) / (SWITCH_HI - SWITCH_LO), 0, 1)
            pdead = (dp + dq + dv) / 3
            if tag == "v26":
                pdead = np.where(mali["valid"], (dp + dq + dv + mali["p_dead"].astype(np.float64)) / 4, pdead)
            if video_dead_half:                       # P9fh: the video at 1/2, the tracking-based models share the other half
                pdead = 0.5 * dv + 0.5 * np.where(mali["valid"], (dp + dq + mali["p_dead"].astype(np.float64)) / 3, (dp + dq) / 2)
            cfg = semimarkov.CONFIG
            if video_dead_half:                       # P9fi: D's emission weight 1.0 instead of 1.15 with the P9fh dead ball
                import copy
                cfg = copy.deepcopy(cfg)
                cfg["decoder"]["emission_weights"][2] = 1.0
            codes = np.where(g, semimarkov.decode_halves(p, pdead, g, b, forced_dead=rule, cfg=cfg), pm.X).astype(np.int8)
            if os.environ.get("MTAH_DUMP_HSMM"):     # the semi-Markov decoder's inputs, for GT-free variant studies on our match
                np.savez_compressed(os.environ["MTAH_DUMP_HSMM"], p=p, pdead=pdead, b=b, rule=rule, g=g, dp=dp, dq=dq, dv=dv, codes=codes)
        if os.environ.get("MTAH_DUMP"):              # the chain's inputs, for GT-free decode studies on our match
            np.savez_compressed(os.environ["MTAH_DUMP"], p=p, switch=np.broadcast_to(np.asarray(SWITCH, float), p.shape), dead=dead, g=g)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(codes)), "label": np.array(pm.LABELS)[codes]}).to_csv(out, index=False)
    print("wrote", out)
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), out], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
