"""Regenerate improved RDRE benchmark figures from existing rdre_outputs CSV files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "rdre_outputs"
raw = pd.read_csv(OUT / "dual_sensor_input.csv")
fr = pd.read_csv(OUT / "frame_features_and_states.csv")
summary = json.loads((OUT / "run_summary.json").read_text())

def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

tms = raw.time_s.to_numpy() * 1000
fig, axes = plt.subplots(3, 1, figsize=(11, 8))
axes[0].plot(tms, raw.sensor1_arb, lw=.45, label="Sensor 1")
axes[0].plot(tms, raw.sensor2_arb, lw=.45, label="Sensor 2", alpha=.8)
for x in [40, 80, 120, 160]: axes[0].axvline(x, ls="--", lw=.8)
axes[0].set(xlim=(0,200), ylabel="Amplitude (a.u.)", title="Overview")
axes[0].legend(ncol=2)
axes[1].plot(tms, raw.sensor1_arb, lw=.65)
axes[1].set(xlim=(76,84), ylabel="Sensor 1 (a.u.)", title="Transition detail (76–84 ms)")
axes[2].plot(tms, raw.sensor2_arb, lw=.65)
axes[2].set(xlim=(76,84), ylabel="Sensor 2 (a.u.)", xlabel="Time (ms)", title="Transition detail (76–84 ms)")
for ax in axes:
    for x in [40,80,120,160]: ax.axvline(x, ls="--", lw=.8, color="black")
    ax.grid(True, alpha=.25)
fig.suptitle("FIG. 2 — Synthetic dual-sensor input; dashed lines are programmed transition times", y=1.01)
save(fig, "fig02_dual_sensor_input")

t = fr.time_s.to_numpy() * 1000
fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True)
axes[0].plot(t,fr.phase_rad,lw=.7); axes[0].set_ylabel("Phase (rad)"); axes[0].set_title("Smoothed inter-sensor phase")
axes[1].plot(t,fr.continuous_wave_count_proxy,lw=.7); axes[1].set_ylabel("Wave-count proxy"); axes[1].set_title("Derived proxy, not an independent wave-count measurement")
axes[2].plot(t,fr.frame_power,lw=.7); axes[2].set_ylabel("Frame power (a.u.)"); axes[2].set_xlabel("Time (ms)")
for ax in axes:
    for x in [40,80,120,160]: ax.axvline(x,ls="--",lw=.8,color="black")
    ax.grid(True,alpha=.25)
fig.suptitle("FIG. 3 — Phase-derived features and signal power",y=1.01)
save(fig,"fig03_time_frequency_phase")

gt=fr.ground_truth_signed_label.to_numpy()
pred=fr.predicted_signed_label.to_numpy()
fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={"height_ratios":[2,1]})
axes[0].step(t,gt,where="post",lw=1.3,label="Ground truth")
axes[0].step(t,pred,where="post",lw=.8,label="Predicted signed label",alpha=.85)
axes[0].set_ylabel("Signed wave label"); axes[0].legend(ncol=2); axes[0].grid(True,alpha=.25)
mis=(gt!=pred)
axes[1].fill_between(t,0,mis.astype(int),step="post",alpha=.7)
axes[1].set_yticks([0,1],["Match","Mismatch"])
axes[1].set_ylabel("Frame status"); axes[1].set_xlabel("Time (ms)"); axes[1].grid(True,alpha=.25)
for ax in axes:
    for x in [40,80,120,160]: ax.axvline(x,ls="--",lw=.8,color="black")
fig.suptitle("FIG. 4 — Ground truth versus predicted signed labels; mismatch panel is frame-level",y=1.01)
save(fig,"fig04_state_comparison")

names={0:"QUENCH",1:"DEFLAGRATION",2:"AXIAL_PULSE",3:"W1+",4:"W1-",5:"W2+",6:"W2-",7:"W3+",8:"W3-",9:"REV"}
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
axes[0].step(t,fr.predicted_state_index,where="post",lw=.8)
axes[0].set_yticks(list(names),[f"{k}: {v}" for k,v in names.items()],fontsize=8)
axes[0].set_ylabel("Predicted HMM state"); axes[0].set_title("Decoded state index")
axes[1].step(t,gt,where="post",lw=1,label="Ground-truth signed label")
axes[1].set_ylabel("Reference label"); axes[1].set_xlabel("Time (ms)")
for ax in axes:
    for x in [40,80,120,160]: ax.axvline(x,ls="--",lw=.8,color="black")
    ax.grid(True,alpha=.25)
fig.suptitle("FIG. 5 — Decoded ten-state sequence and reference signed labels",y=1.01)
save(fig,"fig05_full_state_timeline")

metrics=summary.get("metrics",{})
labels=["True transitions","Matched","Missed","False alarms"]
vals=[metrics.get("True Transition Events",np.nan),metrics.get("Matched Transitions",np.nan),metrics.get("Missed Transitions",np.nan),metrics.get("False Alarm Events",np.nan)]
fig,ax=plt.subplots(figsize=(9,4.8))
bars=ax.bar(labels,vals); ax.bar_label(bars,fmt="%.0f")
ax.set_ylabel("Count"); ax.set_title("FIG. 6 — Event counts from current evaluator (preliminary)")
ax.text(.02,.95,"Not independently validated; evaluator matching logic requires correction.",transform=ax.transAxes,va="top",fontsize=9)
ax.grid(True,axis="y",alpha=.25)
save(fig,"fig06_event_counts")

agreement=float((gt==pred).mean())
pd.DataFrame([{"frame_level_signed_label_agreement_fraction":agreement,"frame_count":len(fr),"mismatched_frames":int((gt!=pred).sum()),"note":"Descriptive agreement on this synthetic run only; not a corrected event metric."}]).to_csv(OUT/"descriptive_agreement.csv",index=False)
print(f"Regenerated figures in {OUT}")
print(f"Frame-level signed-label agreement: {agreement:.4f}; mismatched frames: {int((gt!=pred).sum())}")
