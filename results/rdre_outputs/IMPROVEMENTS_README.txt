RDRE figure review update
=========================

This package regenerates FIG. 2–FIG. 6 from the CSV outputs you supplied.

Improvements:
- FIG. 2: full-duration overview plus transition zooms, with separate sensor views.
- FIG. 3: clearer phase, derived wave-count proxy, and frame-power panels.
- FIG. 4: ground truth versus predicted signed labels plus a frame-level mismatch panel.
- FIG. 5: all ten predicted state labels are named and displayed beside reference signed labels.
- FIG. 6: existing event counts are explicitly marked PRELIMINARY.
- descriptive_agreement.csv: simple frame-level signed-label agreement for this one synthetic run.
- regenerate_improved_figures.py: script to regenerate FIG. 2–FIG. 6 from the two CSV files.

Important limitation:
The uploaded archive contained generated outputs only, not the Python source modules (especially event_evaluator.py).
Therefore this package improves figure readability and adds descriptive comparison, but does NOT correct the underlying event evaluator.
Do not report the event counts or the previous event-latency numbers as validated performance until the source evaluator is corrected and retested.
All plotted signals are synthetic benchmark data, not physical combustion/injector measurements.
