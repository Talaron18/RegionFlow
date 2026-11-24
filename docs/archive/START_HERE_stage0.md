# START HERE

Reading order for resuming this project cold:

1. **RESEARCH_REPORT.md** — the complete document. Goal, design decisions and
   why, model structure, every experiment, findings, withdrawn claims, next
   steps. If you read one file, read this.
2. **STAGE0_FINDINGS.md** — the terse findings-of-record. Same conclusions,
   evidence-first, no design narrative.
3. **SETUP.md** — environment. Note that `bench/run.py` is referenced there but
   was never written.
4. **STAGE0_REPORT.md** — superseded. Written at the end of the synthetic phase,
   before any real-photograph work. Its §1 "verified working" table is still
   accurate; its conclusions about r and softness were later contradicted on
   real photographs. Kept for history.

## Bottom line

Localisation works. Attribute control does not, for a structural reason:
τ and λ are the same knob. See RESEARCH_REPORT §17.

## The one command that reproduces the headline result

    python3 scripts/tau_sweep.py --case 17320_t1 \
        --taus 0.5,0.7,0.85 --lam-ins 4,10,16 \
        --captions <your captions.json> --out /data/outputs/grid

Expect corr(dE, struct) near −0.95. Look at the contact sheet; the automated
"WINDOW EXISTS" verdict has been wrong every time it has fired.

## Non-negotiable habits, learned the hard way

- `ulimit -n 8192` before any run that touches SAM.
- Clear `__pycache__` and `grep` for your change after every edit. Three times in
  Stage 0, byte-identical output from a stale file was briefly read as a finding.
- Captions are hand-written from the image. A wrong caption cost 3.65 dB, more
  than the entire solver effect.
- Look at the image before recording any number.
