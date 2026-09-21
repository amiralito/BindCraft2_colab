# BindCraft 2 Design Pipeline (Colab) - **BETA**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amiralito/BindCraft2_Colab/blob/main/BindCraft2_design_pipeline.ipynb)

End-to-end protein binder design on Google Colab with
**[BindCraft 2 (BC2)](https://github.com/PacesaLab/BindCraft2)** — AlphaFold 2 gradient design
(hallucination) → ProteinMPNN redesign → re-prediction with held-out AF2 models → filters →
ranking by `i_pDAE`.

Every design modality is driven from form fields, with a raw-JSON escape hatch for full control:
de novo miniproteins, large binders, linear and cyclic peptides, homo-oligomers, multidomain
binders, the scaffolded antibody formats (VHH, scFv, Fab), ankyrin repeat proteins, and the
conformational objectives `induced_fit` and `fold_switch` — plus multi-target campaigns and
off-target detargeting.

> BC2 ships its own Colab notebook. This one is a rewrite in the house style used by
> [RFdiffusion3_Colab](https://github.com/amiralito/RFdiffusion3_Colab),
> [Protenix_Colab](https://github.com/amiralito/Protenix_Colab) and
> [Tiberius_Colab](https://github.com/amiralito/Tiberius_Colab): `#@param` form cells,
> timestamped run folders on Drive, a cached weight store, a raw-input override cell, metric
> plots with chain-boundary markers, and a packaged tar.gz result bundle.

## Features

- **Two ways to configure a campaign** — a form builder covering every shipped modality, property
  and filter threshold, or paste raw campaign JSON for anything the form can't reach
  (`losses`, per-loss `weights_*`, `filters`, `binder_shapes`, `binder_scaffold`,
  `parameter_sweep`, multitarget scheduling, `mpnn_variant`, `validation_models`, …).
- **Google Drive integration** — the 5.3 GB AF2 parameter set and the JAX compiled-graph cache are
  stored once and reused by every future session.
- **Timestamped, named runs** — each run writes to `runs/<date-time>_<name>/`, so runs never
  overwrite each other. `RESUME_RUN` carries on in an existing folder.
- **Resume after a Colab timeout** — `resume` is on and results live on Drive, so re-running the
  run cell picks up where it stopped, claiming only the trajectories it hasn't run.
- **Multi-target and detargeting** — add targets one at a time; mark any of them `detarget` to
  select against an off-target, with the detarget iPTM ceilings applied at every stage.
- **Memory-aware target prep** — the target cell prints BC2's own per-worker memory estimate
  (`2.0 × (3.4 GB + 38 kB × N²)`) for your card before you commit, and the trim cell narrows the
  input to the chains and residue window you design against while preserving residue numbering,
  so hotspot/coldspot selections stay valid.
- **Diagnostics that read the rejections** — accepted designs plotted against the full candidate
  distribution, a tally of which filters are rejecting candidates and where attempts stopped,
  per-residue pLDDT along the complex with chain boundaries, and the gradient-design loss curve.
- **Clean outputs** — ranked table, accepted complexes, binder FASTA, a ChimeraX `.cxc`, and
  `campaign_metadata.json`, packed into one tar.gz.

## Pipeline

```
AF2 gradient design   →   ProteinMPNN      →   AF2 re-prediction   →   filters   →   rank
(hallucination,           (sequence            (held-out models,       (acceptance    (i_pDAE)
 5 multimer models)        redesign)            2 monomer models)       criteria)
```

## Requirements

- Google Colab with a **GPU runtime** (`Runtime → Change runtime type → GPU`).
  **There is no CPU path** — a single trajectory folds an AF2 ensemble hundreds of times.
- **T4 (16 GB) is the floor**; L4 / A100 much more comfortable. Compute capability ≥ 8.0 gets
  native bfloat16.
- Python 3.12 or newer in the runtime (BC2's minimum).
- ~11 GB free disk while the 5.3 GB parameter archive unpacks; cache it on Drive to download once.

## Quick start

1. Open the notebook in Colab, select a GPU runtime.
2. Run **step 1** (GPU check) and **Setup** (name the run, mount Drive).
3. Run **step 2** (install BC2) and **step 3** (AF2 parameters — skipped if cached).
4. Add your target(s) (**step 4**) and trim them (**step 5**).
5. Configure the campaign — the **form (step 6)** or the **raw JSON cell (step 6b)**.
6. Run it (**step 7**), read the results (**steps 8–10**), package them (**step 11**).

Tick `TEST_RUN` in step 6 to watch the whole notebook run through in minutes. It opens every stage
gate and switches every filter off, so the first attempt is accepted — **nothing it accepts means
anything**; it only checks the plumbing.

## Notebook structure

| Step | Purpose |
|------|---------|
| 1 | Check the allocated GPU |
| Setup | Name the run, mount Drive, set the per-run `WORK` directory, point at the cached weights |
| 2 | Clone and install BindCraft 2 (accelerator wheels chosen from the driver and the card) |
| 3 | Fetch the AF2 parameters (skipped if cached) |
| 4 | Add a target — shipped preset, PDB ID, upload or path; chains, hotspots, coldspots, objective |
| 5 | Trim / clean a target (chains, residue window, strip non-polymer) |
| 6 | Build `campaign.json` from form fields |
| 6b | Paste raw campaign JSON to override the form |
| 7 | Run the campaign, with live counts and a heartbeat |
| 8 | Accepted designs, candidates and their failed filters, attempts |
| 9 | Metric distributions, per-residue pLDDT, gradient-design trajectory |
| 10 | py3Dmol viewer for the accepted complexes |
| 11 | FASTA, ChimeraX script, `campaign_metadata.json`, tar.gz bundle |

## Configuring a campaign

### Form (step 6)

**Binder format** (`MODALITY`):

| Modality | Use it for |
|----------|------------|
| `binder` | De novo miniprotein, folded from nothing |
| `large_binder` | Binders over ~300 aa |
| `peptide` | Linear peptides below 25 aa (may fold only when bound) |
| `cyclic_peptide` | Head-to-tail cyclisation geometry |
| `homo_oligomer` | Identical binder chains — set `COPIES` |
| `multidomain` | Several domains on one chain |
| `VHH`, `scFv`, `Fab`, `ARP` | Scaffolded formats — the scaffold sets the length |

`EXTRA_MODALITY` adds a compatible conformational objective: `induced_fit` (the interface moves on
binding) or `fold_switch` (the whole fold changes). BC2 checks declared incompatibilities before
starting.

**Properties** — `forced_targeting` (concentrate binding on named hotspots *and* require the
accepted design to touch them), `humanize`, `protease_stable`, `disulfide_staple`,
`mixed_topology`, `termini_together`, `termini_accessible`, `initial_guess`, `bigbang`.

**Targets** — run step 4 once per target with `RESET_TARGETS` unticked after the first.
`OBJECTIVE = "detarget"` marks an off-target, which adds the `max_detarget_iptm_*` ceilings at
every stage. Hotspots and coldspots use the **input file's** residue numbering, chain-prefixed
(`A54,B12-16`) for multi-chain targets.

### Raw JSON (step 6b)

For anything the form can't express. The cell validates your keys against BC2's own
`settings/core/reference.json` and names anything it doesn't recognise. `__TARGETS__` and
`__PROJECT__` are substituted with what steps 4–5 built; replace them with literals to bypass.

`target_path` entries resolve relative to the JSON file's directory, so the absolute paths
steps 4–5 write are the safe form.

## Reading the results

Start with `3_Ranked/!_Ranked.csv` — accepted designs, best first by **`i_pDAE`** (distance-masked
interface confidence, 0–1, higher better). Then look at the structures.

| Metric | Scale |
|--------|-------|
| `i_pDAE` | Distance-masked interface confidence, 0–1, higher better. BC2's ranking score. |
| `i_pTM` | Interface confidence, 0–1, higher better |
| `i_pAE` | Mean interface PAE / 31 Å, lower better |
| `pLDDT` | Binder confidence in the bound state, 0–1 |
| `Interface_Residues` | Binder residues within 4 Å of the target |
| `Interface_BuriedArea` | Binder-side buried SASA, Å² |
| `Surface_Hydrophobicity` | Exposed hydrophobic fraction — high values may aggregate |

**None of these are affinities.** A confident pose can still be biologically inaccessible: check the
binder against the full-length target rather than the crop, and consider glycans, membrane
orientation and competing partners. `autotuned` in the trajectory table names attempts that ran on
the desperation ladder — against an easier task than the campaign asked for.

Re-rank and re-filter without redesigning, from a clone:

```bash
python3 bindcraft.py rank   <results folder> --on i_pTM
python3 bindcraft.py filter <results folder> --where Interface_BuriedArea>=600
python3 bindcraft.py score  <structure.cif>
```

## Scaling beyond Colab

Colab is for a handful of designs and for working out settings. A real campaign belongs on a
cluster — `sbatch bindcraft.slurm campaign.json` from a clone, using the `campaign.json` this
notebook writes. Because `resume` is on, a Colab session and a cluster job can even share a
results folder.

## Citation

BindCraft 2 is by the [Pacesa lab](https://github.com/PacesaLab). It builds on
[AlphaFold 2](https://github.com/google-deepmind/alphafold) (DeepMind),
[ColabDesign](https://github.com/sokrypton/ColabDesign) (Sergey Ovchinnikov),
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (Justas Dauparas) and HyperMPNN.
This notebook only drives it — cite the underlying work.
