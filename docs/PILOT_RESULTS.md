# Physical 12-attempt pilot results — 2026-10-23

Only this pilot was authorized. The protocol remains provisional; final collection, augmentation and training are paused. All expert attempts and three physical replay tests ran in the visible Isaac simulator on committed source `877f9b1`, with the pinned stack and approved cameras unchanged. No failed attempt was regenerated.

## Verdict and review

**12/12 physical full-task successes; 12/12 automated data-QA passes; 0 automatic rejections; 12 pending human visual review.** All four independent stage flags passed in every case. All processes exited 0. No forbidden-contact violation (0.05 N detection threshold), controller/evaluator failure, failed grasp, unintentional drop or failure-recovery reset was reported. No unexpected reset was reported or indicated by the recording clocks; each process performed its declared initial reset. These are actual PhysX task results, not IK/geometry proxies.

- [Consolidated visual review](../outputs/protocol_v1_pilot_report/index.html): videos, phase sheets and original full-resolution PNGs.
- [Machine-readable report](../outputs/protocol_v1_pilot_report/pilot_report.json) and [per-case CSV](../outputs/protocol_v1_pilot_report/pilot_cases.csv).
- [Automated QA and human-review template](../outputs/protocol_v1_pilot_qa/quality.json), `human_review_template.json` in the same folder.
- Raw attempts: `datasets/protocol_v1_pilot/`; collection commands/start-finish ledger, authorization and logs: `outputs/protocol_v1_pilot_run/`. These artifacts remain local and ignored by Git.

Expand each case in the review. Check cube/gripper detail at grasp, receiver acquisition, donor release, right-only hold and bin release, plus full motion. Videos are diagnostic H.264 derivatives; original lossless RGB is linked for detail inspection. Do not infer physical contact or human acceptance from image nonblank checks.

## Every declared attempt

Positions are requested settled world XY in metres; seeds are controlled per case. Actual settled position/quaternion, initial pose errors, independent stage flags, contacts, failure/reset information, timing, source hashes and data-QA/replay evidence are in the JSON. These are designed corners/reference poses, not 12 uniform random draws.

| Case | Seed | Cube XY, m | Yaw | Physics stages | Sim s | Process wall s | Raw GB | Auto QA |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |
| pilot_000 | 2026102300 | [-0.35, -0.052] | -6° | pickup/lift/handover/place PASS | 160.100 | 697.98 | 1.007 | PASS |
| pilot_001 | 2026102301 | [-0.35, -0.052] | +6° | pickup/lift/handover/place PASS | 152.767 | 685.48 | 0.961 | PASS |
| pilot_002 | 2026102302 | [-0.35, -0.044] | -6° | pickup/lift/handover/place PASS | 155.783 | 691.50 | 0.979 | PASS |
| pilot_003 | 2026102303 | [-0.35, -0.044] | +6° | pickup/lift/handover/place PASS | 151.850 | 678.49 | 0.955 | PASS |
| pilot_004 | 2026102304 | [-0.33, -0.052] | -6° | pickup/lift/handover/place PASS | 148.500 | 657.45 | 0.936 | PASS |
| pilot_005 | 2026102305 | [-0.33, -0.052] | +6° | pickup/lift/handover/place PASS | 145.333 | 650.48 | 0.912 | PASS |
| pilot_006 | 2026102306 | [-0.33, -0.044] | -6° | pickup/lift/handover/place PASS | 146.650 | 652.47 | 0.914 | PASS |
| pilot_007 | 2026102307 | [-0.33, -0.044] | +6° | pickup/lift/handover/place PASS | 145.267 | 648.47 | 0.910 | PASS |
| pilot_008 | 2026102308 | [-0.34, -0.04] | +0° | pickup/lift/handover/place PASS | 153.183 | 661.46 | 0.959 | PASS |
| pilot_009 | 2026102309 | [-0.34, -0.04] | -1° | pickup/lift/handover/place PASS | 138.033 | 582.46 | 0.859 | PASS |
| pilot_010 | 2026102310 | [-0.34, -0.04] | +1° | pickup/lift/handover/place PASS | 146.200 | 648.50 | 0.911 | PASS |
| pilot_011 | 2026102311 | [-0.34, -0.048] | +0° | pickup/lift/handover/place PASS | 146.100 | 650.47 | 0.920 | PASS |

Failure breakdown: startup/reset 0; physical grasp/lift/transfer/placement 0; forbidden-contact violations 0; incomplete attempts 0; automated recording-quality rejections 0. Human acceptance is pending for all 12; pending review is separate from a failed demonstration. There is no accepted training dataset and the pilot is excluded from future training partitions.

## Timing, throughput and storage

- Task duration: mean **149.147 s**, range **138.033–160.100 s**, excluding initial reset. Runner duration including reset is retained separately.
- Collection process wall time: **7905.22 s /2.196 h**, mean **658.77 s**, range **582.46–697.98 s**. This includes startup, planning pauses, simulation/rendering, PNG writes and shutdown.
- **35,792 complete 20 Hz observation/action rows**, **107,376 paired RGB frames**. Actual wall throughput **4.528 rows/s**, **5.465 demonstrations/hour**; the temporal dataset rate remains 20 Hz.
- 107,394 PNGs total: 107,376 paired frames plus 18 terminal diagnostic images in six attempts. Their 1–2 final physics ticks are excluded from complete action rows. No partial hold is mislabeled as a full 50 ms action.
- Raw data: **11,222,342,915 bytes /11.222 GB**, mean **0.935 GB** per completed demonstration. Sizes include lossless PNGs, arrays and diagnostic metadata.
- Measured aggregate timers: image writes **2,596.22 s**, viewport rendering **1,861.84 s**, camera capture/readback **869.86 s**, physics stepping **457.76 s**, planning including initial approach **1,663.53 s**. Reset time overlaps some physics/render timers; these are not independent additive totals. Collision planning is significant, but is not the sole wall-time cost.

| Completed demonstrations | Estimated raw GB | Observed-size range, GB | GUI generation hours at measured rate |
| ---: | ---: | ---: | ---: |
| 100 | 93.52 | 85.90–100.67 | 18.30 |
| 500 | 467.60 | 429.50–503.36 | 91.50 |
| 1,000 | 935.20 | 859.00–1006.72 | 182.99 |

GB means 10⁹ bytes. Projections are conditional on this narrow distribution/duration and current GUI/PNG workflow, not guarantees. They exclude converted copies, checkpoints and review media. A second full copy approximately doubles raw storage. The next proposed 100 train +20 validation attempts imply about **112.22 GB raw /21.96 h generation**, before QA/human review; the earlier 10–20 h estimate was optimistic.

## Synchronization and image usefulness

All 12 passed finite float32 state/action dimensions and authored joint bounds, absolute and relative clocks, exact held targets for all three 60 Hz ticks, advancing sensor frame IDs, complete RGB files, frozen intrinsics/extrinsics, required task phases and measured wrist attachment. Sensor attachment tolerance is 20 µm /rotation-matrix 2e-4. Each complete pre-action state/action row has the three corresponding images at the same physics tick. This is software simulation synchronization, not independent real-camera exposure measurement.

The report includes eight matched phase samples per case. A red-pixel proxy restricted to the projected 40 mm cube ROI found a cube-coloured region at least 12 px wide in at least one camera in all 96 samples. At initial HOME, only overhead sees the cube; its visible red ROI width is 15–28 px and some robot occlusion remains. Relevant grasp/transfer wrist samples have roughly 168–184 px red ROI width; the receiver wrist shows release at 103–105 px. These are colour/ROI measurements, not certified segmentation, full-object visibility or a quality score for every frame. Human review of complete video and original PNGs is still required. No camera parameters were changed.

Peak measured joint speed **2.4415 rad/s**, effort **1.5271 Nm**, and stable-grasp slip **1.1648 mm** remain within the existing benchmark criteria. These are simulation measurements, not physical-hardware safety certification.

## Recording format versus physical replay

All 12 recording formats passed QA. Separately, the preselected **corner pilot_000, nominal pilot_008 and −1° pilot_009** were executed through `BenchmarkEnv.step` using their recorded 12-target actions, actual reset/physics/actuators/gravity compensation and independent evaluation. No expert IK, planner or grip-feedback controller supplied replay actions. These diagnostic tests did not create new expert demonstrations.

| Replay | Physical stages | Replay sim s | Original sim s | Unused final recorded rows |
| --- | --- | ---: | ---: | ---: |
| pilot_000 | all four PASS | 157.267 | 160.100 | 56 |
| pilot_008 | all four PASS | 150.350 | 153.183 | 56 |
| pilot_009 | all four PASS | 135.200 | 138.033 | 56 |

**3/3 physical replay successes; the other nine are not physically replayed.** The intended policy interface terminates at the first independent success. It therefore omits 56 recorded settling-tail rows in each tested case; this is not playback of every stored row. Pointwise joint/cube-state equivalence was not measured. Only elapsed time and desired-target tracking metrics differ from the original aggregate evaluations; the target representation is float32, while original desired-pose calculations used higher precision. No physical failures were hidden or repaired. Replay logs/commands/results: `outputs/protocol_v1_pilot_replay/`.

The new nominal expert episode exactly matches the prior camera-regression reference independent metrics and complete transfer trace (`outputs/controller/overview_v3_comparison/full_20261023T142830953036Z.json`). Expert/model/planner/scene/physics files remain unchanged.

## Recommendation and stop gate

Retain the tested primary support: **x [−0.350,−0.330] m, y [−0.052,−0.044] m, yaw ±6°, fixed validated HOME**, fixed bin/physics/cameras. The three nominal-y reference cases remain separate regression checks. This purpose-selected 12/12 result is not a general success probability or proof that every interior pose will work. Do not widen ranges or silently replace failures.

After human data-quality acceptance and separate authorization, proceed with the proposed **100 train /20 validation** attempt manifests and fixed held-out evaluation set. Use nested smaller training subsets later if desired, retaining source-case splits. Do not jump to 500/1,000 until the first data/training/evaluation cycle demonstrates useful learning. The measured costs justify profiling lossless image writing and rendering before large scale; any optimization must preserve decoded RGB, synchronization and the controller. No optimization, augmentation or training was performed in this pilot.

**Stop here for user review.** All 12 pilot attempts are consumed. No final collection, dataset augmentation, conversion into an accepted training corpus, optimizer updates or remote push is authorized.

## Reproduce the report without collecting

Activate `leisaac-sim` as in README. Choose a fresh report folder:

```bash
python scripts/demonstration_pipeline.py pilot-report \
  --manifest outputs/demonstration_protocol_v1/pilot.json \
  --run outputs/protocol_v1_pilot_run \
  --quality-report outputs/protocol_v1_pilot_qa/quality.json \
  --replay-run outputs/protocol_v1_pilot_replay \
  --output outputs/protocol_v1_pilot_report_rebuilt
```

The command checks the complete attempt denominator and unchanged automatically validated raw bytes, samples visibility, and writes JSON/CSV/HTML. It never starts Isaac or grants human approval. All 45 CPU tests passed in **72.465 s** after adding report support; six focused protocol tests passed again after the final reporting cleanup. Final checks matched all 12 run evaluations to their hash-bound raw evaluations, decoded the 18 terminal diagnostic PNGs, confirmed equal camera frame counters at every capture, and verified all 326 local HTML links. Control/scene/camera/protocol files remain byte-identical to `877f9b1`.
