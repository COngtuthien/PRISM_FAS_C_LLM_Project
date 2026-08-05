# PRISM-FAS-B v1.1 specification snapshot

Source: `PRISM_FAS_B_v1_1_CASIA_MSU_SiWMv2_LocalPreprocess_DualBackend_Codex_Spec.docx`

PRISM-FAS-B

v1.1 — Local Data Factory & Dual-Backend Training

CASIA-FASD + MSU-MFSD for source training  |  SiW-Mv2 for target-only testing

ĐẶC TẢ NGHIÊN CỨU VÀ TRIỂN KHAI CHI TIẾT DÀNH CHO CODEX

Tài liệu này thay thế các lựa chọn dữ liệu và hạ tầng trong bản v1.0, nhưng giữ nguyên tinh thần nghiên cứu: recipe theo thuộc tính vật lý, synthetic artifact có mask, regional real manifold và open-set fusion.

Trạng thái tài liệu và các quy tắc không thương lượng

LLM chỉ sinh structured attack recipe và prior text; không trực tiếp tạo pixel và không tham gia fast-path inference của MVP.

Raw dataset không bao giờ upload lên Modal trong flow mặc định. Chỉ processed package đã được audit trên laptop mới được đồng bộ.

CASIA-FASD và MSU-MFSD là hai source domains duy nhất cho train/dev. SiW-Mv2 là target-only evaluation.

Mọi threshold, prototype count, loss coefficient, synthetic ratio và checkpoint selection chỉ dựa trên source-dev.

Mỗi run phải ghi git commit, config resolved, seed, data package hash, model weight hash và environment fingerprint.

Trainer core tuyệt đối không import Modal. Modal chỉ là wrapper gọi đúng trainer core như local runner.

Mọi pipeline phải idempotent và resume-safe: chạy lại không tạo sample ID mới, không ghi đè mù và không làm thay đổi manifest hash nếu input/config không đổi.

Không claim kết quả hoặc novelty trước khi có thí nghiệm. Không dùng từ “first” nếu chưa systematic review.

Mục lục triển khai

1. Tóm tắt kiến trúc đã chốt

2. Mục tiêu, phạm vi và các quyết định thiết kế

3. Protocol dữ liệu cuối cùng: CASIA-FASD + MSU-MFSD -> SiW-Mv2

4. Local Data Factory: xử lý dataset hoàn toàn trên laptop

5. Data package, manifest, cache và shard contract

6. Anti-leakage và target isolation

7. Attack ontology và recipe bank

8. Synthetic engine và quality gate

9. Regional CNN-VLM detector và real manifolds

10. Loss, batch composition và optimization

11. Flow huấn luyện theo giai đoạn

12. Kiến trúc dual-backend: GPU PC và Modal

13. Configuration, CLI và path mapping

14. Cấu trúc source code và module contracts

15. Logging, checkpoint, resume và run registry

16. Evaluation trên SiW-Mv2 và report contract

17. Baselines, ablations và reliability tests

18. Unit/integration tests bắt buộc

19. Kế hoạch triển khai cho Codex

20. Definition of Done và checklist tái lập

Phụ lục A-F. Schema, YAML, lệnh chạy, Modal wrapper, Windows bootstrap và model registry

1. Tóm tắt kiến trúc đã chốt

PRISM-FAS-B v1.1 dùng hai dataset nguồn nhỏ nhưng khác domain — CASIA-FASD và MSU-MFSD — để học real/spoof cues và tạo pseudo-unknown attacks. SiW-Mv2 được giữ hoàn toàn ngoài quá trình phát triển để đo cross-domain và cross-attack generalization. Toàn bộ việc giải nén, đọc metadata, lấy frame, detect/crop mặt, cache geometry prior và đóng gói shard đều chạy trên laptop. Sau khi package được audit và đóng băng, cùng một trainer có thể chạy trên GPU PC hoặc Modal GPU mà không thay logic khoa học.

Hình 1. Luồng tổng thể: local preprocessing, processed package bất biến và hai backend huấn luyện dùng chung trainer core.

1.1. Pipeline khoa học

1.2. Cấu hình MVP chốt

2. Mục tiêu, phạm vi và các quyết định thiết kế

2.1. Mục tiêu kỹ thuật

2.2. Không nằm trong MVP

Không train trực tiếp trên SiW-Mv2; không leave-one-attack-out trên SiW-Mv2 trong core experiment.

Không dùng CelebA-Spoof, OULU-NPU, Replay-Attack hoặc dataset khác trong main result.

Không dùng masked diffusion trong MVP; chỉ thêm sau khi physics + GPAT + manifold chạy ổn.

Không triển khai web dashboard trước khi CLI, run registry và report HTML hoạt động.

Không tối ưu edge deployment trong giai đoạn đầu; latency detector chỉ được đo sau khi correctness đạt DoD.

Không tự động fine-tune text encoder từ đầu; chỉ LoRA/adapter sau ablation frozen backbone.

2.3. Research question và giả thuyết

H1: Domain-balanced training trên CASIA + MSU tốt hơn naive concatenation khi đánh giá SiW-Mv2.

H2: Multi-prototype regional real manifold giảm false reject trên real target so với một global center.

H3: Mask-aware outlier loss tốt hơn image-level outlier loss với partial/local attacks.

H4: Structured recipe bank tốt hơn random augmentation cùng số sample và cùng detector.

H5: Quality-weighted synthetic samples tốt hơn coi mọi synthetic sample có trọng số bằng nhau.

H6: Cùng một data package và config cho local/Modal cho kết quả nằm trong expected seed variance.

3. Protocol dữ liệu cuối cùng: CASIA-FASD + MSU-MFSD -> SiW-Mv2

3.1. Vai trò dataset

3.2. Source split policy

Mặc định reproducible và dễ triển khai nhất: adapter giữ nguyên official train/test grouping của từng source dataset. Trong project này, official train được ánh xạ thành source_train; official test được ánh xạ thành source_dev. Source_dev chỉ dùng chọn checkpoint, calibration và quality-gate threshold, không được báo cáo như final test benchmark.

Không random split theo frame.

Nếu official metadata không cung cấp subject ID rõ, dùng video group làm đơn vị tối thiểu và ghi subject_id=null.

Tất cả frame từ một video bắt buộc ở cùng split.

Nếu dataset copy hiện có không đi kèm protocol file, adapter phải fail với thông báo rõ; không được đoán tên folder.

Tùy chọn source cross-validation chỉ là ablation riêng và phải có config khác, không thay core protocol.

3.3. Target test policy

SiW-Mv2 processed package tách thành features manifest và evaluation labels manifest.

Training process chỉ được mount source package và không có đường dẫn tới SiW-Mv2.

Target inference xuất prediction trước, sau đó evaluator mới join label theo sample_id/video_id.

Threshold live/spoof, unknown/reject và video aggregation được đóng băng từ source_dev.

Attack family của SiW-Mv2 chỉ được dùng post-hoc để tính attack-wise APCER sau khi prediction đã lưu.

3.4. Sampling unit và batch balance

4. Local Data Factory: xử lý dataset hoàn toàn trên laptop

Hình 2. Stage 0 được chạy local và tạo processed package duy nhất cho cả PC lẫn Modal.

4.1. Cấu trúc thư mục local đề xuất

Tất cả đường dẫn thực tế lấy từ biến môi trường hoặc config. Code không được hard-code D:\; đường dẫn trên chỉ là layout khuyến nghị cho Windows.

4.2. Stage S0 — Inventory và checksum

Quét raw tree và tạo raw_inventory.parquet: relative_path, size, modified_time, sha256, extension.

Xác minh file video đọc được bằng ffprobe/OpenCV; ghi codec, fps, frame_count, width, height.

Đọc official protocol/metadata qua dataset adapter; nếu thiếu file bắt buộc thì fail trước khi extract frame.

Tạo dataset_fingerprint = SHA256(sorted(relative_path + sha256 + adapter_version)).

Không hash lại file không đổi: dùng local checksum cache theo size + mtime nhưng có chế độ --full-hash để audit cuối.

4.3. Stage S1 — Dataset adapters

4.4. Stage S2 — Deterministic frame sampling

Không random frame theo epoch trong MVP; frame set được đóng băng trong processed package.

frame_id = integer frame index gốc; timestamp_ms được lưu.

Nếu video có ít frame hơn N, lấy tất cả frame hợp lệ và ghi sampling_status=short_video.

Frame decode fail không được im lặng bỏ; ghi preprocess_failures.parquet với reason.

4.5. Stage S3-S4 — Face detection và canonical crop

4.6. Stage S5 — Offline priors và quality cache

Mỗi prior file đặt theo sample_id; không dùng absolute raw path trong NPZ.

preprocessing_version và model revision phải đi cùng mỗi cache record.

Nếu FaceXFormer dependency xung đột, chạy môi trường prism-preprocess-geometry riêng nhưng output schema không đổi.

Cache computation phải có --resume và skip sample khi output tồn tại, checksum hợp lệ.

4.7. Stage S6-S7 — Packaging và audit

4.8. Lệnh local preprocessing contract

5. Data package, manifest, cache và shard contract

5.1. Canonical sample schema

5.2. Stable sample ID

Không đưa absolute path, machine name hoặc run timestamp vào sample_id. Cùng raw data + adapter + frame policy + preprocessing version phải tạo cùng sample_id trên mọi máy.

5.3. Shard policy

WebDataset-compatible tar shard, mặc định 1.000 samples/shard hoặc tối đa 1.5 GB/shard.

Mỗi sample trong tar gồm <sample_id>.png, <sample_id>.npz và <sample_id>.json.

Shard được nhóm theo split; không trộn source_train với source_dev/target_test.

Không nén tar bằng gzip trong MVP để random/stream read nhanh hơn trên Volume.

Mỗi shard có sha256 và row count trong shards_index.parquet.

Nếu train local từ loose files, dataset loader vẫn phải đọc cùng manifest và trả cùng sample dict.

5.4. Package lock

6. Anti-leakage và target isolation

6.1. Filesystem isolation

6.2. Code-level guards

TrainConfig validator reject bất kỳ manifest có project_split=target_test.

Dataset loader assert label is not None cho source loader và assert dataset in {casia_fasd, msu_mfsd}.

Recipe builder chỉ nhận source ontology; không nhận SiW paths, labels hoặc attack family.

Checkpoint metadata lưu source manifest hashes; evaluator không cho save optimizer state sau khi join target labels.

CI test quét resolved config và run manifest để tìm chuỗi siw_mv2 trong train stage; chỉ exception là blocked-target list.

6.3. Blind-like evaluation flow

7. Attack ontology và recipe bank

7.1. Vai trò

LLM sinh JSON recipe theo các thuộc tính vật lý tổng quát. LLM không biết target taxonomy, không xem SiW-Mv2 và không tạo ảnh. Recipe compiler chuyển JSON thành operator graph, region mask policy và conditioning vector cho physics/GPAT.

7.2. Recipe schema v1.1

7.3. Bank generation and freeze

Generate candidate recipes với fixed seed ranges và constrained JSON schema.

Rule validation: coverage, region compatibility, artifact-medium consistency, severity budget.

Diversity validation trên categorical axis coverage + text embedding distance.

Compile test: recipe phải map được thành operator graph hiện có.

Đóng băng recipes.jsonl, ontology.yaml, prompt.txt, model ID/revision và BANK_LOCK.json.

Mọi thay đổi bank tạo version mới; không patch bank giữa các run chính.

7.4. MVP recipe coverage

8. Synthetic engine và quality gate

8.1. Routes

8.2. GPAT core

8.3. Pair sampler

Live target và spoof source đều chỉ từ source_train.

Ưu tiên khác identity nếu subject_id có sẵn; nếu không, khác video_id là bắt buộc.

Cân bằng CASIA/MSU và print/replay-like source groups nếu metadata cho phép.

Optional compatibility cost dùng pose/scale/illumination, không dùng target identity similarity để ép cùng người.

Pair plan được materialize thành pair_manifest.parquet để run có thể tái lập.

8.4. Quality gate

8.5. Synthetic bank layout

9. Regional CNN-VLM detector và real manifolds

9.1. Inputs and region set

RGB 224x224 train input từ canonical 256 crop.

Optional high-pass/wavelet detail computed on-the-fly.

Region set MVP: left_eye, right_eye, nose, mouth, forehead, left_cheek, right_cheek, face_boundary, context.

Region prior là soft mask/query initialization; không crop cứng thành chín ảnh riêng.

Visibility mask ngăn prototype update từ region bị khuất hoặc detection lỗi.

9.2. Architecture

9.3. Multiple real prototypes

MVP K_r=4 cho mọi region; tune K in {2,4,6} trên source_dev.

Khởi tạo bằng K-means trên source_train live embeddings sau detector warm-up.

MVP dùng diagonal covariance + epsilon floor để tránh singular matrix.

Prototype update chỉ dùng live samples và region visibility hợp lệ.

Lưu prototype state trong checkpoint và export prototypes.npz.

9.4. Image-level fusion

10. Loss, batch composition và optimization

10.1. Losses

10.2. Batch contract

Ví dụ batch size 32: 12 live + 12 real spoof + 8 synthetic. Nếu GPU nhỏ, dùng gradient accumulation nhưng giữ effective composition trên accumulation window.

10.3. Default optimization

10.4. Initial loss weights

Các giá trị chỉ là điểm khởi đầu. Search space phải được khai báo trước và tune source-only.

11. Flow huấn luyện theo giai đoạn

11.1. Stage transition rules

Mỗi stage nhận input lock/hash và tạo output lock/hash.

Stage không được đọc artifact tương lai; G5 không đọc target package.

Resume dùng last checkpoint chỉ khi config hash và data package hash trùng.

Nếu thay data package, recipe bank hoặc model revision, phải tạo run_id mới.

Synthetic bank có thể build một lần rồi tái sử dụng nhiều detector runs nếu BANK_LOCK trùng.

11.2. State machine

12. Kiến trúc dual-backend: GPU PC và Modal

Hình 3. Backend chỉ khác runner/resource binding; TrainerCore, config và data contract không đổi.

12.1. Core design rule

12.2. Runtime interfaces

12.3. Local PC runner

12.4. Modal runner

12.5. Đồng bộ processed data lên Modal

Chỉ upload data/processed/prism_data_v1 hoặc selected shards; không upload raw/.

Trước upload, validate-package phải pass và PACKAGE_LOCK status=validated.

Sau upload, chạy remote verify để tính/check manifest/shard hashes.

Không sửa file trong package đã upload. Version mới dùng subfolder/package_id mới.

Synthetic bank có volume path riêng theo BANK_LOCK hash để tránh overwrite.

12.6. Modal app contract

12.7. Same-run portability

A run có thể bắt đầu trên PC và resume trên Modal nếu checkpoint, resolved config, package hashes và code commit trùng.

Không cho resume nếu optimizer/library state không tương thích; tool phải báo diff thay vì silently load partial.

Checkpoint chứa model/optimizer/scheduler/scaler/RNG states/prototypes/epoch/global_step.

Backend name được log nhưng không tham gia model state hoặc sample ordering.

DataLoader seed kết hợp base_seed + epoch + rank; worker seed deterministic.

13. Configuration, CLI và path mapping

13.1. Config hierarchy

13.2. Path mapping

13.3. Runtime YAML examples

13.4. Unified CLI

13.5. Config validation

Pydantic/dataclass schema validate before importing torch-heavy modules.

Reject unknown keys để tránh typo silently ignored.

Resolve config and save before run starts.

Validate paths and package hashes before allocating GPU.

Print concise run plan: backend, GPU, dataset counts, batch composition, estimated steps, output path.

14. Cấu trúc source code và module contracts

14.1. Dataset __getitem__ contract

14.2. Model forward contract

14.3. Separation of concerns

15. Logging, checkpoint, resume và run registry

15.1. Run directory

15.2. run.json required fields

run_id, parent_run_id, stage, status, backend, host/GPU info.

git_commit, dirty flag, package lock hash, recipe bank hash, synthetic bank hash.

resolved config hash, seed, start/end timestamp.

best checkpoint rule và selected checkpoint path.

metric summary và links/relative paths tới report artifacts.

resume lineage nếu run chuyển backend.

15.3. Checkpoint rules

Save last every fixed number of optimizer steps và at epoch end.

Save best theo source_dev criterion, không theo SiW result.

Atomic write; verify file size/hash before replacing pointer.

Keep last, best và top-3; optional prune old epoch files sau COMPLETE.

Modal Volume commit sau best/last checkpoint quan trọng.

Resume restores Python, NumPy, Torch CPU/CUDA RNG states và sampler epoch.

15.4. Backend parity audit

16. Evaluation trên SiW-Mv2 và report contract

16.1. Frame and video outputs

16.2. Calibration

Temperature scaling fit source_dev only.

Live/spoof threshold chosen by declared source-dev criterion: min ACER hoặc BPCER constraint.

Unknown/reject threshold chosen bằng source-dev corruptions/synthetic unknown exposure, không dùng target quantile.

Calibration JSON lưu raw criterion, threshold, temperature, source-dev metrics và hash.

16.3. Metrics

16.4. report.html sections

Run identity and reproducibility block.

Dataset/package counts and excluded samples.

Source-dev checkpoint selection and calibration.

SiW-Mv2 overall frame/video metrics.

Attack-wise and region-wise analysis.

Reliability/calibration plots and risk-coverage.

Confusion matrix and threshold table.

Failure gallery using sample IDs and local relative paths; no raw sensitive metadata.

Compute and backend information.

Ablation table and statistical comparison.

17. Baselines, ablations và reliability tests

17.1. Baselines bắt buộc

17.2. Key ablations

17.3. Shortcut and causal tests

Train probe phân biệt synthetic và real spoof; accuracy quá cao báo generator fingerprint.

Test live with JPEG/resize/color corruption nhưng không spoof; score không được tăng mạnh có hệ thống.

Residual scale=0 phải trả score gần live và region distance giảm.

Đổi recipe region phải làm anomaly heatmap dịch tương ứng.

Hoán đổi artifact map giữa samples phải làm local supervision performance giảm.

Different crop padding/interpolation test để phát hiện preprocessing shortcut.

Cross-route: train GPAT test physics synthetic và ngược lại.

Benign glasses/makeup/low-light stress để kiểm tra BPCER.

18. Unit/integration tests bắt buộc

18.1. Data tests

Adapter fixture parses expected video count/splits from miniature directory tree.

Frame sampler deterministic and no duplicate indices unless video shorter than requested N.

No video_id appears in more than one project_split.

Target feature manifest contains no label column/value.

Crop output is RGB PNG 256x256 and hash stable.

Package lock changes when frame config or model revision changes.

Corrupt video and no-face sample are recorded, not silently dropped.

18.2. Model/loss tests

Forward output shape for B=2 and all R regions.

Mahalanobis distance finite with covariance floor.

Live prototype distance decreases after one optimization step on synthetic toy data.

Mask-aware outlier loss only applies to attacked regions.

Zero artifact map yields zero/near-zero local synthetic loss.

Fusion score monotonic when one evidence source increases with others fixed.

AMP forward/backward finite.

18.3. Backend and resume tests

Local smoke test 5 steps on CPU/GPU.

Modal smoke test 5 steps using tiny fixture package.

Checkpoint round-trip preserves global_step, optimizer, scheduler, prototypes and RNG.

Interrupted run resumes at next correct batch/epoch boundary.

Resolved config mismatch blocks resume with human-readable diff.

Volume path and local path produce identical canonical relative paths.

18.4. Evaluation tests

Known fixture validates APCER/BPCER/ACER formula and label convention.

Frame-to-video aggregation deterministic.

Evaluation command cannot write checkpoint or optimizer file.

Scoring refuses if prediction checkpoint/calibration hash missing.

Report generation works with missing optional attack metadata.

19. Kế hoạch triển khai cho Codex

19.1. Milestones and acceptance criteria

19.2. Codex guardrails

Trước mỗi code batch, đọc config/schema và tests liên quan; không thay API tùy ý.

Mỗi PR/commit chỉ giải quyết một milestone hoặc một vertical slice.

Không download dataset/model tự động trong test suite.

Không thêm dependency nặng nếu standard library/đang có đã đủ.

Tất cả external model loading phải pinned revision và có offline cache path.

Không swallow exception bằng broad except; failure phải có sample_id/stage/reason.

Không dùng notebook làm implementation chính; notebook chỉ analysis/report optional.

Mỗi CLI write artifact phải hỗ trợ --dry-run hoặc validate trước khi thay đổi dữ liệu.

19.3. First Codex task prompt

20. Definition of Done và checklist tái lập

20.1. MVP Definition of Done

Local Data Factory builds a validated package from all three datasets without manual crop copying.

CASIA/MSU source_train and source_dev are subject/video-disjoint according to adapter metadata.

SiW target feature manifest is label-free and inaccessible to train commands.

B00 baseline trains and evaluates end-to-end on both PC and Modal using the same trainer core.

Recipe bank and synthetic bank are versioned, deterministic and source-only.

Regional manifold model trains without NaN/collapse and exports prototype state.

Full model outputs frame/video metrics and report.html for SiW-Mv2.

At least 3 seeds for main baseline and full method; mean/std reported.

All required tests pass; run package contains hashes/config/checkpoints/predictions/report.

No result is selected based on SiW-Mv2 metrics.

20.2. Reproducibility checklist

☐ Raw dataset version and raw fingerprint recorded.

☐ Official metadata/protocol file paths and adapter versions recorded.

☐ Frame sampling strategy, N, head/tail skip and seed recorded.

☐ Face detector model/revision, threshold, padding and resize interpolation recorded.

☐ Crop/output format and preprocessing version recorded.

☐ Parsing/identity model revisions and cache hashes recorded.

☐ Package lock, manifest hashes and shard hashes recorded.

☐ Recipe prompt, model revision, schema, seed and bank hash recorded.

☐ Synthetic generator checkpoints, operator parameters, pair plan and quality thresholds recorded.

☐ Backbone checkpoints, region definitions, K, covariance type and initialization recorded.

☐ Loss formulas/weights, optimizer, LR, scheduler, batch composition and AMP recorded.

☐ Backend/GPU/library environment and git state recorded.

☐ Checkpoint selection and source-only calibration criterion recorded.

☐ Prediction, aggregation and metric code version recorded.

☐ Seeds, repeats, negative results, compute cost and failures included.

Phụ lục A. Example data YAML

Phụ lục B. Example experiment YAML

Phụ lục C. Windows bootstrap

Phụ lục D. Dependency groups

Phụ lục E. Modal operational commands

Phụ lục F. Model registry fields

Tài liệu nền và ghi chú kỹ thuật

PRISM-FAS-B v1.0 internal research specification, 2026.

GPAT-FAS internal specifications on geometry-preserving artifact transfer and artifact-map supervision.

Official dataset publications/protocols for CASIA-FASD, MSU-MFSD and SiW-Mv2.

Official Modal documentation for Apps/entrypoints, GPU configuration, Volumes, volume CLI and detached long-running jobs; commands must be rechecked when Modal version is upgraded.

Official model repositories/model cards for SCRFD, FaceXFormer, AdaFace, ConvNeXt V2 and SigLIP2; pin exact revisions before experiments.

## Table 1

Thuộc tính | Giá trị
Loại tài liệu | Research method specification + software implementation contract
Phiên bản | PRISM-FAS-B v1.1 — 08/2026
Nguồn train/dev | CASIA-FASD và MSU-MFSD duy nhất; không dùng CelebA-Spoof
Target test | SiW-Mv2; target-only, không tune và không sinh recipe từ target
Xử lý dataset | Chạy hoàn toàn trên laptop/PC; có resume, audit và đóng gói dữ liệu bất biến
Huấn luyện | Cùng một codebase chạy được trên GPU PC hoặc Modal GPU
Trạng thái khoa học | Thiết kế đề xuất; mọi contribution phải được kiểm chứng bằng thực nghiệm

## Table 2

Implementation mandate: Codex phải triển khai theo contract trong tài liệu này, ưu tiên code chạy được, kiểm thử được và tái lập được. Không tự suy đoán cấu trúc dataset, không hard-code đường dẫn máy cá nhân, không dùng SiW-Mv2 để chọn checkpoint hoặc hyperparameter.

## Table 3

Quyết định quan trọng: Modal không phải nơi “chuẩn bị dữ liệu”. Modal chỉ nhận processed package đã có hash và chạy compute-heavy stages. Nhờ đó kết quả local và Modal có thể so sánh trực tiếp, đồng thời tránh phát sinh hai pipeline preprocessing khác nhau.

## Table 4

Khối | Input | Output | Chạy ở đâu
Data Factory | Raw CASIA/MSU/SiW videos + official metadata | Crops, priors, manifests, shards, audit report | Laptop/PC duy nhất
Recipe engine | Ontology source-only + frozen LLM | Versioned recipe bank JSON | Laptop hoặc CPU job; output đóng băng
Synthetic bank | Source live/spoof + recipe | Synthetic PNG, artifact map, mask, q score | PC GPU hoặc Modal; cùng code
Detector training | Real live, real spoof, synthetic spoof | Checkpoint + prototypes + calibration | PC GPU hoặc Modal
Target inference | SiW-Mv2 processed samples, không cần label | Prediction parquet + trace | PC GPU hoặc Modal evaluation app
Metric evaluation | Prediction + evaluation-only labels | Metrics, plots, report | Laptop hoặc isolated evaluation job

## Table 5

Thành phần | Mặc định MVP | Fallback/ablation
Face detector | SCRFD-2.5G, frozen | SCRFD-10G / RetinaFace
Canonical crop | 256x256 lossless PNG, train resize 224 | 224 direct / 320 crop
Local branch | ConvNeXt V2 Atto hoặc Tiny | CDC-ResNet18 / LDC branch
Global branch | SigLIP2 Base P16-224 frozen, train fusion/heads | DINO ViT-S/16 không text
Geometry prior | FaceXFormer offline cache | MediaPipe + BiSeNet fallback
Identity/quality gate | AdaFace IR-50 frozen | ArcFace iResNet-50
Synthetic route | Physics + GPAT residual | Physics-only; diffusion không thuộc MVP
Regional prototypes | K=4/region, diagonal covariance | K in {2,4,6}; global center baseline
Precision | AMP fp16/bf16 khi tương thích | fp32 debug
Primary input | Image/frame-level; video aggregate khi evaluate | Single-frame ablation

## Table 6

ID | Mục tiêu kiểm chứng
O1 | Xây local preprocessing pipeline deterministic cho ba dataset và tạo một canonical data contract chung.
O2 | Train chỉ bằng CASIA-FASD + MSU-MFSD, không cần attack taxonomy đầy đủ ở train time.
O3 | Sinh source-only pseudo-unknown spoof theo recipe có mask, artifact map và quality score.
O4 | Học multiple real manifolds theo semantic region và tránh ép vùng sạch của partial attack thành spoof.
O5 | Dùng đúng một trainer core cho PC và Modal; chênh lệch backend không làm đổi preprocessing, split hoặc loss.
O6 | Đánh giá SiW-Mv2 theo target-only protocol, threshold/calibration cố định từ source-dev.
O7 | Xuất run package đủ để tái lập: config, hash, checkpoint, prediction, metric, failure samples.

## Table 7

Research question: Với chỉ hai source datasets chứa chủ yếu print/replay cues, liệu structured source-only attack recipes, regional synthetic supervision và multi-prototype real manifolds có cải thiện generalization trên SiW-Mv2 mà không làm BPCER tăng quá mức hay không?

## Table 8

Dataset | Vai trò | Được phép dùng | Không được phép
CASIA-FASD | Source train + source-dev | Raw video, official metadata, live/spoof label, optional attack/device metadata | Không dùng sample từ SiW để tune CASIA preprocessing
MSU-MFSD | Source train + source-dev | Raw video, official metadata, live/spoof label, optional attack/device metadata | Không trộn frame của cùng video/subject qua train-dev
SiW-Mv2 | Target-only test | Xử lý ảnh/crop local; chạy inference; label chỉ mở ở evaluation step | Không train, không chọn checkpoint, không tune threshold, không tạo recipe từ attack name/image

## Table 9

Cấp | Quy tắc
Video | Đơn vị chống leakage; không split frame từ cùng video.
Frame | Sample huấn luyện; frame index deterministic từ frame sampler config.
Identity | Không bắt buộc cho classification nhưng phải lưu nếu metadata cho phép.
Domain | Batch sampler cân bằng CASIA và MSU ở mức gần 1:1.
Class | Trong mỗi domain ưu tiên 1:1 live/spoof; khi thiếu dùng replacement sampler.
Synthetic | Không vượt 25% batch trong MVP; batch luôn có real live + real spoof.
Video evaluation | Frame scores aggregate bằng trimmed mean 10% hoặc median; mặc định trimmed mean.

## Table 10

Nguyên tắc: Raw data là local-only. Mọi preprocessing output phải có manifest, checksum, version và audit report. Không được copy thủ công các folder crop rồi train trực tiếp.

## Table 11

D:\PRISM-FAS-B\ ├── repo\                         # Git repository ├── data\ │   ├── raw\ │   │   ├── casia_fasd\ │   │   ├── msu_mfsd\ │   │   └── siw_mv2\ │   ├── staging\                 # extracted metadata / temporary indices │   ├── processed\ │   │   └── prism_data_v1\ │   │       ├── images\ │   │       ├── priors\ │   │       ├── manifests\ │   │       ├── shards\ │   │       ├── audit\ │   │       └── PACKAGE_LOCK.json │   └── tmp\                     # safe to delete ├── models\pretrained\ ├── runs\ └── cache\

## Table 12

Adapter | Trách nhiệm | Output tối thiểu
CasiaFasdAdapter | Khám phá video, đọc official split/label, xác định video/subject nếu có | CanonicalVideoRecord
MsuMfsdAdapter | Khám phá video, official split/label, device/attack optional | CanonicalVideoRecord
SiwMv2Adapter | Khám phá target video, tạo feature manifest; labels tách riêng | TargetVideoRecord + eval label file

## Table 13

@dataclass(frozen=True) class CanonicalVideoRecord:     dataset: str     video_id: str     subject_id: str | None     raw_path: str     official_split: str     project_split: Literal["source_train", "source_dev", "target_test"]     label: int | None          # 0 live, 1 spoof; None allowed for blind target inference     attack_family: str | None     capture_device: str | None     presentation_device: str | None     fps: float | None     frame_count: int | None

## Table 14

Config | Mặc định | Ý nghĩa
strategy | uniform_n | Lấy N frame phân bố đều trong vùng hợp lệ của video.
n_frames_train | 32 | Áp dụng cho source_train.
n_frames_dev | 32 | Áp dụng cho source_dev để calibration ổn định.
n_frames_target | 48 | Target test có sampling dày hơn nhưng vẫn cố định trước inference.
skip_head_tail_ratio | 0.05 | Bỏ 5% đầu và cuối để tránh transition/blank frame.
min_frame_gap | 2 | Không chọn frame quá sát nhau khi video ngắn.
seed | 20260804 | Chỉ dùng khi cần tie-break; uniform indices phải deterministic.
decode_backend | opencv | Fallback ffmpeg khi OpenCV decode fail.

## Table 15

Thuộc tính | Mặc định
Detector | SCRFD-2.5G ONNX; batch inference nếu GPU local
Input color | BGR decode -> RGB trước khi lưu
Face selection | Largest valid face; nếu nhiều mặt ghi num_faces và selected_bbox
Padding | 0.25 theo max(width,height), clip vào image boundary
Alignment | Không warp mạnh trong MVP; crop theo padded bbox, landmark chỉ làm prior
Output size | 256x256
Output format | PNG lossless, compression level vừa; không re-encode JPEG
Interpolation | AREA khi downscale, BICUBIC khi upscale; ghi vào manifest
No-face policy | Retry detector threshold thấp hơn một lần; sau đó mark failed
Tiny-face policy | Reject nếu face short side < 64 px trước resize, trừ config override

## Table 16

Lý do lưu PNG 256x256: giữ texture/capture artifact gốc và tránh tạo thêm JPEG shortcut. Model vẫn nhận 224x224 sau train-time resize/crop.

## Table 17

Cache | Kiểu | Bắt buộc MVP | Mục đích
bbox + 5 landmarks | float32 [4], [5,2] | Có | Crop trace và region initialization.
parsing logits/masks | uint8/float16 NPZ | Có nếu FaceXFormer chạy được | Semantic region prior.
pose | float32 yaw/pitch/roll | Có | Audit và optional sampler.
visibility | float16 vector | Khuyến nghị | Mask region bị che/khuất.
identity embedding | float16 [512] | Chỉ live target dùng synthetic gate | Kiểm tra identity drift.
quality metrics | float32 dict | Có | blur, brightness, face size, detection score.
high-pass/wavelet cache | Không | Không | Tính on-the-fly để tránh storage lớn.

## Table 18

Artifact | Nội dung
samples.parquet | Canonical sample rows cho source_train/source_dev/target_test.
source_train.parquet | Chỉ CASIA/MSU train; có label.
source_dev.parquet | Chỉ CASIA/MSU dev; có label.
siw_target_features.parquet | SiW target paths/metadata cần cho inference; label bị loại.
siw_target_labels.parquet | Evaluation-only; không được load bởi train code.
shards/source-00000.tar ... | PNG + NPZ grouped for efficient local/Modal streaming.
audit/data_report.html | Counts, class/domain balance, failures, duplicates, quality plots.
PACKAGE_LOCK.json | Hashes của config, manifests, shards, adapters, model revisions.

## Table 19

# 1) Kiểm tra raw data và official metadata python -m prism.cli data audit-raw --config configs/data/local_windows.yaml  # 2) Build video index và deterministic frame plan python -m prism.cli data index --config configs/data/local_windows.yaml --resume  # 3) Extract frame + detect/crop face python -m prism.cli data crop --config configs/data/local_windows.yaml --workers 8 --device cuda --resume  # 4) Cache parsing/pose/identity/quality python -m prism.cli data priors --config configs/data/local_windows.yaml --device cuda --resume  # 5) Build manifests + tar shards + package lock python -m prism.cli data package --config configs/data/local_windows.yaml  # 6) Validate the immutable package python -m prism.cli data validate-package --package D:\PRISM-FAS-B\data\processed\prism_data_v1

## Table 20

sample_id: string                 # stable SHA1(dataset/video/frame/preprocess_version) dataset: enum[casia_fasd, msu_mfsd, siw_mv2] domain_id: int subject_id: string|null video_id: string frame_id: int timestamp_ms: float|null project_split: enum[source_train, source_dev, target_test] label: int|null                    # 0 live, 1 spoof; null in target feature manifest attack_family: string|null raw_relpath: string                # local audit only; remove in Modal shard index if desired image_relpath: string prior_relpath: string|null bbox_xyxy: list[float] landmarks_5: list[list[float]] detection_score: float num_faces: int face_size_px: float pose_yaw_pitch_roll: list[float]|null quality_blur: float quality_brightness: float quality_contrast: float crop_padding: float resize_interpolation: string preprocessing_version: string raw_sha256: string crop_sha256: string status: enum[ok, failed, excluded] exclude_reason: string|null

## Table 21

sample_id = sha1(     dataset + "|" + video_id + "|" + str(frame_id) + "|" + preprocessing_version ).hexdigest()[:20]

## Table 22

{   "package_id": "prism_data_v1-<12char_hash>",   "created_at": "ISO-8601",   "datasets": {     "casia_fasd": {"raw_fingerprint": "...", "adapter_version": "..."},     "msu_mfsd": {"raw_fingerprint": "...", "adapter_version": "..."},     "siw_mv2": {"raw_fingerprint": "...", "adapter_version": "..."}   },   "preprocess_config_sha256": "...",   "pretrained_model_revisions": {"scrfd": "...", "facexformer": "...", "adaface": "..."},   "manifest_sha256": {"source_train": "...", "source_dev": "...", "target_features": "..."},   "shards_index_sha256": "...",   "status": "validated" }

## Table 23

Process | Được mount | Không được mount
Preprocess local | Raw CASIA, MSU, SiW; metadata | Không hạn chế vì đây là data build, nhưng phải tách label artifact
Train local | source_train, source_dev, recipe/synthetic source package | siw_target_features, siw_target_labels
Train Modal | /data/source + /data/synthetic + /models | /data/target và evaluation labels
Target inference | target_features + checkpoint + frozen calibration | source labels không bắt buộc; target labels không cần
Evaluation | predictions + target_labels | Không update model/checkpoint

## Table 24

# Step A: prediction without target labels python -m prism.cli evaluate predict   --checkpoint runs/<run_id>/checkpoints/best.pt   --manifest data/processed/prism_data_v1/manifests/siw_target_features.parquet   --output runs/<run_id>/predictions/siw_mv2.parquet  # Step B: metrics after prediction file is finalized python -m prism.cli evaluate score   --predictions runs/<run_id>/predictions/siw_mv2.parquet   --labels data/processed/prism_data_v1/evaluation_only/siw_target_labels.parquet   --calibration runs/<run_id>/calibration/source_dev.json

## Table 25

{   "recipe_id": "R-000184",   "medium": {"family": "display-like", "transparency": 0.0, "roughness": 0.25},   "geometry": {"shape": "partial-curved", "rigidity": 0.4, "coverage": 0.22},   "regions": ["right_eye", "upper_right_cheek"],   "artifacts": [     {"name": "specular_reflection", "strength": 0.32},     {"name": "texture_smoothing", "strength": 0.18},     {"name": "boundary_inconsistency", "strength": 0.12}   ],   "capture": {"yaw": 15, "illumination": "side", "compression_q": 82},   "forbidden_shortcuts": ["always_moire", "always_halftone"],   "generator_route": ["physics", "gpat"],   "seed": 7319,   "schema_version": "1.1" }

## Table 26

Axis | Allowed values tối thiểu
Medium | paper-like, display-like, plastic-like, fabric-like, reflective-film-like
Geometry | flat, curved, partial-curved, flexible, rigid, boundary-only
Region | eyes, nose, mouth, cheeks, forehead, face boundary, context, multi-region
Artifact | halftone, pixel grid, moire, reflection, smoothing, color shift, boundary inconsistency, blur
Capture | illumination direction, yaw, scale, compression, motion, defocus
Severity | continuous [0,1] with per-operator safe ranges

## Table 27

Route | Input | Output | MVP
Physics | Live crop + recipe | Synthetic crop + exact mask + operator metadata | Bắt buộc
GPAT residual | Live target + source spoof + recipe | Geometry-preserving spoof + artifact map | Bắt buộc
Masked diffusion | Live + mask + prompt/control | Complex edit + mask | Không dùng main MVP

## Table 28

z_a = E_art(x_spoof, recipe) {Delta_LL, Delta_LH, Delta_HL, Delta_HH, M} = G_res(DWT(x_live), z_a, c_recipe) x_hat = IDWT(LL_live,              LH_live + M * Delta_LH,              HL_live + M * Delta_HL,              HH_live + M * Delta_HH)  # MVP hard-locks low-frequency geometry: Delta_LL is disabled.

## Table 29

Metric | Hard/soft rule | Mục đích
Face detection | score >= tau_fd | Ảnh vẫn là khuôn mặt hợp lệ.
Identity cosine | >= tau_id với live target | Không đổi identity.
Landmark NME | <= tau_lm | Không méo geometry.
Parsing consistency | Dice >= tau_parse ngoài mask | Không sửa topology vùng sạch.
Outside-mask error | <= tau_out | Residual không leak ngoài region.
Artifact strength | a_min <= mean(A) <= a_max | Không quá yếu/quá mạnh.
Fingerprint probe | score <= tau_fp | Không quá dễ nhận bằng generator shortcut.
Recipe match | >= tau_prompt khi prompt head có sẵn | Output phù hợp recipe.

## Table 30

accept_i = all(hard_gate_k(i)) q_i = geometric_mean(normalized_soft_metrics_i)   # q in [0, 1]  # Detector uses q_i as a weight; q_i is not a live/spoof label.

## Table 31

synthetic/bank_v1/ ├── images/<sample_id>.png ├── artifact_maps/<sample_id>.npz ├── masks/<sample_id>.png ├── metadata/<sample_id>.json ├── manifest.parquet ├── rejected.parquet ├── quality_summary.json └── BANK_LOCK.json

## Table 32

F_local = ConvNeXtV2(x_rgb, optional_highpass) T_global, z_global = SigLIP2.image_encoder(x_rgb)  q_r = RegionQuery(parsing_r, landmarks_r, learnable_token_r) z_r = CrossAttention(q_r, K=F_local, V=F_local) + RegionPool(T_global, parsing_r)  p_global = GlobalHead(z_global) d_r = RealManifold.distance(z_r) p_prompt = PromptHead(z_r, frozen_recipe_text_embeddings)

## Table 33

M_r_real = { N(mu_rk, Sigma_rk) } for k = 1..K_r  d_r(x) = min_k (z_r - mu_rk)^T Sigma_rk^{-1} (z_r - mu_rk) assignment_rk = softmax(-d_rk / tau_prototype)

## Table 34

s_region = TopKMean(normalize(d_r), k=2) s_final = 1 - (1 - p_global) * (1 - s_region) * (1 - p_prompt_spoof)  # Unknown/reject also considers entropy and global-local disagreement.

## Table 35

L_cls       = CE(p_global, y_image) L_local     = weighted_BCE(local_token_logits, artifact_map) L_MIL       = CE(LogSumExpMIL(token_logits), y_image) L_real      = mean_live sum_r SoftMin_k d_rk L_out       = mean_syn sum_r m_r * max(0, margin_r - d_r) L_clean     = mean_syn sum_r (1-m_r) * min(d_r, clean_cap) L_prompt    = InfoNCE(z_attack_region, text_embedding(recipe)) L_cons      = |p_global - stopgrad(s_region)| + |s_region - stopgrad(p_global)| L_risk      = Var(domain_risk) + Var(artifact_family_risk)  L_total = L_cls_real + lambda_syn * q * (L_cls_syn + lambda_local*L_local + lambda_out*L_out)         + lambda_M*L_real + lambda_clean*L_clean + lambda_MIL*L_MIL         + lambda_P*L_prompt + lambda_cons*L_cons + lambda_risk*L_risk

## Table 36

Partition | MVP ratio | Rules
Real live | 37.5% | CASIA/MSU domain-balanced; required every batch.
Real spoof | 37.5% | CASIA/MSU domain-balanced; required every batch.
Synthetic spoof | 25% | Quality-weighted; mix physics and GPAT.

## Table 37

Parameter | Default MVP
Optimizer | AdamW
Backbone LR | 1e-5
Heads/manifold LR | 1e-4
Weight decay | 0.05
Scheduler | 5% warm-up + cosine decay
Epochs | 30 detector epochs after warm-up
Warm-up | 3 detector + 2 manifold epochs
Gradient clipping | 1.0
AMP | bf16 when supported, else fp16
EMA | 0.999 optional; report whether enabled
Checkpoint criterion | source_dev ACER primary, BPCER tie-break, then calibration NLL

## Table 38

lambda_syn=0.50 lambda_M=1.00 lambda_out=1.00 lambda_clean=0.25 lambda_local=1.00 lambda_MIL=0.50 lambda_P=0.20 lambda_cons=0.05 lambda_risk=0.10 margin_out=3.0 synthetic_ratio=0.25

## Table 39

Stage | Tên | Input | Output | Backend
G0 | Local data build | Raw datasets | Validated processed package | Laptop only
G1 | Baseline warm-up | Source real live/spoof | Binary detector checkpoint | PC or Modal
G2 | Real manifold init | Source live embeddings | Prototype init + covariance | PC or Modal
G3 | Recipe bank freeze | Source-only ontology | BANK_LOCK | Laptop/CPU
G4 | Synthetic bank | Source train + recipes | Quality-gated bank | PC or Modal
G5 | Mixed training | Real + synthetic | Full PRISM checkpoint | PC or Modal
G6 | Source calibration | Source-dev predictions | thresholds.json | PC or Modal
G7 | Target prediction | SiW target features | prediction parquet | PC or Modal eval
G8 | Final scoring | prediction + eval labels | report | Laptop/isolated eval

## Table 40

PENDING -> RUNNING -> {COMPLETED | FAILED | INTERRUPTED} INTERRUPTED -> RESUMING -> RUNNING FAILED may resume only when failure is marked recoverable.  Each stage writes:   stage_state.json   resolved_config.yaml   input_hashes.json   stdout.log / stderr.log   metrics.jsonl   output_hashes.json

## Table 41

TrainerCore phải chạy được bằng một lệnh Python bình thường và không import modal. modal_app.py chỉ mount Volume, chọn GPU, đặt environment variables và gọi subprocess/entry function vào cùng CLI.

## Table 42

class RuntimeContext(Protocol):     data_root: Path     run_root: Path     pretrained_root: Path     device: str     num_workers: int     backend: Literal["local", "modal"]  class ArtifactStore(Protocol):     def save_checkpoint(...): ...     def append_metrics(...): ...     def commit(...): ...

## Table 43

Item | Contract
OS | Windows 11 hoặc Linux; path resolved bằng pathlib.
GPU | CUDA-capable NVIDIA GPU; fallback CPU chỉ cho smoke tests.
Data | Loose files hoặc shards trong local processed package.
Runs | D:\PRISM-FAS-B\runs hoặc PRISM_RUN_ROOT.
Resume | --resume auto tìm last.pt trong run_id hiện tại.
Monitoring | TensorBoard + metrics.jsonl + console logs.
Failure | Checkpoint atomic write: write temp -> fsync -> rename.

## Table 44

Resource | Đề xuất
App | prism-fas-b
Data Volume | prism-fas-data; chỉ processed source/target packages và synthetic bank
Run Volume | prism-fas-runs; checkpoints, logs, predictions, reports
Model Volume | prism-fas-models; pinned pretrained weights
Mounts | /data, /runs, /models
GPU default | L40S; fallback list có thể L40S -> A100-40GB tùy config
Timeout | Theo stage; long training bật detach và checkpoint định kỳ
Commit | Commit Volume sau checkpoint/report quan trọng
Image | Pinned Python + torch + project dependencies; source mounted last

## Table 45

# Create volumes once modal volume create prism-fas-data modal volume create prism-fas-runs modal volume create prism-fas-models  # Upload immutable processed package from Windows/local shell modal volume put prism-fas-data D:\PRISM-FAS-B\data\processed\prism_data_v1 /packages/prism_data_v1/  # Inspect and download artifacts modal volume ls prism-fas-runs /runs/ modal volume get prism-fas-runs /runs/<run_id>/ D:\PRISM-FAS-B\runs\<run_id>

## Table 46

# modal_app.py: wrapper only, simplified contract import modal  app = modal.App("prism-fas-b") data_vol = modal.Volume.from_name("prism-fas-data") runs_vol = modal.Volume.from_name("prism-fas-runs") models_vol = modal.Volume.from_name("prism-fas-models")  @app.function(     gpu="L40S",     volumes={"/data": data_vol, "/runs": runs_vol, "/models": models_vol},     timeout=24 * 60 * 60, ) def train_remote(config_name: str, run_id: str, resume: bool = True):     # call the same prism.cli train entrypoint used locally     ...  @app.local_entrypoint() def main(config_name: str, run_id: str, gpu: str = "L40S", resume: bool = True):     train_remote.with_options(gpu=gpu).remote(config_name, run_id, resume)

## Table 47

configs/ ├── data/ │   ├── local_windows.yaml │   ├── package_v1.yaml │   └── modal_package_v1.yaml ├── model/ │   ├── baseline_cnn.yaml │   ├── baseline_cnn_vit.yaml │   └── prism_b_region_siglip.yaml ├── train/ │   ├── warmup.yaml │   ├── synthetic_bank.yaml │   └── full.yaml ├── runtime/ │   ├── local_gpu.yaml │   └── modal_l40s.yaml ├── experiment/ │   ├── b00_binary.yaml │   ├── b10_manifold.yaml │   └── b20_full_prism.yaml └── config.yaml

## Table 48

Logical key | Local example | Modal example
data.package_root | D:/PRISM-FAS-B/data/processed/prism_data_v1 | /data/packages/prism_data_v1
data.source_train_manifest | ${package_root}/manifests/source_train.parquet | Same logical relative path
runs.root | D:/PRISM-FAS-B/runs | /runs/runs
models.root | D:/PRISM-FAS-B/models/pretrained | /models/pretrained
cache.root | D:/PRISM-FAS-B/cache | /tmp/prism-cache

## Table 49

# configs/runtime/local_gpu.yaml backend: local device: cuda num_workers: 8 persistent_workers: true pin_memory: true amp: fp16 paths:   package_root: ${oc.env:PRISM_DATA_ROOT}/prism_data_v1   run_root: ${oc.env:PRISM_RUN_ROOT}   pretrained_root: ${oc.env:PRISM_MODEL_ROOT}  # configs/runtime/modal_l40s.yaml backend: modal device: cuda num_workers: 12 persistent_workers: true pin_memory: true amp: bf16 paths:   package_root: /data/packages/prism_data_v1   run_root: /runs/runs   pretrained_root: /models/pretrained

## Table 50

# Local PC python -m prism.cli train experiment=b20_full_prism runtime=local_gpu run.id=prism_b_seed1 seed=1  # Modal modal run --detach modal_app.py --config-name b20_full_prism --run-id prism_b_seed1 --gpu L40S  # Resume python -m prism.cli train experiment=b20_full_prism runtime=local_gpu run.id=prism_b_seed1 resume=true modal run --detach modal_app.py --config-name b20_full_prism --run-id prism_b_seed1 --gpu L40S --resume true  # Dry run / smoke test python -m prism.cli train experiment=b20_full_prism runtime=local_gpu trainer.max_steps=5 data.limit_batches=2

## Table 51

prism_fas_b/ ├── pyproject.toml ├── README.md ├── modal_app.py ├── configs/ ├── src/prism/ │   ├── cli.py │   ├── config_schema.py │   ├── runtime/{context.py, local.py, artifacts.py} │   ├── data/ │   │   ├── adapters/{casia_fasd.py, msu_mfsd.py, siw_mv2.py} │   │   ├── inventory.py │   │   ├── frame_sampling.py │   │   ├── face_crop.py │   │   ├── priors.py │   │   ├── package.py │   │   ├── validation.py │   │   ├── dataset.py │   │   └── samplers.py │   ├── recipes/{schema.py, generate.py, validate.py, compile.py, bank.py} │   ├── synthesis/{physics.py, gpat.py, pairs.py, quality.py, bank.py} │   ├── models/{local_cnn.py, global_vlm.py, region_fusion.py, manifold.py, heads.py, prism_b.py} │   ├── losses/{classification.py, local.py, manifold.py, prompt.py, risk.py, total.py} │   ├── train/{engine.py, stages.py, checkpoint.py, callbacks.py, seed.py} │   ├── evaluate/{predict.py, aggregate.py, metrics.py, calibration.py, report.py} │   └── utils/{hashing.py, io.py, logging.py, distributed.py} ├── tests/ │   ├── unit/ │   ├── integration/ │   ├── fixtures/ │   └── smoke/ └── scripts/{bootstrap_windows.ps1, sync_modal.ps1, export_run.py}

## Table 52

{   "image": FloatTensor[3, H, W],   "label": LongTensor[] | None,   "dataset_id": LongTensor[],   "sample_id": str,   "video_id": str,   "region_priors": FloatTensor[R, Ht, Wt] | None,   "visibility": FloatTensor[R] | None,   "artifact_map": FloatTensor[1, Ht, Wt] | None,   "attack_region_mask": FloatTensor[R] | None,   "quality_weight": FloatTensor[] | None,   "recipe_id": str | None,   "is_synthetic": bool }

## Table 53

ModelOutput(     global_logit: Tensor[B, 1],     local_logits: Tensor[B, P],     region_embeddings: Tensor[B, R, D],     region_distances: Tensor[B, R],     prompt_logits: Tensor[B, N_prompt] | None,     confidence_features: dict[str, Tensor],     aux: dict[str, Tensor], )

## Table 54

Module | Không được làm
dataset adapters | Không train model, không biết Modal, không random split frame.
preprocessing | Không đọc train hyperparameters hoặc target attack taxonomy.
model | Không đọc file paths trực tiếp; nhận tensor batch.
trainer | Không biết raw dataset layout; chỉ đọc canonical manifest/package.
Modal wrapper | Không chứa loss/model logic hoặc duplicate train loop.
evaluator | Không update model, optimizer hoặc calibration từ target labels.
reporter | Không recompute predictions; chỉ đọc frozen artifacts.

## Table 55

runs/<run_id>/ ├── run.json ├── resolved_config.yaml ├── environment.txt ├── git_state.json ├── data_lock.json ├── logs/{train.log, metrics.jsonl} ├── checkpoints/{last.pt, best.pt, epoch_*.pt} ├── calibration/source_dev.json ├── predictions/{source_dev.parquet, siw_mv2.parquet} ├── reports/{report.html, summary.json, plots/} ├── failures/{train_failures.parquet, eval_failures.parquet} └── COMPLETE.json

## Table 56

Test | Pass criterion
5-step smoke local vs Modal | Loss finite, same batch IDs for same seed/config where nondeterminism disabled.
Single-batch forward | Output shapes and value tolerance documented.
Checkpoint migration | Local save -> Modal load and Modal save -> local load.
Manifest/shard reading | Same sample count and sample_id set.
Metric implementation | Synthetic fixture returns known APCER/BPCER/ACER values.

## Table 57

# frame prediction parquet columns sample_id, video_id, frame_id, p_global, s_region, p_prompt, s_final, confidence, decision, top_region_ids, region_distances, checkpoint_hash, calibration_hash, inference_config_hash  # video aggregation video_score = trimmed_mean(frame_s_final, trim=0.10) video_confidence = median(frame_confidence) video_decision = threshold_and_reject(video_score, video_confidence)

## Table 58

Nhóm | Metrics
FAS core | APCER, BPCER, ACER, HTER, ROC-AUC, EER; frame và video level.
Attack-wise | APCER theo attack group sau prediction; không dùng trong tuning.
Open-set/reject | AUROC/AUPR unknown nếu protocol labels hỗ trợ, FPR@95TPR, risk-coverage, rejection rate.
Calibration | ECE, Brier score, NLL.
Synthetic quality | Identity cosine, landmark NME, parsing Dice, outside-mask error, acceptance rate.
Efficiency | Params, FLOPs estimate, latency, peak VRAM, data throughput; tách offline/online cost.
Statistics | Mean +/- std qua tối thiểu 3 seeds; paired bootstrap theo video khi so model.

## Table 59

ID | Baseline
B00 | ConvNeXt binary classifier; domain-balanced source training.
B01 | SigLIP/ViT binary classifier.
B02 | CNN + ViT simple concat, no region/manifold/prompt.
B03 | B02 + physics augmentation only.
B04 | B02 + GPAT synthetic bank, no recipe/manifold.
B05 | B02 + one global real center/Gaussian.
B06 | Regional detector + global center.
B07 | Regional detector + multi-prototype manifolds, no synthetic.
B08 | Full PRISM-FAS-B v1.1.

## Table 60

Ablation | Variants | Question
Data balance | naive concat vs domain/class balanced | Gain có phải do sampler?
Recipe | random operators vs structured recipe | Recipe composition có giá trị không?
Synthetic route | physics, GPAT, physics+GPAT | Route nào tạo gain?
Quality | hard gate only vs q weighting | Soft reliability có ích không?
Region | global vs semantic regions | Local anomaly có cần region prior?
Prototype K | 1,2,4,6 | Multiple real modes có cần thiết?
Outlier | image-level vs mask-aware | Vùng sạch có được bảo toàn?
Prompt | off, frozen prompt, adapter | Prompt gain độc lập ra sao?
Backend | PC vs Modal same seed/config | Infrastructure có làm lệch kết quả?
Frame count | 16,32,48/64 | Sampling density ảnh hưởng thế nào?

## Table 61

Codex execution rule: triển khai theo từng milestone nhỏ, mỗi milestone phải có tests và lệnh chạy. Không viết toàn bộ model phức tạp trước khi data package + baseline + run registry chạy end-to-end.

## Table 62

Mốc | Nội dung | Acceptance criteria
M0 | Repo skeleton, config schema, logging, hashing | pytest pass; CLI --help; resolved config saved.
M1 | Dataset adapters + raw audit | Mini fixtures parsed; missing metadata fails clearly.
M2 | Frame extraction + SCRFD crop + manifest | Resume-safe; deterministic sample IDs; audit report.
M3 | Priors + package/shards + lock | Package validate pass; target labels isolated.
M4 | Canonical dataset loader + balanced sampler | Batch contract verified by tests.
M5 | B00 baseline local training/evaluation | Checkpoint, source calibration, target prediction/report.
M6 | Modal wrapper + data sync + parity smoke | Same trainer core; checkpoint migration pass.
M7 | Recipe schema/compiler + physics engine | Frozen bank; exact masks; deterministic operators.
M8 | GPAT + quality gate + synthetic bank | Bank lock, q score, rejected manifest.
M9 | Regional fusion + manifolds + losses | Toy tests and source-dev training stable.
M10 | Full experiment matrix + report | 3 seeds, baselines/ablations, reproducible summary.

## Table 63

Implement Milestone M0 and M1 only.  Requirements: 1. Create the repository structure and pyproject.toml. 2. Implement strict Pydantic configuration models and a Typer CLI. 3. Implement hashing, atomic JSON/YAML writes, structured logging and run metadata. 4. Implement CanonicalVideoRecord and dataset adapter interfaces. 5. Implement CASIA-FASD/MSU-MFSD/SiW-Mv2 adapters using explicit YAML layout rules; do not guess unknown layouts. 6. Implement raw audit and miniature fixtures. 7. Add unit tests and one integration test. 8. Do not implement training/model code yet. 9. Return a file-by-file summary, commands run and test results.

## Table 64

version: 1 preprocessing_version: prism-preprocess-1.1.0  paths:   raw_root: ${oc.env:PRISM_RAW_ROOT}   output_root: ${oc.env:PRISM_DATA_ROOT}/prism_data_v1   temp_root: ${oc.env:PRISM_TMP_ROOT}  sources:   casia_fasd:     root: ${paths.raw_root}/casia_fasd     adapter: casia_fasd     layout_rules: configs/datasets/casia_fasd_layout.yaml     split_mapping: {train: source_train, test: source_dev}   msu_mfsd:     root: ${paths.raw_root}/msu_mfsd     adapter: msu_mfsd     layout_rules: configs/datasets/msu_mfsd_layout.yaml     split_mapping: {train: source_train, test: source_dev}   siw_mv2:     root: ${paths.raw_root}/siw_mv2     adapter: siw_mv2     layout_rules: configs/datasets/siw_mv2_layout.yaml     split_mapping: {test: target_test}     isolate_labels: true  frames:   strategy: uniform_n   n_frames_train: 32   n_frames_dev: 32   n_frames_target: 48   skip_head_tail_ratio: 0.05   seed: 20260804  crop:   detector: scrfd_2.5g   detector_threshold: 0.50   retry_threshold: 0.35   padding: 0.25   output_size: 256   output_format: png   min_face_short_side: 64  package:   shard_size_samples: 1000   tar_compression: none   build_package_lock: true   require_zero_split_leakage: true

## Table 65

defaults:   - /data: package_v1   - /model: prism_b_region_siglip   - /train: full   - /runtime: local_gpu   - _self_  experiment_name: b20_full_prism seed: 1 run:   id: ${experiment_name}_seed${seed}   resume: true  data:   allowed_datasets: [casia_fasd, msu_mfsd]   forbidden_splits: [target_test]   batch:     live: 12     real_spoof: 12     synthetic_spoof: 8   domain_balance: true  model:   input_size: 224   region_count: 9   prototype_k: 4   covariance: diagonal  trainer:   max_epochs: 30   validate_every_epochs: 1   checkpoint_every_steps: 1000   selection_metric: source_dev/acer   tie_break_metric: source_dev/bpcer

## Table 66

# PowerShell conda create -n prism-preprocess python=3.11 -y conda activate prism-preprocess pip install -e .[preprocess]  $env:PRISM_RAW_ROOT   = "D:\PRISM-FAS-B\data\raw" $env:PRISM_DATA_ROOT  = "D:\PRISM-FAS-B\data\processed" $env:PRISM_TMP_ROOT   = "D:\PRISM-FAS-B\data\tmp" $env:PRISM_RUN_ROOT   = "D:\PRISM-FAS-B\runs" $env:PRISM_MODEL_ROOT = "D:\PRISM-FAS-B\models\pretrained"  python -m prism.cli doctor python -m prism.cli data audit-raw --config configs/data/local_windows.yaml

## Table 67

# Core/train python>=3.11 pytorch, torchvision transformers, timm, accelerate, peft, safetensors hydra-core, omegaconf, pydantic, typer numpy, pandas, pyarrow, scikit-learn opencv-python, pillow, albumentations pywavelets, einops, tensorboard  # Preprocess local insightface, onnxruntime-gpu ffmpeg/ffprobe available on PATH scikit-image optional FaceXFormer environment pinned separately  # Modal wrapper modal  # Development pytest, pytest-xdist, ruff, mypy

## Table 68

# Authenticate/install Modal according to the active workspace. # Create persistent volumes. modal volume create prism-fas-data modal volume create prism-fas-runs modal volume create prism-fas-models  # Upload processed data and model weights. modal volume put prism-fas-data <LOCAL_PACKAGE_PATH> /packages/prism_data_v1/ modal volume put prism-fas-models <LOCAL_MODEL_DIR> /pretrained/  # Detached long training. modal run --detach modal_app.py --config-name b20_full_prism --run-id prism_b_seed1 --gpu L40S  # Inspect/download results. modal volume ls prism-fas-runs /runs/prism_b_seed1/ modal volume get prism-fas-runs /runs/prism_b_seed1/ <LOCAL_RUN_DEST>

## Table 69

Field | Required
component | scrfd / facexformer / adaface / convnext / siglip / gpat
model_id | Official repository/model card identifier
revision | Git commit, tag or model revision
file_sha256 | Hash of exact local weight file
license | Recorded license/use restriction
downloaded_at | Timestamp
local_relpath | Path relative to PRISM_MODEL_ROOT
input_contract | Color, size, normalization
output_contract | Shape and semantic meaning
notes | Known dependency or precision constraints

## Table 70

Final implementation principle: dữ liệu được chuẩn hóa một lần trên laptop, đóng băng thành package; huấn luyện có thể đổi nơi chạy nhưng không đổi dữ liệu, code, config hoặc tiêu chí khoa học. Đây là điểm “xương sống” để Codex không tạo hai hệ thống local và cloud lệch nhau.
