# Prepared data accepted by this release

The training package starts from **pre-extracted CONCH-v1 tile embeddings**.
Raw-slide tiling, tissue segmentation and CONCH extraction use a separate
preprocessing environment. See the DSMIL and TRIDENT workflow overview in
[README.md](../README.md#data). Readers prepare these features with their chosen
pipeline, or download the prepared features and annotations from
[WinstonHu/LoC-Path-Data](https://huggingface.co/datasets/WinstonHu/LoC-Path-Data).
Preprocessing tools and model weights are not included in this package. The data
release contains no raw images or original pathology reports. Its WSI-Bench task
partitions are inherited from WSI-LLaVA/WSI-Bench through the parent repository.

## Features and coordinates

Training uses this layout (the cancer code is lowercase on disk):

```text
features/conch_v1/
  lusc_224x224_b20_t15/
    npy_files/
      TCGA-EXAMPLE_features.npy
      TCGA-EXAMPLE_coords.npy
    h5_files/                         # default benchmark inference format
      TCGA-EXAMPLE.h5
```

- Features: numeric array `[N, 512]`, `float16` or `float32` on disk. N is the
  number of tiles. The prepared release retains the existing NPY precision.
- Coordinates: matching numeric array `[N, 2]`, `float32`, in `(x, y)` order.
  Training NPY coordinates are tile-grid indices, zero-anchored per slide, rather
  than raw pixel positions. Preserve the feature/coordinate row correspondence.
- H5 inference files contain datasets `features` and `coords`. The default
  inference loader snaps CONCH pixel coordinates to a tile grid; already gridded
  coordinates also work. For the NPY-only Hugging Face release, pass
  `--image_feature_suffix .npy` at inference. Do not remove coordinates: STM needs
  spatial positions.
- The `_224x224_b20_t15` directory suffix is a lookup convention in the released
  configs, not a command to extract features. If changing your directory layout,
  also update `identifier` in the dataset config and `--identifier` for inference.
- MAE pretraining scans `*_features.npy` recursively and selects slides listed in the
  report JSON. Stage 1/2 resolve each JSON image stub into its cancer directory.
  The original extension in the JSON stub does not select the training backend.
- The stage configs cap sampled visual features at 61,440 tiles using `linspace`.
  `per_image_length=256` is the text pipeline's image-token reservation; the
  explicit `sampler_cfg.max_tokens` controls visual feature sampling.

The parent repository's conversion tool,
`tools/convert_conch_features_to_npy.py`, creates these NPY pairs from H5/PT
features, casts features to fp16 and converts pixel coordinates to grid indices.
It is not bundled here. Use the above contract for your prepared features. The smoke test generates both NPY
and H5 examples locally, without patient data.

## Training JSON

Stage 1 (reports) and Stage 2 (instruction/VQA) both use a JSON **array** of
LLaVA-style records; the intended supervision differs between stages:

```json
[
  {
    "id": "TCGA-EXAMPLE",
    "image": ["./LUSC/TCGA-EXAMPLE.h5"],
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe the slide."},
      {"from": "gpt", "value": "The reference report or answer."}
    ]
  }
]
```

The dataset resolves the example to
`$LOCPATH_FEATURE_ROOT/lusc_224x224_b20_t15/npy_files/TCGA-EXAMPLE_features.npy`
and the corresponding `_coords.npy` file. Provide one slide per record. Include
`<image>` in the human turn and nonempty assistant supervision. When constructing
a new dataset, keep all records for a slide/patient within the same
train/validation/test partition. The published WSI-Bench annotations retain the
inherited task partitions described above.

`LOCPATH_STAGE1_REPORT_JSON` selects the MAE slide set; it can use the same report
JSON as `LOCPATH_STAGE1_DATA_PATH`. It does not supervise MAE with report text.
`LOCPATH_UNWANTED_PREFIX_CSV` is optional: one unwanted slide prefix per line.

## Benchmark JSON

Place held-out records in files named for the task, such as
`diagnosis_choice_test.json` or `report_test.json`. They use the same record
schema. The inference command scans only `*_test.json` files. Keep the task file
names: evaluation uses them to decide between closed-ended accuracy,
open-ended judging and report-generation metrics. Generated prediction JSONL
contains `file`, `id`, `ref_answer` and `output` fields for the evaluator.
