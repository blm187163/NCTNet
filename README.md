# NCTNet

NCTNet is a fixed-block, two-speaker speech-separation model for embedded NPU
deployment. This release contains the NCTNet configuration used in the experiments:
LayerNorm-C and GCAv2, with a 2-s input block at 16 kHz.

## Repository Contents

```text
NCTNet/
├── README.md                         # Repository overview
├── NCTNet_ONNX_ATC_DEPLOYMENT.md     # NCTNet ONNX export and ATC compilation guide
├── LICENSE                           # MIT license for repository source code
├── MODEL_LICENSE.md                  # CC BY 4.0 license for converted NCTNet models
├── DATA_NOTICE.md                    # LibriSpeech attribution for example audio
├── Speech/                           # A 2-s mixture, references, and estimates
├── board-side inference code/        # SS928 ACL and RK3588 RKNN inference sources
├── model-atc-logs/                   # Selected ONNX and ATC diagnostic logs
└── models/
    ├── NCTNet.om                     # Compiled Ascend OM artifact
    └── NCTNet.rknn                   # Compiled Rockchip RKNN artifact
```

## NCTNet Configuration

| Item | Value |
|---|---|
| Model class | `NCTNet` |
| Input / output | `[1, 1, 1, 32000]` -> `[1, 2, 32000]` |
| Sampling rate / block length | 16 kHz / 2 s |
| Sources | 2 |
| Encoder width / bottleneck width | 256 / 128 |
| Separator blocks | 8 |
| Normalization | LayerNorm-C (`eps=1e-4`) |
| Context module | GCAv2 (reduction ratio 4) |
| Convolution kernel / FFN expansion | 9 / 4 |
| Dropout / hard clamp | 0.0 / disabled |
| ONNX export | Opset 11, static shape, constant folding enabled |

## Audio Example

`Speech/` provides one 2-s, 16-kHz, two-speaker WAV example.

| Directory | File(s) | Description |
|---|---|---|
| `Speech/mixed speech/` | `mix.wav` | Mixture input |
| `Speech/clean speech/` | `s1.wav`, `s2.wav` | Reference sources |
| `Speech/separated speech/` | `estimate_s1.wav`, `estimate_s2.wav` | NCTNet estimates |

## ONNX Conversion Record

The public repository provides the converted `NCTNet.om` and `NCTNet.rknn`
artifacts only. The training source code, checkpoint, model-definition code,
and intermediate ONNX file are not included. The recorded ONNX conversion
parameters, operator mapping, and ATC command are documented in
[`NCTNet_ONNX_ATC_DEPLOYMENT.md`](NCTNet_ONNX_ATC_DEPLOYMENT.md).

For questions about the training implementation, contact the authors at
[`blm187163@163.com`](mailto:blm187163@163.com).

## Deployment Artifacts

| File | Target runtime | Input constraint |
|---|---|---|
| `models/NCTNet.om` | Ascend NPU runtime | One `[1, 1, 1, 32000]` float32 block |
| `models/NCTNet.rknn` | Rockchip RKNN runtime | Refer to the board-side inference implementation |

The released model is non-causal fixed-block inference. Audio longer than 2 s
must be split into 2-s blocks. Zero-pad a final shorter block to 32,000 samples
before inference, then crop each output source back to the original length.

## Board-side Inference

`board-side inference code/` contains the SS928 ACL evaluator
(`ss928_npu_sep_eval.c`) and the RK3588 RKNN batch evaluator
(`npu_sep_batch_eval_rk3588.c`). Both programs target the released NCTNet
artifacts and require compilation with the corresponding board SDK. Their
interfaces, fixed 32,000-sample input assumptions, and metric limitations are
documented in
[`NCTNet_ONNX_ATC_DEPLOYMENT.md`](NCTNet_ONNX_ATC_DEPLOYMENT.md).

## Selected Deployment Diagnostics

Selected diagnostic logs are retained in `model-atc-logs/`. Personal
home-directory components, where present, are redacted as `/home/user/`; the
recorded diagnostics are otherwise unchanged. The following is a strict summary
of the recorded messages; it separates model failures from toolchain setup and
command-configuration failures.

| Model / log | Stage | Recorded diagnostic | Scope of conclusion |
|---|---|---|---|
| DPRNN (`DPRNN_export_onnx.log`) | ONNX export | `aten::col2im` is unsupported by ONNX opset 13; the exporter reports support beginning with opset 18 | Export failed for the recorded opset-13 configuration |
| Lite-ConvTasNet | ATC compilation | `ConvTranspose_459` failed during `InferShapePass` | ATC ConvTranspose shape-inference failure |
| SuDoRM-RF++ | ATC compilation | ATC aborted with an internal protobuf bounds-check failure | The log does not identify a specific ONNX operator |
| MossFormer | ATC initialization | `ModuleNotFoundError: No module named 'te_fusion'` | Toolchain initialization issue, not proof of a model-level incompatibility |
| Conv-TasNet | ATC compilation | Dynamic input `audio_input=-1` requires `--input_shape` | The recorded command lacks a required static input shape |
| Conformer-TasNet (`Conformer-TasNet_L16000_tsnsc_atc.log`) | ATC compilation | TBE/Python error `cannot import name 'Iterable' from 'collections'`, followed by unsupported `InstanceNorm`, `FullyConnection`, `Conv2DTransposeD`, `StridedSliceD`, `GatherV2`, and `ConcatD` diagnostics; the log ends with `build graph failed` | Failed compilation in the recorded environment; not evidence of one isolated operator failure |
| TDANet (`TDANet_L32000_tsnsc_atc.log`) | ATC graph parsing | `Param:src_out_anchor is nullptr`; `Model parse to graph failed` and `ATC model parse ret fail` | Internal graph-parser failure; no OM file was generated |

## Citation and License

Please cite the accompanying NCTNet paper when using this repository.

The repository source code is released under the [MIT License](LICENSE).
The converted NCTNet model artifacts are released under
[CC BY 4.0](MODEL_LICENSE.md). The example audio is derived from LibriSpeech;
its required attribution and redistribution conditions are given in
[DATA_NOTICE.md](DATA_NOTICE.md).

Third-party runtimes and SDKs, including Ascend ACL and RKNN, are not
redistributed by this repository and remain subject to their respective terms.
