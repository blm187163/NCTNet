# NCTNet: ONNX Export and ATC Compilation

This document records the deployment artifacts and conversion settings for the
released NCTNet model. The public repository distributes converted deployment
models only; the training source code, checkpoint, model-definition code, and
intermediate ONNX file are not included. It does not claim that one ATC command
works unchanged across all Ascend toolkits.

For questions about the training implementation, contact the authors at
`blm187163@163.com`.

The recorded deployment environment, including known versions and artifact
checksums, is listed in [`ENVIRONMENT.md`](ENVIRONMENT.md).

## 1. Included Files

| Path | Purpose |
|---|---|
| `models/NCTNet.om` | Compiled deployment artifact included for reference |
| `models/NCTNet.rknn` | Rockchip deployment artifact included for reference |

## 2. NCTNet Network Configuration

| Setting | Value |
|---|---|
| Model class | `NCTNet` |
| Sampling rate | 16 kHz |
| Input duration | 2 s |
| Input tensor | `input`, `float32`, `[1, 1, 1, 32000]` |
| Output tensor | `output`, `float32`, `[1, 2, 32000]` |
| Sources | 2 |
| Encoder width (`N`) | 256 |
| Bottleneck width | 128 |
| Separator blocks | 8 |
| Normalization | LayerNorm-C, `eps=1e-4` |
| Attention/context module | GCAv2, reduction ratio 4 |
| Convolution-module kernel | 9 |
| Feed-forward expansion | 4 |
| Dropout | 0.0 |
| Hard activation clamp | Disabled |
| Range-regularization weight | 0.0 |

The release is a static, non-causal, fixed-block model. It accepts exactly one
2-s block. Pad a final incomplete input block to 32,000 samples before
inference and crop each estimated source to the original length afterward.

## 3. Recorded ONNX Conversion Configuration

The following settings describe the recorded PyTorch-to-ONNX conversion. They
are provided for deployment traceability; the exporter and intermediate ONNX
file are not part of this release.

| Setting | Recorded value |
|---|---|
| ONNX opset | 11 |
| Input name, type, and shape | `input`, `float32`, `[1, 1, 1, 32000]` |
| Output name, type, and shape | `output`, `float32`, `[1, 2, 32000]` |
| Dynamic axes | Disabled (static shape) |
| Constant folding | Enabled |

The fixed 32,000-sample input corresponds to one 2-s block at 16 kHz.

## 4. PyTorch-to-ONNX Operator Mapping

This table describes source components and their expected exported ONNX
operators. The ATC compiler can fuse neighboring nodes, so it is not a list of
one-to-one final NPU kernels.

| Source component | ONNX operator(s) |
|---|---|
| Encoder, bottleneck, feed-forward layers, GCAv2, mask projections | `Conv` |
| Depthwise temporal convolution | `Conv` with groups |
| Decoder | `ConvTranspose` |
| LayerNorm-C | `Transpose`, `ReduceMean`, `Sub`, `Mul`/`Pow`, `Add`, `Sqrt`, `Div`, `Mul`, `Add`, `Transpose` |
| Global-context pooling | `GlobalAveragePool` |
| GCAv2 gating | `Relu`, `Sigmoid`, `Mul`, `Add` |
| Tensor reshaping, mask assembly, output cropping | `Shape`, `Gather`, `Unsqueeze`, `Concat`, `Reshape`, `Slice` |

## 5. ATC Compilation

### 5.1 Toolchain Environment

The archived compilation environment used Ascend Toolkit
`5.20.t6.2.b060`. The accepted `--soc_version` values are toolkit-dependent.
Before compiling, check the exact local values instead of reusing a target
identifier from another toolkit:

```bash
export TOOLKIT=/usr/local/Ascend/ascend-toolkit/5.20.t6.2.b060
export ATC_BIN="$TOOLKIT/x86_64-linux/bin/atc"

"$ATC_BIN" --help | grep -i soc_version
```

For the archived toolkit, `TsnsC` is the identifier used in the command below.
If the installed toolkit lists a different accepted target, replace only the
value of `SOC_VERSION` with that listed value.

```bash
export COMPAT="$HOME/ascend_py310_compat"
unset PYTHONHOME PYTHONPATH

export PATH="$TOOLKIT/x86_64-linux/bin:$TOOLKIT/x86_64-linux/atc/bin:$PATH"
export LD_LIBRARY_PATH="$TOOLKIT/x86_64-linux/devlib:$TOOLKIT/x86_64-linux/runtime/lib64:$TOOLKIT/x86_64-linux/atc/lib64:$TOOLKIT/x86_64-linux/compiler/lib64:$TOOLKIT/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COMPAT:$TOOLKIT/x86_64-linux/compiler/python/site-packages:$TOOLKIT/x86_64-linux/python/site-packages:$TOOLKIT/x86_64-linux/opp/op_impl/built-in/ai_core/tbe"
export ASCEND_OPP_PATH="$TOOLKIT/x86_64-linux/opp"
export SOC_VERSION=TsnsC

test -x "$ATC_BIN"
```

### 5.2 ATC Command

The following is the recorded ATC command used for the released NCTNet OM
artifact. `NCTNet_L32000_op11.onnx` is an intermediate conversion file and is
not distributed in this public repository:

```bash
mkdir -p om_output

"$ATC_BIN" \
  --framework=5 \
  --model=NCTNet_L32000_op11.onnx \
  --output=om_output/NCTNet_L32000_TsnsC \
  --input_format=NCHW \
  --input_shape="input:1,1,1,32000" \
  --soc_version="$SOC_VERSION" \
  2>&1 | tee om_output/NCTNet_L32000_TsnsC_atc.log

grep -E "ATC run success|ATC run failed|ERROR|WARN" om_output/NCTNet_L32000_TsnsC_atc.log
ls -lh om_output/NCTNet_L32000_TsnsC.om
```

The successful completion marker is:

```text
ATC run success, welcome to the next use.
```

Do not overwrite the released `NCTNet.om` when trying a different toolkit,
target, optimization option, or exported graph. Preserve the ONNX file, ATC
log, toolkit version, and `--soc_version` value with each generated OM artifact.

## 6. Board-side NCTNet Inference

The directory `board-side inference code/` contains two NCTNet evaluation
programs. They are runtime-specific and are not additional model variants:

| Platform | Source file | Runtime | Model artifact | Input/output |
|---|---|---|---|---|
| SS928 | `ss928_npu_sep_eval.c` | Ascend ACL | `models/NCTNet.om` | 16-kHz PCM, one 32,000-sample block; two output sources |
| RK3588 | `npu_sep_batch_eval_rk3588.c` | Rockchip RKNN | `models/NCTNet.rknn` | 16-kHz signed PCM16, exactly 32,000 samples; two output sources |

The source files must be built with the SDK and runtime libraries installed on
the target board. The exact include and library paths are board-image
dependent; the commands below show the program interfaces after compilation.

### 6.1 SS928 ACL evaluator

The SS928 program expects matching file names under `mix/`, `s1/`, and `s2/`:

```text
test_dir/
├── mix/  # mixture PCM files
├── s1/   # reference source 1, same file names as mix/
└── s2/   # reference source 2, same file names as mix/
```

Run one NCTNet OM model as follows:

```bash
./npu_sep_eval models/NCTNet.om test_dir \
  --pcm_dtype s16 \
  --input_len 32000 \
  --sample_rate 16000 \
  --out_csv ss928_nctnet_metrics.csv \
  --device_id 0 \
  --out_layout planar
```

The evaluator reports SI-SDR, SI-SDRi, SDR, inference time, and optional
PESQ/STOI columns. If `--pesq_cmd` or `--stoi_cmd` is not supplied, the
program uses its built-in proxy metrics; these proxies are not the official
ITU-T P.862 PESQ or a standards-compliant STOI implementation. Use external
metric commands when reporting standard PESQ/STOI values.

The model is non-causal and fixed-shape. For an utterance longer than 2 s,
split it into 32,000-sample blocks, zero-pad only the final block, run each
block independently, and crop the concatenated estimates to the original
length.

### 6.2 RK3588 RKNN evaluator

The RK3588 program reads one 32,000-sample signed PCM16 file at a time from
each directory and writes separated PCM16 files plus a metrics CSV:

```bash
./npu_sep_batch_eval_rk3588 \
  models/NCTNet.rknn \
  test_dir/mix test_dir/s1 test_dir/s2 \
  rk3588_nctnet_output
```

The program converts PCM16 input to float32, submits an NCHW tensor to RKNN,
expects two output channels of 32,000 samples, and records per-file NPU time,
RTF, SI-SDR, and SI-SDRi in `rk3588_nctnet_output/metrics.csv`. The source
does not implement padding or arbitrary-length chunking; prepare fixed-length
32,000-sample inputs before invoking it.

## 7. Archived Baseline Diagnostics

The selected logs in `model-atc-logs/` document failed compilation attempts for
other models. Personal home-directory components, where present, are redacted
as `/home/user/`; the recorded diagnostics are otherwise unchanged. These
entries are diagnostic records for the tested toolchain, not NCTNet commands or
universal claims about the models:

| Model / log | Recorded result |
|---|---|
| Conformer-TasNet (`Conformer-TasNet_L16000_tsnsc_atc.log`) | ATC compilation failed in the recorded environment. The log reports the TBE/Python error `cannot import name 'Iterable' from 'collections'`, followed by unsupported-operator and graph-build diagnostics involving `InstanceNorm`, `FullyConnection`, `Conv2DTransposeD`, `StridedSliceD`, `GatherV2`, and `ConcatD`, ending with `build graph failed`. |
| TDANet (`TDANet_L32000_tsnsc_atc.log`) | ATC parsing failed with the internal graph-parser error `Param:src_out_anchor is nullptr`; the model could not be parsed into an ATC graph and no OM file was generated. |
