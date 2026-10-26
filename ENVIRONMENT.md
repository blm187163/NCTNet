# Environment Record

This file records the software and deployment configuration associated with the
released NCTNet artifacts. It is a deployment record, not a training or ONNX
export environment specification: the training implementation, checkpoint,
model-definition code, and intermediate ONNX file are not distributed in this
repository.

## Recorded Conversion Configuration

| Item | Recorded value | Status |
|---|---|---|
| Python | 3.10 | Observed in export and ATC diagnostic logs |
| ONNX opset | 11 | Recorded conversion setting |
| Input tensor | `float32`, `[1, 1, 1, 32000]` | Static, NCHW input |
| Output tensor | `float32`, `[1, 2, 32000]` | Two separated sources |
| Dynamic axes | Disabled | Recorded conversion setting |
| Constant folding | Enabled | Recorded conversion setting |
| Ascend Toolkit | `5.20.t6.2.b060` | Recorded ATC toolchain |
| ATC target identifier | `TsnsC` | Toolkit-specific value used in the recorded command |
| ATC input format | `NCHW` | Recorded ATC setting |
| ATC precision | FP16 | Released OM inference configuration |

## Board-side Runtime Configuration

| Platform | Runtime | Artifact | Input requirement |
|---|---|---|---|
| HiSilicon SS928 | Ascend ACL | `models/NCTNet.om` | 16-kHz PCM; 32,000 samples per block |
| Rockchip RK3588 | RKNN | `models/NCTNet.rknn` | 16-kHz PCM16; exactly 32,000 samples per file |

The board programs are written for 64-bit ARM Linux. Their SDK include paths,
linker flags, and runtime libraries must match the board image used for
compilation and execution.

## Verify a Local Environment

Before regenerating an OM artifact, verify the installed ATC target names and
record the output with the generated artifact:

```bash
export TOOLKIT=/usr/local/Ascend/ascend-toolkit/5.20.t6.2.b060
export ATC_BIN="$TOOLKIT/x86_64-linux/bin/atc"

python3 --version
"$ATC_BIN" --help | grep -i soc_version
```

## Released Artifact Checksums

| Artifact | SHA-256 |
|---|---|
| `models/NCTNet.om` | `cbebec3f16e92abd56ae6c394118158f6607337da804d45e5dde629f8efff36b` |
| `models/NCTNet.rknn` | `27976a60d67c767fb00c821520c97b74467dbbf3d197317601592c4633bfc554` |

On Linux, verify an artifact with:

```bash
sha256sum models/NCTNet.om models/NCTNet.rknn
```
