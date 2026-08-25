# altxs

Public compressor for the [Matt Mahoney Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html#notes).

LTCB ranks **Total S = |compressed enwik9| + |zip of the decompresser|**. This git tree is how the result is published and verified. It is not the same object as Total S.

| Tag | Algorithm | Kind | Bitstream | Decoder zip | **S** |
|---|---|---|---:|---:|---:|
| `ltcb-3.15bpw` | Transformer + AC + dict/peel | `xd` (source/binary + separate bitstream) | 93,434,410 | 13,490,401 | **106,924,811** |

Decoder zip (`zip -9`, 35 members, 89.4% of raw) from [`final_total_S.md`](final_total_S.md):

| Group | Raw | In zip |
|---|---:|---:|
| bin (`blsmc_prepare`, UPX -9) | 91,200 | 89,942 |
| dict (`english.dic`) | 411,996 | 175,326 |
| sidecars (`payload_sim.trailer.bin`) | 3,208 | 2,639 |
| weights (`mixed_da_bpw3.15_upb1.8`) | 13,943,908 | 13,064,110 |
| code (decode import closure) | 615,690 | 144,806 |
| (root) `MANIFEST.txt` / `DECODE.md` / `DECODE.env` | 10,226 | 4,418 |
| packed members | 15,076,228 | 13,481,241 |
| zip central-dir / local-hdr overhead | | 9,160 |
| **S zip** `blsmc_ac_decoder.zip` | | **13,490,401** |

**S** = 93,434,410 + 13,490,401 = **106,924,811**. The AC bitstream is counted separately and is not in the zip.

[Files available for review here](https://github.com/altvectra/altxs/releases/tag/ltcb-3.15bpw)

## Submission for Large Text Compression Benchmark

Submission Details (Made for the Submission Guidance in https://mattmahoney.net/dc/textrules.html:

Name: altxs
Version: 1.0.0

enwik9: 93,434,410 bytes (enwiki9), 13,490,401 bytes (decoder zip), total S combined: 106,924,811 bytes
enwik8: NA (we haven't run the full algorithm on enwiki8 as of yet)


Options you used to obtain best compression of enwik9.
Available in [DECODE.env](DECODE.env)

Size of the decompressor as a zip file (smaller of source or executable).
13,490,401 bytes (decoder zip),

Approximate compression and decompression time.
ENCODE:
63:09:32 (63 hours, 9 minutes, 32 seconds)

(ending log of encode)
```
AC incr: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35174/35174 [63:09:32<00:00,  6.46s/seg, 0.5x:21% 1x:25% 2x:52% 4x:2%]
[xsa_ttt] AC bits/sym=1.29317 source_bpb=1.29317 payload=93,153,018 B / 576,278,322 sym est_full_bin=93,153,537 B (93.2 MB) roundtrip=n/a retrains=35173 sha256_ok=None
```

DECODE:
330:32:56 (330 hours, 32 minutes, 56 seconds)
That is 13d 18:32:56, or 1,189,976 seconds.
On a 1 GB enwik9 denominator that would be 1,189,976 ns/byte.


Approximate memory used.
RAM: ~8Gbs
VRAM: ~80Gbs (on GPU such as NVIDIA H100)

Here's the system we ran on:
```
inxi -Fxxxz

System:
  Kernel: 6.17.0-1021-azure arch: x86_64 bits: 64 compiler: gcc v: 13.3.0 clocksource: tsc
  Console: pty pts/0 Distro: Ubuntu 24.04.4 LTS (Noble Numbat)
Machine:
  Type: Desktop Mobo: Microsoft model: Virtual Machine v: Hyper-V UEFI Release v4.1
    serial: <superuser required> uuid: 72d96d6f-baed-43bb-b958-8e4ba1b988cc UEFI: Microsoft
    v: Hyper-V UEFI Release v4.1 date: 01/08/2026
CPU:
  Info: 40-core model: AMD EPYC 9V84 bits: 64 type: MCP smt: <unsupported> arch: Zen 4 rev: 1
    cache: L1: 2.5 MiB L2: 40 MiB L3: 160 MiB
  Speed (MHz): avg: 2464 high: 3699 min/max: N/A cores: 1: 2400 2: 2400 3: 2400 4: 2400 5: 2400
    6: 2400 7: 2400 8: 2400 9: 2400 10: 2400 11: 2400 12: 2400 13: 2400 14: 2400 15: 2400 16: 2400
    17: 3699 18: 2400 19: 2400 20: 2400 21: 2400 22: 2400 23: 2400 24: 2400 25: 2400 26: 2400
    27: 2400 28: 2400 29: 2400 30: 2400 31: 2400 32: 2400 33: 3697 34: 2400 35: 2400 36: 2400
    37: 2400 38: 2400 39: 2400 40: 2400 bogomips: 192002
  Flags: avx avx2 ht lm nx pae sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3 svm
Graphics:
  Device-1: NVIDIA GH100 [H100L 94GB] driver: nvidia v: 595.71.05 arch: Hopper
    bus-ID: 0001:00:00.0 chip-ID: 10de:2321 class-ID: 0302
  Display: server: X.org v: 1.21.1.11 driver: gpu: hyperv_drm tty: 237x67
  Monitor-1: Virtual-1 size-res: N/A in console modes: max: 1024x768 min: 640x480
  API: EGL v: 1.5 hw: drv: nvidia platforms: device: 0 drv: nvidia device: 2 drv: swrast
    surfaceless: drv: nvidia inactive: gbm,wayland,x11,device-1
  API: OpenGL v: 4.6.0 compat-v: 4.5 vendor: mesa v: 25.2.8-0ubuntu0.24.04.2
    note: console (EGL sourced) renderer: NVIDIA H100 NVL/PCIe/SSE2, llvmpipe (LLVM 20.1.2 256 bits)
Audio:
  Message: No device data found.
Network:
  Message: No PCI device data found.
  IF-ID-1: eth0 state: up speed: 100000 Mbps duplex: full mac: <filter>
Drives:
  Local Storage: total: 3.74 TiB used: 111.41 GiB (2.9%)
  ID-1: /dev/nvme0n1 vendor: Microsoft model: NVMe Direct Disk v2 size: 3.49 TiB speed: 16 Gb/s
    lanes: 4 tech: SSD serial: <filter> fw-rev: NVMDV002 temp: 71.8 C
  ID-2: /dev/sda model: Virtual Disk size: 128 GiB tech: N/A serial: N/A fw-rev: 1.0 scheme: GPT
  ID-3: /dev/sdb model: Virtual Disk size: 128 GiB tech: N/A serial: N/A fw-rev: 1.0 scheme: MBR
Partition:
  ID-1: / size: 122.95 GiB used: 111.13 GiB (90.4%) fs: ext4 dev: /dev/sda1
  ID-2: /boot size: 880.4 MiB used: 273.9 MiB (31.1%) fs: ext4 dev: /dev/sda16
  ID-3: /boot/efi size: 104.3 MiB used: 6.1 MiB (5.9%) fs: vfat dev: /dev/sda15
Swap:
  Alert: No swap data was found.
Sensors:
  System Temperatures: cpu: N/A mobo: N/A gpu: nvidia temp: 53 C
  Fan Speeds (rpm): N/A
Info:
  Memory: total: 320 GiB available: 314.69 GiB used: 5.78 GiB (1.8%)
  Processes: 474 Power: uptime: 17d 22h 59m states: freeze,mem,disk suspend: s2idle wakeups: 0
    hibernate: shutdown Init: systemd v: 255 target: graphical (5) default: graphical
  Packages: pm: dpkg pkgs: 1105 Compilers: clang: 17 gcc: 13.3.0 alt: 12 Shell: Bash v: 5.2.21
    running-in: pty pts/0 (SSH) inxi: 3.3.34

```

## Hardware (decode)

| | |
|---|---|
| GPU | NVIDIA H100 (80 GB). CUDA required. CPU-only is **not** supported. |
| VRAM | ~80 GB class (profile `large`, ~32M params + KV + fused window) |
| RAM / disk | A few GiB host RAM; ~2–3 GiB scratch for peel inverse |
| Encode wall clock | ~60 h for the full 576,278,322-symbol AC on one H100 |
| Decode wall clock | Encode pays one window step per W=64 symbols; decode pays one step per accepted token. Full reconstruct is multi-day (up to ~18 days on this class of machine). |

(here's the system we used)

```sh
inxi -Fxxxz

System:
  Kernel: 6.17.0-1021-azure arch: x86_64 bits: 64 compiler: gcc v: 13.3.0 clocksource: tsc
  Console: pty pts/0 Distro: Ubuntu 24.04.4 LTS (Noble Numbat)
Machine:
  Type: Desktop Mobo: Microsoft model: Virtual Machine v: Hyper-V UEFI Release v4.1
    serial: <superuser required> uuid: 72d96d6f-baed-43bb-b958-8e4ba1b988cc UEFI: Microsoft
    v: Hyper-V UEFI Release v4.1 date: 01/08/2026
CPU:
  Info: 40-core model: AMD EPYC 9V84 bits: 64 type: MCP smt: <unsupported> arch: Zen 4 rev: 1
    cache: L1: 2.5 MiB L2: 40 MiB L3: 160 MiB
  Speed (MHz): avg: 2464 high: 3699 min/max: N/A cores: 1: 2400 2: 2400 3: 2400 4: 2400 5: 2400
    6: 2400 7: 2400 8: 2400 9: 2400 10: 2400 11: 2400 12: 2400 13: 2400 14: 2400 15: 2400 16: 2400
    17: 3699 18: 2400 19: 2400 20: 2400 21: 2400 22: 2400 23: 2400 24: 2400 25: 2400 26: 2400
    27: 2400 28: 2400 29: 2400 30: 2400 31: 2400 32: 2400 33: 3697 34: 2400 35: 2400 36: 2400
    37: 2400 38: 2400 39: 2400 40: 2400 bogomips: 192002
  Flags: avx avx2 ht lm nx pae sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3 svm
Graphics:
  Device-1: NVIDIA GH100 [H100L 94GB] driver: nvidia v: 595.71.05 arch: Hopper
    bus-ID: 0001:00:00.0 chip-ID: 10de:2321 class-ID: 0302
  Display: server: X.org v: 1.21.1.11 driver: gpu: hyperv_drm tty: 237x67
  Monitor-1: Virtual-1 size-res: N/A in console modes: max: 1024x768 min: 640x480
  API: EGL v: 1.5 hw: drv: nvidia platforms: device: 0 drv: nvidia device: 2 drv: swrast
    surfaceless: drv: nvidia inactive: gbm,wayland,x11,device-1
  API: OpenGL v: 4.6.0 compat-v: 4.5 vendor: mesa v: 25.2.8-0ubuntu0.24.04.2
    note: console (EGL sourced) renderer: NVIDIA H100 NVL/PCIe/SSE2, llvmpipe (LLVM 20.1.2 256 bits)
Audio:
  Message: No device data found.
Network:
  Message: No PCI device data found.
  IF-ID-1: eth0 state: up speed: 100000 Mbps duplex: full mac: <filter>
Drives:
  Local Storage: total: 3.74 TiB used: 111.41 GiB (2.9%)
  ID-1: /dev/nvme0n1 vendor: Microsoft model: NVMe Direct Disk v2 size: 3.49 TiB speed: 16 Gb/s
    lanes: 4 tech: SSD serial: <filter> fw-rev: NVMDV002 temp: 71.8 C
  ID-2: /dev/sda model: Virtual Disk size: 128 GiB tech: N/A serial: N/A fw-rev: 1.0 scheme: GPT
  ID-3: /dev/sdb model: Virtual Disk size: 128 GiB tech: N/A serial: N/A fw-rev: 1.0 scheme: MBR
Partition:
  ID-1: / size: 122.95 GiB used: 111.13 GiB (90.4%) fs: ext4 dev: /dev/sda1
  ID-2: /boot size: 880.4 MiB used: 273.9 MiB (31.1%) fs: ext4 dev: /dev/sda16
  ID-3: /boot/efi size: 104.3 MiB used: 6.1 MiB (5.9%) fs: vfat dev: /dev/sda15
Swap:
  Alert: No swap data was found.
Sensors:
  System Temperatures: cpu: N/A mobo: N/A gpu: nvidia temp: 53 C
  Fan Speeds (rpm): N/A
Info:
  Memory: total: 320 GiB available: 314.69 GiB used: 5.78 GiB (1.8%)
  Processes: 474 Power: uptime: 17d 22h 59m states: freeze,mem,disk suspend: s2idle wakeups: 0
    hibernate: shutdown Init: systemd v: 255 target: graphical (5) default: graphical
  Packages: pm: dpkg pkgs: 1105 Compilers: clang: 17 gcc: 13.3.0 alt: 12 Shell: Bash v: 5.2.21
    running-in: pty pts/0 (SSH) inxi: 3.3.34
```

Determinism: `COMPRESSION_DETERMINISTIC=strict` (see `DECODE.env`).

## Student weights (what is public)

The dense student is **not** a dumped checkpoint. The Release ships a mixed-bit **ΔW** codec (`mixed_da_bpw*.safetensors` + `.json`). Anyone rebuilds the same student with:

```text
student = Init(seed from codec metadata) + dequantized ΔW
```

That codec is the product. How it was fit is not part of this repo.

**Not published:** a teacher model, calibration / distillation recipes, or any other path that produces the ΔW.

## Not supported

- Legacy non-incremental / chunked-TF bitstreams
- CPU-only decode
- Reproducing the student from anything except the shipped mixed-bit ΔW + `Init(seed)`
- Hutter Prize `archive9` limits (≤10 GiB RAM, CPU, Geekbench time). This is an LTCB GPU entry, same class as nncp / jax-compress.

## One-command encode

From a clean checkout. Official `enwik9` is not in git. Needs the shipped ΔW in `weights/`, CUDA PyTorch, and an H100-class GPU. Same `DECODE.env` as decode.

```bash
WITH_PYTHON=1 WITH_ENWIK9=1 ./scripts/setup.sh   # vendors, blsmc_prepare, venv, enwik9
./scripts/encode.sh                              # peel → Init(seed)+ΔW → full AC
```

Writes `work/ac_encode/payload_final_fullsha.bin` (full stream, ~60 h). Zip the decompresser and close S:

```bash
./scripts/package_s.sh \
  --bitstream work/ac_encode/payload_final_fullsha.bin \
  --product data/enwik9.blsmc_full.m3v2.payload_sim
```

If `data/enwik9` and the peel product are already present:

```bash
./scripts/encode.sh                  # peel again, then AC
./scripts/encode.sh --bitstream-only # skip peel; AC only
```

`--bytes 4194304` encodes a 4 MiB prefix (not a prefix of the ranking bitstream). Details: [ENCODE.md](ENCODE.md).

## One-command decode

Release assets for tag `ltcb-3.15bpw`: `payload_final_fullsha.bin` + `blsmc_ac_decoder.zip`.

```bash
./scripts/setup.sh                  # peel vendors + build blsmc_prepare
./scripts/decode.sh \
  --bitstream /path/to/payload_final_fullsha.bin \
  --decoder-zip /path/to/blsmc_ac_decoder.zip \
  --out work/enwik9
```

Or from a checkout that already has weights + trailer (after `package_decoder.sh` or a unpacked zip):

```bash
./scripts/decode.sh --bitstream /path/to/payload_final_fullsha.bin --out work/enwik9
```

Details: [DECODE.md](DECODE.md). Payload (`payload_sim`): [PEEL.md](PEEL.md). Encode / S: [ENCODE.md](ENCODE.md). Incremental AC check (4 MiB, not the multi-day full verify): [VALIDATE.md](VALIDATE.md).

```bash
./scripts/setup.sh                         # vendors + peel binary
WITH_ENWIK9=1 WITH_PEEL=1 ./scripts/setup.sh   # also fetch enwik9 and emit payload_sim
WITH_PYTHON=1 ./scripts/setup_python.sh    # uv sync --extra dev (.venv)
```

## Tests

```bash
./scripts/test_lockstep.sh    # fast, CI-able: encode ≡ decode on a short prefix
./scripts/roundtrip_4mb.sh    # production incremental path, 4 MiB prefix (GPU, hours)
./scripts/test_enwik9.sh      # full 1 GB reconstruct + official checksums (GPU, days — skip)
./scripts/package_s.sh --bitstream payload_final_fullsha.bin   # zip -9 + Total S
./scripts/measure_s.sh --bitstream payload_final_fullsha.bin --decoder-zip work/blsmc_ac_decoder.zip
```

See [VALIDATE.md](VALIDATE.md). Reviewers should run the 4 MiB roundtrip, not the full reconstruct.

## Report a result to LTCB

Send Matt Mahoney the bitstream, the `zip -9` decoder zip, this repo URL + tag, and the machine (GPU model, wall clock). Notes: https://mattmahoney.net/dc/text.html#notes

## Layout

```
README.md  LICENSE  NOTICE  DECODE.md  ENCODE.md  PEEL.md  VALIDATE.md  DECODE.env
pyproject.toml uv.lock requirements.txt   pinned AC runtime (uv sync)
src/                      encode + decode Python import closure
blsmc/prepare/            peel (enwik9 ↔ payload_sim)
vendor/cmix-lex/          PHDA9 / WRT / reorder sources (GPL-3)
dict/english.dic          WRT dictionary (counts in S)
dict/new_article_order    M2 article table (needed to *make* payload_sim)
scripts/setup.sh          procure vendors, UPX, compiler build, optional enwik9/venv
scripts/peel.sh           enwik9 → payload_sim
scripts/package_s.sh      decoder zip + Total S = |bitstream| + |zip -9|
scripts/                 decode, encode, measure S, tests
tests/                   lockstep unit tests (not packed into S)
```

## Acknowledgements

Peel and Total S stand on work we did not write. Licenses and the exact vendored subset: [NOTICE](NOTICE).

- [cmix-lex](https://github.com/blahem/cmix-lex) — PHDA9 / WRT / article reorder (`vendor/cmix-lex/`), plus `dict/english.dic` and `dict/new_article_order`
- [cmix](https://github.com/byronknoll/cmix) — WRT dictionary lineage for `english.dic`
- [UPX](https://upx.github.io) — packs `bin/blsmc_prepare` before `zip -9`; that packed binary is what enters S
- PyTorch, CUDA, and safetensors — AC encode / decode runtime (not vendored)
- [Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html) — Matt Mahoney
