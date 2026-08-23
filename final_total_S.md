./scripts/package_s.sh \
  --product data/enwik9.blsmc_full.m3v2.payload_sim \
  --weights-dir weights \
  --bitstream /home/azureuser/clone_model/blossom/xsa_ttt/runs/ac_full_w64_final/payload_final_fullsha.bin
[1/6] peel binary + WRT dictionary
  UPX -9  /home/azureuser/clone_model/altxs/vendor/upx/upx  → bin/blsmc_prepare
[2/6] sidecar trailer (not in the AC stream)
  from sidecar  /home/azureuser/clone_model/altxs/sidecars/payload_sim.trailer.bin  3208 B
[3/6] mixed-bit ΔW codec (the published student product)
  mixed_da_bpw3.15_upb1.8.safetensors
[4/6] decode import closure
[5/6] DECODE.env + DECODE.md
[6/6] MANIFEST + zip -9
LTCB Total S  =  |compressed enwik9|  +  |zip -9 decompresser|
This zip is the decompresser. The AC bitstream is counted separately.

group      file                                                                    raw       in zip
--------------------------------------------------------------------------------------------------
bin        bin/blsmc_prepare                                                    91,200       89,942
           — bin subtotal                                                       91,200       89,942

dict       dict/english.dic                                                    411,996      175,326
           — dict subtotal                                                     411,996      175,326

sidecars   sidecars/payload_sim.trailer.bin                                      3,208        2,639
           — sidecars subtotal                                                   3,208        2,639

weights    weights/mixed_da_bpw3.15_upb1.8.safetensors                      13,933,426   13,061,788
weights    weights/mixed_da_bpw3.15_upb1.8.json                                 10,482        2,322
           — weights subtotal                                               13,943,908   13,064,110

code       code/xsa_ttt/compress.py                                             78,285       19,398
code       code/xsa_ttt/model.py                                                81,452       17,676
code       code/xsa_ttt/incremental.py                                          76,667       17,436
code       code/xsa_ttt/train.py                                                58,964       14,032
code       code/xsa_ttt/mega_encode.py                                          63,858       12,146
code       code/xsa_ttt/mega_step.py                                            47,181        9,241
code       code/xsa_ttt/persistent_step.py                                      36,014        8,869
code       code/xsa_ttt/config.py                                               19,826        6,032
code       code/hyperflow_distillation/mixed_bit_delta.py                       16,064        4,930
code       code/xsa_ttt/fused_step.py                                           22,894        4,906
code       code/xsa_ttt/ttt_lora.py                                             15,953        4,029
code       code/model/arithmetic_coder_lm.py                                    11,985        3,532
code       code/xsa_ttt/ac_gemv.py                                              16,667        3,528
code       code/xsa_ttt/gpu_ac.py                                               14,378        3,524
code       code/xsa_ttt/split_attn.py                                           13,068        3,322
code       code/xsa_ttt/train_attn.py                                           17,231        3,157
code       code/xsa_ttt/data.py                                                  6,796        2,338
code       code/model/deterministic_mode.py                                      7,237        1,924
code       code/xsa_ttt/row_commit.py                                            3,247        1,319
code       code/xsa_ttt/checkpoint.py                                            2,637        1,016
code       code/xsa_ttt/chart.py                                                 1,983          809
code       code/xsa_ttt/device.py                                                1,900          721
code       code/hyperflow_distillation/train_hyperflow.py                          614          308
code       code/hyperflow_distillation/weight_space.py                             382          275
code       code/xsa_ttt/__init__.py                                                163          133
code       code/xsa_ttt/__main__.py                                                164          128
code       code/hyperflow_distillation/__init__.py                                  80           77
           — code subtotal                                                     615,690      144,806

(root)     MANIFEST.txt                                                          5,618        2,016
(root)     DECODE.md                                                             3,210        1,711
(root)     DECODE.env                                                            1,398          691
           — (root) subtotal                                                    10,226        4,418

--------------------------------------------------------------------------------------------------
           sum of members                                                   15,076,228   13,481,241
           zip central-dir / local-hdr overhead                                               9,160
S zip      /home/azureuser/clone_model/altxs/work/blsmc_ac_decoder.zip                   13,490,401

files=35  raw=15,076,228  packed members=13,481,241  zip=13,490,401  (89.4% of raw members)

bitstream    93,434,410   /home/azureuser/clone_model/blossom/xsa_ttt/runs/ac_full_w64_final/payload_final_fullsha.bin
decoder      13,490,401   /home/azureuser/clone_model/altxs/work/blsmc_ac_decoder.zip
S           106,924,811   = |bitstream| + |zip -9|
note: tagged ltcb-3.15bpw is bitstream=93,154,708 zip=13,437,796 S=106,592,504

wrote /home/azureuser/clone_model/altxs/work/blsmc_ac_decoder.S.txt

decoder zip  /home/azureuser/clone_model/altxs/work/blsmc_ac_decoder.zip  (13490401 B)
accounting   /home/azureuser/clone_model/altxs/work/blsmc_ac_decoder.S.txt  (not in S)
bitstream    /home/azureuser/clone_model/blossom/xsa_ttt/runs/ac_full_w64_final/payload_final_fullsha.bin  (93434410 B)  — not in the zip