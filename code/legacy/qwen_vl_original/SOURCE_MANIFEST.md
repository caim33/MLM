# QwenVL legacy data source manifest

Source checkout:

```text
path: /wangbenyou-sulongjie/qwen-vl-finetune
branch: main
commit: 7f2f6c1d5651e069f849128435081d98e367909c
commit_time: 2026-07-06T20:07:57+00:00
commit_subject: Keep description branch sample ids aligned
qwenvl/data tree: bb3d97b58240e78eb18caf92c049b86364adc4be
```

The five working-tree files matched the blobs at that commit during the
2026-08-29 audit.

| File | Git blob | SHA-256 |
|---|---|---|
| `qwenvl/data/__init__.py` | `57600e1d484fb947ab8f3f671f942f453434fbdd` | `f440cda919ef722d6e55150b779c62b46d8f310b63dd1e78c1639907eba8a9fb` |
| `qwenvl/data/data_processor.py` | `50678665e698592c86305e64acabd2bbcd55740a` | `eebc145f09cc5d25c18dc4fbfd8fa17a2cbe90312f3c8748fbb439673f41cdf4` |
| `qwenvl/data/rope2d.py` | `748c5a19534f68d693bb9a2d5e1980dea6562b0c` | `97c9f8f49a3a1f782f6dacea538e3032fadfe24e5b60a630d3ae62053250542c` |
| `qwenvl/data/Mean.npy` | `945f41b511d641e859134c5768dbb74ed808b4e9` | `26e136555dab04c94a129d446c26e6b9939cbf045fbf77bcf5462c1fb5a2001c` |
| `qwenvl/data/Std.npy` | `30114b3f7b6335671d3ed97030d9668b7d435027` | `6565a65ed9b31e23c328829a309e1c482be8b85fd23b43d65451a9b19a917f40` |

The source registry contains personal absolute paths, and the source processor
does not implement the clean codebase identity, containment, no-substitution,
or motion-ownership contracts. This directory is intentionally not a Python
package and must remain outside active imports.
