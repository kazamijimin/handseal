# HandSeal Dataset Audit

Dataset root: `C:\Users\balli\Desktop\HandSeal\HandSealProject\handseal_collected`
Total supported images: **4063**

## Counts

| Split | dragon | ram | snake | tiger | no_seal | Total |
|---|---|---|---|---|---|---|
| train | 264 | 192 | 201 | 201 | 201 | 1059 |
| valid | 95 | 95 | 95 | 114 | 95 | 494 |
| test | 502 | 503 | 502 | 501 | 502 | 2510 |

## Balance

- train: min=192 max=264 ratio=0.727
- valid: min=95 max=114 ratio=0.833
- test: min=501 max=503 ratio=0.996

## Integrity checks

- Corrupt images: 0
- Unsupported files: 0
- Exact duplicate groups: 1
- Near-duplicate groups: 47
- Cross-split exact duplicate groups: 0
- Cross-split near-duplicate groups: 2
- Cross-split source-key groups: 0
- Cross-split participant groups: 0
- Cross-split session groups: 0
- Images missing metadata: 0

The source dataset was not modified by this audit.
