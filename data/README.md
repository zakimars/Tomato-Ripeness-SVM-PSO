# Data

| File | What it is |
|---|---|
| `sample_readings.csv` | The 30 example readings printed in the thesis (Table 4.1) — enough to test the code. |
| your dataset | Not included. The thesis used `dataset_tomat.xlsx` (sheet `Data Gabungan`, 381 readings from 82 tomatoes). |

Both scripts accept the thesis spreadsheet as-is: columns `R_norm, G_norm, B_norm, Label`, with labels `Tidak Matang` / `Setengah Matang` / `Matang`.

## Expected format

```csv
fruit_id,R,G,B,C,label
T001,575,153,102,800,ripe
T001,590,160,105,822,ripe
T002,507,232,100,838,partially_ripe
```

- `R,G,B,C` — raw TCS34725 channel values (integration time 50 ms, gain 4×, ~3 cm, closed box, sensor LED).
- `label` — `unripe`, `partially_ripe` or `ripe` (Indonesian `tidak matang` / `setengah matang` / `matang` also accepted).
- `fruit_id` — optional, but **required for `--split group`**: it keeps all readings of one tomato in the
  same split, which removes the train/test leakage noted in the thesis.

Readings with `C < 500` are dropped during preprocessing.
