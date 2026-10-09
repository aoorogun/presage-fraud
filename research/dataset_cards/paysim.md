# Dataset card: PaySim (real-data-derived synthetic mobile-money ledger)

| Item | Value |
|---|---|
| Source | Lopez-Rojas et al., PaySim simulator; Kaggle `ealtman`-independent mirror used here: Hugging Face `theman10/paysim` (tag `license:mit`) |
| Downloaded | `data/raw/paysim.csv`, 6,362,620 rows, 11 columns, 743 hourly steps, fraud 0.129% |
| Licence | MIT per the Hugging Face card; the original Kaggle dataset is distributed under CC BY-SA 4.0 - **verify upstream licence before redistribution** |
| Supports | transactional history, temporal modelling, sender/recipient graph |
| Does NOT support | APP labelling, devices, sessions, beneficiary age, mule labels |

**Honest framing:** PaySim fraud is *account takeover style* (transfer then cash-out), not authorised push payment fraud. It is used as an independent transaction-fraud benchmark and for cross-dataset generalisation only. No APP claim is made on PaySim.
