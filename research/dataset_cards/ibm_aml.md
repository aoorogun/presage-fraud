# Dataset card: IBM Transactions for Anti-Money Laundering (HI-Small)

| Item | Value |
|---|---|
| Source | Altman et al. (IBM, NeurIPS 2023 datasets track) AMLworld generator; mirror used: Hugging Face `eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans` (no licence tag on the mirror) |
| Downloaded | `HI-Small_Trans.csv`, 5,078,345 transactions, 496,995 accounts, 30,470 banks, 2022-09-01 to 2022-09-18, laundering 0.102% (5,177) |
| Licence | upstream Kaggle release is CDLA-Sharing-1.0 (to be confirmed by the user before redistribution); the mirror states none |
| Supports | transactional history, account graph, temporal modelling, AML/mule-like laundering networks, cross-bank edges |
| Does NOT support | APP labels, devices, sessions |

Used for graph/temporal AML benchmarking and cross-dataset checks. Labels are laundering, not APP fraud.
