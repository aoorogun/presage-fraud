# Example explanations (simulated data)

SHAP attributions describe the model, not causal truth. Counterfactuals are model-based what-ifs.

## Payment 330625 (ceo_fraud) risk 1.00

- [graph] beneficiary was already reported as a mule account (SHAP +7.43)
- [behaviour] payment time is 9.1 h away from the customer's usual hour (SHAP +1.94)
- [behaviour] amount is 5.2x this customer's historical mean payment (SHAP +0.41)
- Counterfactual: Had no confirmed mule accounts near the beneficiary and amount in line with customer history and beneficiary not forwarding funds rapidly, estimated risk would fall from 1.00 to 0.18.

## Payment 344269 (impersonation) risk 0.57

- [transaction] payment moves 8% of the account balance (SHAP +0.41)
- Counterfactual: Had no failed beneficiary additions / normal session and beneficiary previously paid by this customer and no confirmed mule accounts near the beneficiary, estimated risk would fall from 0.57 to 0.18.

## Payment 339761 (investment) risk 1.00

- [graph] beneficiary was already reported as a mule account (SHAP +7.72)
- [transaction] payment moves 9% of the account balance (SHAP +0.43)
- [behaviour] amount is 5.6x this customer's historical mean payment (SHAP +0.17)
- Counterfactual: Had no confirmed mule accounts near the beneficiary and amount in line with customer history and beneficiary previously paid by this customer, estimated risk would fall from 1.00 to 0.09.

## Payment 331066 (bank_impersonation) risk 0.82

- [behaviour] payment time is 9.1 h away from the customer's usual hour (SHAP +1.65)
- [beneficiary] beneficiary account is only 27 days old (SHAP +1.36)
- [transaction] payment moves 10% of the account balance (SHAP +0.78)
- [behaviour] amount is 6.7x this customer's historical mean payment (SHAP +0.39)
- Counterfactual: Had beneficiary account older than 90 days and beneficiary previously paid by this customer and no failed beneficiary additions / normal session, estimated risk would fall from 0.82 to 0.10.

## Payment 317076 (impersonation) risk 0.88

- [behaviour] share of balance moved is 1.5x the customer's previous maximum share (SHAP +0.59)
- [beneficiary] beneficiary has paid 12 distinct accounts onward (SHAP +0.49)
- [behaviour] amount is 20.4x this customer's historical mean payment (SHAP +0.44)
- [transaction] payment moves 44% of the account balance (SHAP +0.35)
- Counterfactual: Had amount in line with customer history and beneficiary previously paid by this customer and small share of balance, estimated risk would fall from 0.88 to 0.34.

## Payment 374197 (romance) risk 0.67

- [behaviour] payment time is 10.4 h away from the customer's usual hour (SHAP +1.49)
- [behaviour] amount is 5.1x this customer's historical mean payment (SHAP +0.16)
- Counterfactual: Had beneficiary previously paid by this customer and no confirmed mule accounts near the beneficiary and amount in line with customer history, estimated risk would fall from 0.67 to 0.44.

## Graph evidence (symbolic, strictly pre-payment)

- payment 312015 -> beneficiary 14443: {"confirmed_mules_1hop": [], "confirmed_mules_2hop": [], "shared_device_accounts": [], "n_distinct_recent_senders": 0}
- payment 312741 -> beneficiary 15051: {"confirmed_mules_1hop": [50], "confirmed_mules_2hop": [14478, 14479, 14480, 14481, 15051], "shared_device_accounts": [], "n_distinct_recent_senders": 0}
- payment 313548 -> beneficiary 2981: {"confirmed_mules_1hop": [14637, 14638, 14639, 14640], "confirmed_mules_2hop": [13, 19, 46, 51, 76], "shared_device_accounts": [3838, 14637, 14638, 14639, 14640], "n_distinct_recent_senders": 0}

## GNNExplainer demonstration (mule-node classifier on one snapshot)

- node 13: top-5 influential edges (src,dst,importance) = (14801,13,0.57); (14798,13,0.57); (9078,13,0.57); (14802,13,0.56); (14803,13,0.56)
- node 50: top-5 influential edges (src,dst,importance) = (653,50,0.58); (9182,50,0.58); (5209,50,0.58); (13582,50,0.58); (14481,50,0.58)
- Caveat: edge masks are optimisation artefacts of GNNExplainer for this model; they are evidence to inspect, not proof of causation.