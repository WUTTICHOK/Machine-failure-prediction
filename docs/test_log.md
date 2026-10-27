# Test log – Case 7 Machine Failure Prediction

Tester: ______  Date: ______  App URL: ______  Model trained at: ______ (see sidebar)

Fill **Actual result** and **Pass/Fail** while testing the *deployed* app. Add a screenshot reference for each.

## A. Normal inputs (Predict tab)
| ID | Input (Type, Air K, Process K, rpm, Nm, wear min) | Expected behaviour | Actual result | Pass/Fail |
|---|---|---|---|---|
| N1 | M, 300.0, 310.0, 1500, 40.0, 100 | No Failure, low probability, no warning | | |
| N2 | L, 302.0, 311.0, 1250, 70.0, 220 | Failure (overstrain: wear × torque high), warning-free | | |
| N3 | H, 303.5, 311.0, 1300, 60.0, 50 | High risk – small temperature difference + low speed (heat dissipation) | | |
| N4 | M, 298.0, 308.0, 2500, 12.0, 30 | Failure likely – power below ≈ 3,500 W | | |
| N5 | One test-set case from Notebook agreement tab | Same prediction & probability as notebook | | |

## B. Boundary inputs
| ID | Input | Expected behaviour | Actual result | Pass/Fail |
|---|---|---|---|---|
| B1 | All numeric at training minimum (see help text) | Prediction shown, **no** out-of-range warning | | |
| B2 | All numeric at training maximum | Prediction shown, **no** out-of-range warning | | |
| B3 | Tool wear = 0 (new tool) | Valid, prediction shown | | |
| B4 | Rotational speed = 3,500 rpm (above training max) | Prediction shown **with warning** "outside the training range" | | |
| B5 | Process temp = Air temp (e.g. 300 / 300) | Prediction shown with warning about temperature | | |
| B6 | Number input at hard limit (e.g. torque 150 Nm) | Accepted with extrapolation warning; cannot type above 150 | | |

## C. Invalid inputs (Batch tab – upload `example_inputs.csv` or your own file)
| ID | Input | Expected behaviour | Actual result | Pass/Fail |
|---|---|---|---|---|
| I1 | Type = "X" | Row REJECTED: type must be L/M/H | | |
| I2 | Torque = −5 | Row REJECTED: outside valid range 0–150 | | |
| I3 | Air temperature = "abc" | Row REJECTED: must be a number | | |
| I4 | Process temperature empty | Row REJECTED: missing | | |
| I5 | CSV missing the `torque_nm` column | Error "Missing required columns", app does not crash | | |
| I6 | Non-CSV / corrupted file | Error message, app does not crash | | |

## D. Notebook–dashboard agreement
| ID | Check | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| A1 | Notebook agreement tab | All 14 reference cases "match", max diff < 1e-9 | | |
| A2 | `pytest -q` in Colab/local | All tests pass | __ passed | |

## E. Monitoring
| ID | Action | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| M1 | Make < 30 predictions | Monitoring shows counts; "triggers evaluated once ≥ 30" | | |
| M2 | Log ≥ 30 normal predictions (batch, typical values) | No review trigger | | |
| M3 | Log ≥ 30 high-wear / low-speed predictions | Review trigger: predicted failure rate and mean shift | | |
| M4 | Streamlit Cloud → Manage app → Logs | Lines `prediction {...}` visible | | |
| M5 | Download prediction log | CSV contains inputs, probability, prediction only | | |

## Monitoring observations (short record)
| Date | Records in window | Predicted failure rate | Out-of-range share | Trigger? | Interpretation / action |
|---|---|---|---|---|---|
| | | | | | |
