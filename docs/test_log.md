# Test log – Case 7 Machine Failure Prediction

Tester: ______  Date: ______  App URL: ______  Model trained at: ______ (see sidebar)

Fill **Actual result** and **Pass/Fail** while testing the *deployed* app. Add a screenshot reference for each.
Input order below: air °C, process °C, rpm, torque N·m, tool wear min, vibration mm/s, machine age years.

## A. Normal inputs (Predict tab)
| ID | Input | Expected behaviour | Actual result | Pass/Fail |
|---|---|---|---|---|
| N1 | 28.0, 72.5, 1700, 50.0, 120, 2.4, 7.5 | No Failure, low probability, no warning | | |
| N2 | 28.0, 72.5, 1700, 82.0, 245, 5.5, 14.0 | Failure (high wear, torque, vibration, age) | | |
| N3 | 28.0, 72.5, 1700, 50.0, 230, 4.5, 7.5 | Higher probability than N1 (wear + vibration) | | |
| N4 | One test-set case from the Notebook agreement tab | Same prediction & probability as notebook | | |

## B. Boundary inputs
| ID | Input | Expected behaviour | Actual result | Pass/Fail |
|---|---|---|---|---|
| B1 | All inputs at training minimum (see help text) | Prediction shown, **no** out-of-range warning | | |
| B2 | All inputs at training maximum | Prediction shown, **no** out-of-range warning | | |
| B3 | Tool wear = 0 (new tool) | Valid, prediction shown | | |
| B4 | Rotational speed = 3,500 rpm (above training max) | Prediction shown **with warning** "outside the training range" | | |
| B5 | Process temp = air temp (e.g. 30 / 30) | Prediction shown with warning about temperature | | |
| B6 | Tick "Reading unavailable" for air temperature | Prediction shown with warning "missing; training median used" | | |

## C. Invalid inputs (Batch tab – upload `example_inputs.csv` or your own file)
| ID | Input | Expected behaviour | Actual result | Pass/Fail |
|---|---|---|---|---|
| I1 | Torque = −5 | Row REJECTED: outside valid range | | |
| I2 | Vibration = "abc" | Row REJECTED: must be a number | | |
| I3 | Tool wear empty | Row REJECTED: required | | |
| I4 | Air temperature empty | Row accepted **with warning** (median-filled) | | |
| I5 | CSV missing the `torque_nm` column | Error "Missing required columns", app does not crash | | |
| I6 | Non-CSV / corrupted file | Error message, app does not crash | | |

## D. Notebook–dashboard agreement
| ID | Check | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| A1 | Notebook agreement tab | All 15 reference cases "match", max diff < 1e-9 | | |
| A2 | `pytest -v` in Colab | All tests pass | __ passed | |

## E. Monitoring
| ID | Action | Expected | Actual | Pass/Fail |
|---|---|---|---|---|
| M1 | Make < 30 predictions | Monitoring shows counts; "triggers evaluated once ≥ 30" | | |
| M2 | Log ≥ 30 normal predictions (batch, typical values) | No review trigger | | |
| M3 | Log ≥ 30 high-wear / high-vibration predictions | Review trigger: failure rate and mean shift | | |
| M4 | Log ≥ 30 predictions with a missing temperature | Review trigger: missing temperature readings | | |
| M5 | Streamlit Cloud → Manage app → Logs | Lines `prediction {...}` visible | | |
| M6 | Download prediction log | CSV contains inputs, probability, prediction only | | |

## Monitoring observations (short record)
| Date | Records in window | Predicted failure rate | Out-of-range share | Missing share | Trigger? | Interpretation / action |
|---|---|---|---|---|---|---|
| | | | | | | |
