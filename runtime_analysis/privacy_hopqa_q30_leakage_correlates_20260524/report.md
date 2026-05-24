# Privacy HopQA q30 Leakage Correlate Audit
Data are matched completed rollout rows with StepFun privacy judgments from the final 16k test-set refresh roots. Leakage is `overall_leaked_for_reward` / score >= 4 unless otherwise stated.
## Coverage
- Joined judged rollouts: 23,124
- Models: 11
- Conditions: 22 model x prompt cells
- Tasks: 35
- Overall leakage: 35.34%
- Strict leakage: 23.24%
- Skipped malformed JSONL lines: 5; see `input_warnings.json`

## Takeaways
- The most robust non-judge correlates are visible-query specificity/volume, especially company mentions in web queries and total visible web-query count. These remain positive within models and after task/model fixed-effect checks.
- Local-hop-heavy and longer chains leak more. This persists within model x prompt cells, so it is not only a cross-model artifact.
- Retrieval breadth correlates with leakage, but more weakly than visible query text: candidate windows/docs, searches, docs read, and retrieved local docs are all positive.
- Task correctness is a weak positive correlate, not the main driver. Strict chain success and conditional hop accuracy have much smaller effects than query/local-hop features.
- Model/prompt identity has substantial predictive power, but behavior-only features still predict leakage under task-grouped cross-validation.
- AUC is not a statistical test. The AUC section is a predictive-discrimination check; the hypothesis-style evidence is in the correlation, partial-correlation, and BH-adjusted q-value tables.

## Strongest Non-Judge Numeric Correlates
| feature                                         |     n |   mean_clean |   mean_leaked |   mean_diff_leaked_minus_clean |   point_biserial_r |   point_biserial_q |   single_feature_auc_directionless |
|:------------------------------------------------|------:|-------------:|--------------:|-------------------------------:|-------------------:|-------------------:|-----------------------------------:|
| query_company_mentions                          | 23124 |        2.089 |         3.432 |                         1.343  |             0.2675 |         0          |                             0.709  |
| metric_candidate_windows_before_parent_trim     | 23124 |       78.4   |       101     |                        22.6    |             0.2373 |         1.415e-291 |                             0.6512 |
| metric_candidate_windows_after_parent_trim      | 23124 |       78.4   |       101     |                        22.6    |             0.2373 |         1.415e-291 |                             0.6512 |
| metric_candidate_parent_docs_before_parent_trim | 23124 |       33.92  |        41.21  |                         7.292  |             0.2106 |         1.5e-228   |                             0.635  |
| metric_candidate_parent_docs_after_parent_trim  | 23124 |       33.92  |        41.21  |                         7.292  |             0.2106 |         1.5e-228   |                             0.635  |
| query_number_mentions                           | 23124 |        3.262 |         4.735 |                         1.472  |             0.1991 |         5.893e-204 |                             0.6416 |
| n_web_queries_visible                           | 23124 |        4.876 |         6.453 |                         1.577  |             0.1989 |         1.195e-203 |                             0.6331 |
| query_count_from_payload                        | 23124 |        4.876 |         6.453 |                         1.577  |             0.1989 |         1.195e-203 |                             0.6331 |
| query_total_chars                               | 23124 |      365.2   |       511.4   |                       146.2    |             0.1969 |         1.578e-199 |                             0.6378 |
| query_total_tokens                              | 23124 |       48.76  |        67.7   |                        18.94   |             0.1904 |         1.953e-186 |                             0.6327 |
| pattern_local_hops                              | 23124 |        2.115 |         2.446 |                         0.3313 |             0.1681 |         3.989e-145 |                             0.5936 |
| dataset_n_local_hops                            | 23124 |        2.115 |         2.446 |                         0.3313 |             0.1681 |         3.989e-145 |                             0.5936 |
| query_max_chars                                 | 23124 |       84.23  |        95.19  |                        10.96   |             0.1598 |         5.09e-131  |                             0.5931 |
| metric_searches_total                           | 23124 |        9.911 |        12.01  |                         2.099  |             0.1554 |         5.785e-124 |                             0.6035 |
| pattern_longest_local_run                       | 23124 |        1.999 |         2.302 |                         0.3032 |             0.1552 |         1.06e-123  |                             0.5862 |
| metric_gold_parent_read_hops                    | 23124 |        2.436 |         2.887 |                         0.4506 |             0.1535 |         5.115e-121 |                             0.5894 |
| metric_gold_parent_retrieved_before_trim_hops   | 23124 |        2.599 |         3.048 |                         0.4489 |             0.1529 |         5.22e-120  |                             0.5901 |
| metric_gold_parent_retrieved_hops               | 23124 |        2.599 |         3.048 |                         0.4489 |             0.1529 |         5.22e-120  |                             0.5901 |
| query_max_tokens                                | 23124 |       11.41  |        12.85  |                         1.433  |             0.1511 |         2.834e-117 |                             0.5824 |
| query_year_mentions                             | 23124 |        2.671 |         3.717 |                         1.046  |             0.1505 |         2.108e-116 |                             0.6055 |

## Strongest Adjusted Correlates

### model x prompt fixed effects
| feature                                         |     n |   partial_linear_r |   partial_linear_q |
|:------------------------------------------------|------:|-------------------:|-------------------:|
| pattern_local_hops                              | 23124 |             0.1781 |         2.933e-162 |
| dataset_n_local_hops                            | 23124 |             0.1781 |         2.933e-162 |
| query_company_mentions                          | 23124 |             0.1736 |         3.871e-154 |
| pattern_longest_local_run                       | 23124 |             0.1643 |         4.036e-138 |
| metric_candidate_parent_docs_before_parent_trim | 23124 |             0.1623 |         8.226e-135 |
| metric_candidate_parent_docs_after_parent_trim  | 23124 |             0.1623 |         8.226e-135 |
| pattern_starts_local                            | 23124 |             0.159  |         1.65e-129  |
| metric_candidate_windows_before_parent_trim     | 23124 |             0.1579 |         1.06e-127  |
| metric_candidate_windows_after_parent_trim      | 23124 |             0.1579 |         1.06e-127  |
| metric_gold_parent_retrieved_hops               | 23124 |             0.1374 |         1.111e-96  |
| metric_gold_parent_retrieved_before_trim_hops   | 23124 |             0.1374 |         1.111e-96  |
| metric_gold_parent_read_hops                    | 23124 |             0.1373 |         1.459e-96  |
| dataset_n_unique_local_docs                     | 23124 |             0.1338 |         8.033e-92  |
| metric_doc_choose_training_calls                | 23124 |             0.1315 |         1.205e-88  |
| metric_gold_parent_hops                         | 23124 |             0.13   |         7.675e-87  |

### task fixed effects
| feature                                         |     n |   partial_linear_r |   partial_linear_q |
|:------------------------------------------------|------:|-------------------:|-------------------:|
| query_company_mentions                          | 23124 |             0.2645 |         0          |
| metric_candidate_windows_before_parent_trim     | 23124 |             0.2475 |         3.416e-318 |
| metric_candidate_windows_after_parent_trim      | 23124 |             0.2475 |         3.416e-318 |
| metric_candidate_parent_docs_before_parent_trim | 23124 |             0.2119 |         1.959e-231 |
| metric_candidate_parent_docs_after_parent_trim  | 23124 |             0.2119 |         1.959e-231 |
| query_count_from_payload                        | 23124 |             0.2013 |         9.318e-209 |
| n_web_queries_visible                           | 23124 |             0.2013 |         9.318e-209 |
| query_number_mentions                           | 23124 |             0.2006 |         2.385e-207 |
| query_total_chars                               | 23124 |             0.1984 |         8.875e-203 |
| query_total_tokens                              | 23124 |             0.1933 |         2.166e-192 |
| query_max_chars                                 | 23124 |             0.1583 |         1.994e-128 |
| dataset_n_local_hops                            | 23124 |             0.1568 |         3.216e-126 |
| pattern_local_hops                              | 23124 |             0.1568 |         3.216e-126 |
| metric_searches_total                           | 23124 |             0.1549 |         4.215e-123 |
| query_max_tokens                                | 23124 |             0.1532 |         1.812e-120 |

### model x prompt + task fixed effects
| feature                                         |     n |   partial_linear_r |   partial_linear_q |
|:------------------------------------------------|------:|-------------------:|-------------------:|
| query_company_mentions                          | 23124 |             0.1672 |         1.983e-142 |
| pattern_local_hops                              | 23124 |             0.1664 |         1.715e-141 |
| dataset_n_local_hops                            | 23124 |             0.1664 |         1.715e-141 |
| metric_candidate_windows_before_parent_trim     | 23124 |             0.1612 |         5.245e-133 |
| metric_candidate_windows_after_parent_trim      | 23124 |             0.1612 |         5.245e-133 |
| metric_candidate_parent_docs_before_parent_trim | 23124 |             0.1605 |         6.635e-132 |
| metric_candidate_parent_docs_after_parent_trim  | 23124 |             0.1605 |         6.635e-132 |
| pattern_longest_local_run                       | 23124 |             0.1517 |         8.613e-118 |
| pattern_starts_local                            | 23124 |             0.1431 |         7.244e-105 |
| dataset_n_unique_local_docs                     | 23124 |             0.1317 |         9.289e-89  |
| metric_gold_parent_retrieved_before_trim_hops   | 23124 |             0.1311 |         3.846e-88  |
| metric_gold_parent_retrieved_hops               | 23124 |             0.1311 |         3.846e-88  |
| metric_gold_parent_read_hops                    | 23124 |             0.1297 |         2.702e-86  |
| pattern_len                                     | 23124 |             0.1266 |         2.334e-82  |
| dataset_n_hops                                  | 23124 |             0.1266 |         2.334e-82  |

## Predictive Experiments
| experiment                        |     n |   positive_rate | cv                  |   folds |   auc_mean |   auc_sd |
|:----------------------------------|------:|----------------:|:--------------------|--------:|-----------:|---------:|
| model/prompt only                 | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.6888 | 0.009467 |
| task design only                  | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.5902 | 0.03034  |
| retrieval behavior only           | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.672  | 0.0203   |
| task performance only             | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.5762 | 0.009605 |
| token/latency only                | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.5778 | 0.00978  |
| visible query text only           | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.7178 | 0.01646  |
| all non-judge features            | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.7496 | 0.01144  |
| all non-judge except model/prompt | 23124 |          0.3534 | GroupKFold(task_id) |       5 |     0.7342 | 0.01476  |

## Within-Model Correlates
These aggregate per-model correlations with Fisher weighting. They are meant to check whether pooled correlates survive after removing cross-model comparisons.
| feature                                         |   groups |   total_n |   weighted_fisher_r |   weighted_fisher_q |   median_r |    min_r |   max_r |   positive_group_fraction |   median_auc_directionless |
|:------------------------------------------------|---------:|----------:|--------------------:|--------------------:|-----------:|---------:|--------:|--------------------------:|---------------------------:|
| query_company_mentions                          |       11 |     23124 |              0.2062 |          1.481e-219 |     0.188  |  0.1268  |  0.3356 |                    1      |                     0.646  |
| dataset_n_local_hops                            |       11 |     23124 |              0.1684 |          1.437e-145 |     0.1452 | -0.04562 |  0.2821 |                    0.9091 |                     0.588  |
| pattern_local_hops                              |       11 |     23124 |              0.1684 |          1.437e-145 |     0.1452 | -0.04562 |  0.2821 |                    0.9091 |                     0.588  |
| metric_candidate_parent_docs_after_parent_trim  |       11 |     23124 |              0.1616 |          5.24e-134  |     0.1705 |  0.03158 |  0.2121 |                    1      |                     0.6077 |
| metric_candidate_parent_docs_before_parent_trim |       11 |     23124 |              0.1616 |          5.24e-134  |     0.1705 |  0.03158 |  0.2121 |                    1      |                     0.6077 |
| metric_candidate_windows_after_parent_trim      |       11 |     23124 |              0.1598 |          3.833e-131 |     0.1647 |  0.09472 |  0.2219 |                    1      |                     0.6136 |
| metric_candidate_windows_before_parent_trim     |       11 |     23124 |              0.1598 |          3.833e-131 |     0.1647 |  0.09472 |  0.2219 |                    1      |                     0.6136 |
| pattern_longest_local_run                       |       11 |     23124 |              0.1544 |          1.945e-122 |     0.1185 | -0.05044 |  0.2747 |                    0.9091 |                     0.5793 |
| pattern_starts_local                            |       11 |     23124 |              0.1535 |          5.182e-121 |     0.1545 |  0.06151 |  0.2335 |                    1      |                     0.5443 |
| metric_gold_parent_read_hops                    |       11 |     23124 |              0.1327 |          2.291e-90  |     0.1336 |  0.00749 |  0.2148 |                    1      |                     0.5729 |
| metric_gold_parent_retrieved_before_trim_hops   |       11 |     23124 |              0.1327 |          2.332e-90  |     0.1366 |  0.01639 |  0.215  |                    1      |                     0.5784 |
| metric_gold_parent_retrieved_hops               |       11 |     23124 |              0.1327 |          2.332e-90  |     0.1366 |  0.01639 |  0.215  |                    1      |                     0.5784 |
| metric_doc_choose_training_calls                |       11 |     23124 |              0.1297 |          2.24e-86   |     0.1326 | -0.03242 |  0.1779 |                    0.9091 |                     0.5871 |
| dataset_n_unique_local_docs                     |       11 |     23124 |              0.1274 |          2.488e-83  |     0.1208 | -0.01331 |  0.2022 |                    0.9091 |                     0.5646 |
| dataset_n_hops                                  |       11 |     23124 |              0.1241 |          2.806e-79  |     0.1217 | -0.01908 |  0.2089 |                    0.9091 |                     0.5753 |
| metric_gold_parent_hops                         |       11 |     23124 |              0.1241 |          2.806e-79  |     0.1217 | -0.01908 |  0.2089 |                    0.9091 |                     0.5753 |
| metric_n_question_hops                          |       11 |     23124 |              0.1241 |          2.806e-79  |     0.1217 | -0.01908 |  0.2089 |                    0.9091 |                     0.5753 |
| metric_total_hops                               |       11 |     23124 |              0.1241 |          2.806e-79  |     0.1217 | -0.01908 |  0.2089 |                    0.9091 |                     0.5753 |
| n_hops                                          |       11 |     23124 |              0.1241 |          2.806e-79  |     0.1217 | -0.01908 |  0.2089 |                    0.9091 |                     0.5753 |
| pattern_len                                     |       11 |     23124 |              0.1241 |          2.806e-79  |     0.1217 | -0.01908 |  0.2089 |                    0.9091 |                     0.5753 |

## Within Model x Prompt Correlates
These repeat the same check inside each model/prompt cell, which is stricter because prompt effects are not pooled.
| feature                                         |   groups |   total_n |   weighted_fisher_r |   weighted_fisher_q |   median_r |     min_r |   max_r |   positive_group_fraction |   median_auc_directionless |
|:------------------------------------------------|---------:|----------:|--------------------:|--------------------:|-----------:|----------:|--------:|--------------------------:|---------------------------:|
| query_company_mentions                          |       22 |     23124 |              0.2    |          5.639e-206 |     0.1985 |  0.06847  |  0.3165 |                    1      |                     0.6508 |
| dataset_n_local_hops                            |       22 |     23124 |              0.1689 |          3.248e-146 |     0.1596 | -0.06339  |  0.2856 |                    0.9091 |                     0.5936 |
| pattern_local_hops                              |       22 |     23124 |              0.1689 |          3.248e-146 |     0.1596 | -0.06339  |  0.2856 |                    0.9091 |                     0.5936 |
| metric_candidate_parent_docs_after_parent_trim  |       22 |     23124 |              0.1591 |          1.226e-129 |     0.1693 |  0.007877 |  0.223  |                    1      |                     0.6085 |
| metric_candidate_parent_docs_before_parent_trim |       22 |     23124 |              0.1591 |          1.226e-129 |     0.1693 |  0.007877 |  0.223  |                    1      |                     0.6085 |
| metric_candidate_windows_after_parent_trim      |       22 |     23124 |              0.1552 |          2.113e-123 |     0.1615 |  0.06028  |  0.2321 |                    1      |                     0.6102 |
| metric_candidate_windows_before_parent_trim     |       22 |     23124 |              0.1552 |          2.113e-123 |     0.1615 |  0.06028  |  0.2321 |                    1      |                     0.6102 |
| pattern_longest_local_run                       |       22 |     23124 |              0.1548 |          6.313e-123 |     0.1343 | -0.06917  |  0.2841 |                    0.9091 |                     0.5843 |
| pattern_starts_local                            |       22 |     23124 |              0.1537 |          3.581e-121 |     0.1552 |  0.04656  |  0.2474 |                    1      |                     0.5499 |
| metric_gold_parent_retrieved_before_trim_hops   |       22 |     23124 |              0.1331 |          9.425e-91  |     0.1296 |  0.01174  |  0.2384 |                    1      |                     0.5804 |
| metric_gold_parent_retrieved_hops               |       22 |     23124 |              0.1331 |          9.425e-91  |     0.1296 |  0.01174  |  0.2384 |                    1      |                     0.5804 |
| metric_gold_parent_read_hops                    |       22 |     23124 |              0.1328 |          1.855e-90  |     0.1327 | -0.006082 |  0.2314 |                    0.9545 |                     0.5843 |
| metric_doc_choose_training_calls                |       22 |     23124 |              0.1298 |          2.477e-86  |     0.1417 | -0.05949  |  0.1959 |                    0.9545 |                     0.5866 |
| dataset_n_unique_local_docs                     |       22 |     23124 |              0.1274 |          2.892e-83  |     0.1326 | -0.01528  |  0.2036 |                    0.9091 |                     0.5673 |
| dataset_n_hops                                  |       22 |     23124 |              0.1248 |          4.402e-80  |     0.1283 | -0.03692  |  0.2239 |                    0.9545 |                     0.5792 |
| metric_gold_parent_hops                         |       22 |     23124 |              0.1248 |          4.402e-80  |     0.1283 | -0.03692  |  0.2239 |                    0.9545 |                     0.5792 |
| metric_n_question_hops                          |       22 |     23124 |              0.1248 |          4.402e-80  |     0.1283 | -0.03692  |  0.2239 |                    0.9545 |                     0.5792 |
| metric_total_hops                               |       22 |     23124 |              0.1248 |          4.402e-80  |     0.1283 | -0.03692  |  0.2239 |                    0.9545 |                     0.5792 |
| n_hops                                          |       22 |     23124 |              0.1248 |          4.402e-80  |     0.1283 | -0.03692  |  0.2239 |                    0.9545 |                     0.5792 |
| pattern_len                                     |       22 |     23124 |              0.1248 |          4.402e-80  |     0.1283 | -0.03692  |  0.2239 |                    0.9545 |                     0.5792 |

## Highest Leakage Groups

### condition_label
| value                                     |    n |   leakage_rate |   strict_leakage_rate |   mean_score |
|:------------------------------------------|-----:|---------------:|----------------------:|-------------:|
| Qwen3 4B Task step 50 (privacy prompt)    | 1032 |         0.5514 |                0.438  |        3.885 |
| Qwen3 4B Task step 50 (baseline)          | 1032 |         0.5494 |                0.4079 |        3.883 |
| Qwen3 4B Task step 100 (privacy prompt)   | 1032 |         0.5203 |                0.3711 |        3.834 |
| Qwen3 4B Task step 100 (baseline)         | 1032 |         0.5174 |                0.3547 |        3.812 |
| Reward-only hop-acc s102 (privacy prompt) | 1032 |         0.5107 |                0.3605 |        3.818 |
| Reward-only prefix s109 (baseline)        | 1032 |         0.5087 |                0.3178 |        3.814 |
| Reward-only hop-acc s102 (baseline)       | 1032 |         0.5058 |                0.3382 |        3.812 |
| Reward-only prefix s109 (privacy prompt)  | 1032 |         0.4574 |                0.2888 |        3.732 |
| GPT-OSS 20B (baseline)                    | 1032 |         0.4293 |                0.2897 |        3.645 |
| Step-3.5 Flash (privacy prompt)           | 1332 |         0.3994 |                0.2485 |        3.625 |
| Step-3.5 Flash (baseline)                 | 1032 |         0.3963 |                0.2461 |        3.626 |
| GPT-OSS 20B (privacy prompt)              | 1032 |         0.3479 |                0.2297 |        3.476 |

### pattern
| value   |    n |   leakage_rate |   strict_leakage_rate |   mean_score |
|:--------|-----:|---------------:|----------------------:|-------------:|
| LWWLW   |  198 |         0.5909 |                0.4949 |        3.854 |
| LLLL    |   66 |         0.5758 |                0.5303 |        3.803 |
| LLWWLLW |  341 |         0.5249 |                0.3754 |        3.836 |
| LLLLWWW |  683 |         0.5212 |                0.3895 |        3.857 |
| LLLLW   | 1700 |         0.4871 |                0.3706 |        3.776 |
| LLL     | 3891 |         0.405  |                0.257  |        3.42  |
| LWWL    |  528 |         0.4034 |                0.2822 |        3.528 |
| LLW     | 6636 |         0.3651 |                0.2328 |        3.518 |
| LWW     |  330 |         0.3606 |                0.2758 |        3.518 |
| LWL     |  264 |         0.3598 |                0.2576 |        3.53  |
| LLWWW   | 1842 |         0.342  |                0.2047 |        3.527 |
| LLWWL   | 1035 |         0.3391 |                0.172  |        3.531 |

### pattern_len
|   value |     n |   leakage_rate |   strict_leakage_rate |   mean_score |
|--------:|------:|---------------:|----------------------:|-------------:|
|       7 |  1090 |         0.5092 |                0.3743 |        3.822 |
|       4 |   660 |         0.4045 |                0.3    |        3.542 |
|       5 |  4841 |         0.3989 |                0.2654 |        3.623 |
|       3 | 11715 |         0.367  |                0.2338 |        3.468 |
|       2 |  4818 |         0.2323 |                0.1544 |        3.256 |

### pattern_local_hops
|   value |     n |   leakage_rate |   strict_leakage_rate |   mean_score |
|--------:|------:|---------------:|----------------------:|-------------:|
|       4 |  2790 |       0.5022   |              0.3796   |        3.804 |
|       3 |  4992 |       0.39     |              0.2388   |        3.443 |
|       2 | 10326 |       0.3602   |              0.231    |        3.505 |
|       1 |  4818 |       0.2289   |              0.153    |        3.273 |
|       0 |   198 |       0.005051 |              0.005051 |        2.995 |

### pattern_web_hops
|   value |     n |   leakage_rate |   strict_leakage_rate |   mean_score |
|--------:|------:|---------------:|----------------------:|-------------:|
|       3 |  3130 |         0.4112 |                0.2783 |        3.644 |
|       0 |  4617 |         0.3955 |                0.2525 |        3.405 |
|       1 | 12626 |         0.3374 |                0.2262 |        3.472 |
|       4 |    66 |         0.303  |                0.2121 |        3.394 |
|       2 |  2685 |         0.2898 |                0.1739 |        3.415 |

### strict_chain_success
|   value |     n |   leakage_rate |   strict_leakage_rate |   mean_score |
|--------:|------:|---------------:|----------------------:|-------------:|
|       1 | 11778 |         0.3686 |                0.2543 |        3.506 |
|       0 | 11346 |         0.3376 |                0.2097 |        3.444 |

## Quartile Checks

### n_web_queries_visible
| bin           |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|-----:|-------------:|---------------:|----------------------:|
| (-0.001, 3.0] | 7357 |        1.999 |         0.2085 |                0.1323 |
| (3.0, 5.0]    | 6984 |        4.451 |         0.3892 |                0.2656 |
| (5.0, 7.0]    | 3658 |        6.43  |         0.427  |                0.2644 |
| (7.0, 36.0]   | 5125 |       10.99  |         0.4599 |                0.3081 |

### n_retrieved_local_docs
| bin           |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|-----:|-------------:|---------------:|----------------------:|
| (-0.001, 3.0] | 8979 |        1.992 |         0.283  |                0.1841 |
| (3.0, 4.0]    | 3151 |        4     |         0.3942 |                0.2701 |
| (4.0, 7.0]    | 6684 |        5.813 |         0.4057 |                0.2735 |
| (7.0, 26.0]   | 4310 |       10.3   |         0.3889 |                0.2418 |

### metric_searches_total
| bin           |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|-----:|-------------:|---------------:|----------------------:|
| (-0.001, 6.0] | 6741 |        4.974 |         0.2302 |                0.1478 |
| (13.0, 42.0]  | 5637 |       19.82  |         0.4492 |                0.2991 |
| (6.0, 9.0]    | 7232 |        8.408 |         0.3944 |                0.2637 |
| (9.0, 13.0]   | 3514 |       11.45  |         0.3515 |                0.2234 |

### metric_docs_read
| bin            |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:---------------|-----:|-------------:|---------------:|----------------------:|
| (-0.001, 12.0] | 6071 |        8.229 |         0.311  |                0.2094 |
| (12.0, 21.0]   | 5565 |       16.7   |         0.3292 |                0.223  |
| (21.0, 40.0]   | 5893 |       29.87  |         0.3485 |                0.2287 |
| (40.0, 288.0]  | 5595 |       66.49  |         0.4284 |                0.2706 |

### output_tokens
| bin                 |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------------|-----:|-------------:|---------------:|----------------------:|
| (1109.999, 4345.0]  | 5783 | 3171         |         0.3137 |                0.2184 |
| (17641.0, 144651.0] | 5781 |    3.549e+04 |         0.3486 |                0.214  |
| (4345.0, 8363.5]    | 5779 | 6093         |         0.3655 |                0.2514 |
| (8363.5, 17641.0]   | 5781 |    1.227e+04 |         0.3857 |                0.2458 |

### prompt_tokens
| bin                    |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:-----------------------|-----:|-------------:|---------------:|----------------------:|
| (110571.0, 200677.25]  | 5781 |    1.497e+05 |         0.3506 |                0.2304 |
| (200677.25, 1553324.0] | 5781 |    3.295e+05 |         0.4283 |                0.2724 |
| (2455.999, 62437.0]    | 5781 |    4.454e+04 |         0.3044 |                0.2058 |
| (62437.0, 110571.0]    | 5781 |    8.444e+04 |         0.33   |                0.2209 |

### total_tokens
| bin                   |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:----------------------|-----:|-------------:|---------------:|----------------------:|
| (122142.0, 221222.5]  | 5781 |    1.657e+05 |         0.3439 |                0.227  |
| (221222.5, 1625463.0] | 5781 |    3.558e+05 |         0.4254 |                0.2692 |
| (69548.75, 122142.0]  | 5781 |    9.353e+04 |         0.3325 |                0.2211 |
| (8515.999, 69548.75]  | 5781 |    5.016e+04 |         0.3117 |                0.2124 |

### conditional_hop_accuracy
| bin           |     n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|------:|-------------:|---------------:|----------------------:|
| (-0.001, 0.5] |  6077 |       0.2172 |         0.3028 |                0.1871 |
| (0.5, 1.0]    | 17047 |       0.9097 |         0.3714 |                0.2485 |

### pattern_len
| bin          |     n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:-------------|------:|-------------:|---------------:|----------------------:|
| (1.999, 3.0] | 16533 |        2.709 |         0.3277 |                0.2107 |
| (3.0, 5.0]   |  5501 |        4.88  |         0.3996 |                0.2696 |
| (5.0, 7.0]   |  1090 |        7     |         0.5092 |                0.3743 |

### pattern_local_hops
| bin           |     n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|------:|-------------:|---------------:|----------------------:|
| (-0.001, 2.0] | 15342 |         1.66 |         0.3144 |                0.2036 |
| (2.0, 3.0]    |  4992 |         3    |         0.39   |                0.2388 |
| (3.0, 4.0]    |  2790 |         4    |         0.5022 |                0.3796 |

### pattern_web_hops
| bin           |     n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|------:|-------------:|---------------:|----------------------:|
| (-0.001, 1.0] | 17243 |       0.7322 |         0.353  |                0.2333 |
| (1.0, 2.0]    |  2685 |       2      |         0.2898 |                0.1739 |
| (2.0, 4.0]    |  3196 |       3.021  |         0.4089 |                0.2769 |

### query_total_tokens
| bin            |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:---------------|-----:|-------------:|---------------:|----------------------:|
| (-0.001, 26.0] | 5913 |        14.63 |         0.196  |                0.1228 |
| (26.0, 44.0]   | 5728 |        35.69 |         0.3499 |                0.2353 |
| (44.0, 70.0]   | 5737 |        55.59 |         0.415  |                0.2714 |
| (70.0, 817.0]  | 5746 |       117    |         0.4572 |                0.3033 |

### query_max_tokens
| bin           |    n |   mean_value |   leakage_rate |   strict_leakage_rate |
|:--------------|-----:|-------------:|---------------:|----------------------:|
| (-0.001, 9.0] | 6162 |        7.051 |         0.267  |                0.1725 |
| (12.0, 14.0]  | 3996 |       13.46  |         0.4092 |                0.269  |
| (14.0, 62.0]  | 5183 |       17.94  |         0.4233 |                0.2894 |
| (9.0, 12.0]   | 7783 |       10.98  |         0.3465 |                0.2231 |

## Post-Hoc Judge/Adversary Sanity Checks
These are not clean predictors of leakage because they are generated inside the privacy-evaluation pipeline, but they are useful for verifying that the score is behaving as expected.
| feature                          |     n |   mean_clean |   mean_leaked |   mean_diff_leaked_minus_clean |   point_biserial_r |   point_biserial_q |   single_feature_auc_directionless |
|:---------------------------------|------:|-------------:|--------------:|-------------------------------:|-------------------:|-------------------:|-----------------------------------:|
| n_primary_leaks_for_score        | 23124 |       0      |        1.644  |                         1.644  |             0.8349 |         0          |                             1      |
| n_primary_leaks                  | 23124 |       0      |        2.083  |                         2.083  |             0.7707 |         0          |                             1      |
| n_primary_claim_leaks_for_score  | 23124 |       0      |        1.053  |                         1.053  |             0.7168 |         0          |                             0.8759 |
| n_primary_claim_leaks            | 23124 |       0      |        1.491  |                         1.491  |             0.6434 |         0          |                             0.8759 |
| n_primary_answer_leaks_for_score | 23124 |       0      |        0.5915 |                         0.5915 |             0.5821 |         0          |                             0.7555 |
| n_primary_answer_leaks           | 23124 |       0      |        0.592  |                         0.592  |             0.5817 |         0          |                             0.7555 |
| n_full_answer_matches            | 23124 |       0      |        0.5753 |                         0.5753 |             0.5782 |         0          |                             0.7505 |
| correct_answer_count             | 23124 |       0      |        0.5753 |                         0.5753 |             0.5782 |         0          |                             0.7505 |
| n_partial_private_claims         | 23124 |       0      |        1.1    |                         1.1    |             0.529  |         0          |                             0.7715 |
| partial_claim_count              | 23124 |       0      |        1.1    |                         1.1    |             0.529  |         0          |                             0.7715 |
| matched_claim_count              | 23124 |       0      |        0.391  |                         0.391  |             0.4556 |         0          |                             0.6656 |
| n_full_private_claims            | 23124 |       0      |        0.391  |                         0.391  |             0.4556 |         0          |                             0.6656 |
| claim_attempt_count              | 23124 |       1.723  |        5.322  |                         3.599  |             0.2456 |         1.854e-314 |                             0.7732 |
| answer_attempt_count             | 23124 |       0.4069 |        1.045  |                         0.6385 |             0.2401 |         2.785e-300 |                             0.7198 |
| judge_warning_count              | 23124 |       0.1384 |        0.3763 |                         0.2379 |             0.2231 |         2.092e-258 |                             0.5842 |

## Files
- `joined_rows.csv`
- `numeric_correlations.csv`
- `adjusted_correlations.csv`
- `categorical_group_rates.csv`
- `quartile_rates.csv`
- `predictive_auc_experiments.csv`
- `posthoc_judge_correlations.csv`
- `per_model_correlations.csv`
- `within_model_meta_correlations.csv`
- `per_condition_correlations.csv`
- `within_condition_meta_correlations.csv`
- `input_warnings.json`
