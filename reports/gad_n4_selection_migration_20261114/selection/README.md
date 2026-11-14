# n=4 Actor-rubric 离线选择

状态：await_user_confirmation

只在选择集选参；复核集和 step450–542 单独报告。完整结果见 selection.json。

|分量|w|τ|选择集净修正|C−W 优势差增加|满足约束|复核通过|后期通过|
|---|---|---|---|---|---|---|---|
|full|0.25|0.233532|37|0.0104565|False|False|False|
|full|0.5|0.233532|41|0.0182152|False|False|False|
|full|1|0.233532|60|0.0280985|False|False|False|
|full|2|0.233532|65|0.0313994|False|False|False|
|c_only|0.25|0.0459922|153|0.0131155|True|True|True|
|c_only|0.5|0.0459922|254|0.0218956|False|False|False|
|c_only|1|0.0459922|433|0.0317041|False|False|False|
|c_only|2|0.0459922|634|0.0373679|False|False|False|

建议：{"components": "c_only", "weight": 0.25, "tau": 0.045992180705070496, "options": {"reward_source": "adv_plus_rubric_groupnorm", "actor_reward_weights": {"answer_correctness": 1.0, "verified_progress": 0.0, "unresolved_error_severity": 0.0}, "actor_reward_group_weight": 0.25, "actor_reward_rubric_std_floor": 0.045992180705070496}, "selection": {"samples": 34784, "groups": 8696, "corrected": 204, "damaged": 51, "net_corrections": 153, "correction_precision": 0.8, "cw_net_corrections": 25, "tw_toward_T": 27, "tw_toward_W": 605, "tw_toward_T_share": 0.04272151898734177, "unequal_adv_pairs": 52098, "order_changes": 1927, "order_change_rate": 0.036987984183653885, "cw_strict_flip_rate": 0.02838482793268023, "ct_strict_flip_rate": 0.04152046783625731, "cw_gap_increase": 0.013115499843814493, "mainly_spacing": false, "adv_weak_groups": 0, "adv_weak_advantage_change_mean": null, "constraints": {"1": true, "2": true, "3": true, "4": true, "5": true}, "eligible": true, "diagnostics": {"groups/mixed_correctness_count": 2285, "groups/all_truncated_count": 2487, "groups/all_correct_count": 207, "groups/zero_advantage_count": 0, "groups/passthrough_count": 0, "groups/rubric_zero_std_count": 0, "groups/adv_weak_count": 0, "groups/adv_weak_advantage_change_defined": 0, "groups/cw_gap_defined": 1, "groups/cw_gap_count": 1618, "groups/below_floor_strict_flips": 344, "groups/below_floor_flip_share_defined": 1, "groups/mixed_correctness_rate": 0.2627644894204232, "groups/all_truncated_rate": 0.2859935602575897, "groups/all_correct_rate": 0.023804047838086477, "groups/zero_advantage_rate": 0.0, "groups/cw_gap_adv": 0.2144279235557219, "groups/cw_gap_final": 0.2275434233995364, "groups/cw_gap_increase": 0.013115499843814493, "groups/below_floor_flip_share": 0.1785158277114686, "groups/rubric_std_q0": 0.00017218600260093808, "groups/rubric_std_q25": 0.045992180705070496, "groups/rubric_std_q50": 0.11123654246330261, "groups/rubric_std_q75": 0.20608273148536682, "groups/rubric_std_q100": 0.5117628574371338, "category/C/count": 4407, "category/C/advantage_adv_mean": 0.22749635577201843, "category/C/advantage_final_mean": 0.25535696744918823, "category/W/count": 10618, "category/W/advantage_adv_mean": 0.0993010401725769, "category/W/advantage_final_mean": 0.14413852989673615, "category/T/count": 19729, "category/T/advantage_adv_mean": -0.10356731712818146, "category/T/advantage_final_mean": -0.13397063314914703}}, "review_pass": true, "late_pass": true}

需用户确认后再将分量、w、τ 写入第三组配置并将 selection_confirmed 设为 true。
