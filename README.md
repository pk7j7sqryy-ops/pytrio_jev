----jev-----
微调网络安全人工智能模型——用于安全援助的决策model





# (data.py本文件数据处理流程)
# train.py的train里轮次的每一批数据的每行数据--调用-》data.py的build_sft_datum方法处理数据
#数据如下
# {
# 	"id": "tr_security_incidents_000178",
# 	"state": {
# 		"alert": {
# 			"description": "an unusual volume of files read in a short window",
# 			"evidence": "Between 2026-09-15 01:00:12 UTC and 01:04:58 UTC, the unprivileged CI runner service accessed 4,237 files in the /var/build/artifacts directory, reading a total of 12.3 GB of data. The source IP 198.51.100.77, which is not on the approved list, initiated the activity while MFA was enforced and the runner's credentials had been rotated on 2026-09-12. A scheduled change window was open during this period.",
# 			"rule": "mass_download"
# 		},
# 		"context": {
# 			"asset_criticality": "medium",
# 			"change_window_active": true,
# 			"source_on_allowlist": false
# 		},
# 		"history": {
# 			"credential_rotation_days_ago": 3,
# 			"distinct_countries_30d": 4,
# 			"logins_30d": 688,
# 			"prior_alerts_90d": 0
# 		},
# 		"principal": {
# 			"mfa_enrolled": true,
# 			"name": "svc-analytics-473",
# 			"privileges": ["repo:read"],
# 			"type": "ci_runner"
# 		}
# 	},
# 	"questions": {
# 		"credential_compromise": {
# 			"instructions": "The evidence indicates a credential or account has been compromised.",
# 			"type": "noul"
# 		},
# 		"disposition": {
# 			"criteria": {
# 				"close_benign": "Expected, explainable activity; close without analyst time.",
# 				"contain": "Contain the host or account immediately; do not wait for triage.",
# 				"investigate": "Warrants an analyst opening an investigation.",
# 				"monitor": "Not clearly malicious, but worth watching for recurrence."
# 			},
# 			"instructions": "How should this security alert be dispositioned?",
# 			"type": "choice"
# 		},
# 		"severity": {
# 			"criteria": ["Negligible: no access to anything sensitive.", "Low: limited access, easily reversed.", "Moderate: access to internal systems or non-public data.", "High: access to production, secrets or customer data.", "Critical: active compromise of crown-jewel systems."],
# 			"instructions": "How severe is the potential impact if this alert is real?",
# 			"type": "score"
# 		},
# 		"true_positive": {
# 			"criteria": {
# 				"false": "Benign activity, a misconfiguration, or a known false positive.",
# 				"true": "The underlying behaviour is malicious or unauthorised."
# 			},
# 			"instructions": "This alert reflects genuinely malicious or unauthorised activity.",
# 			"type": "noul"
# 		},
# 		"urgency": {
# 			"criteria": ["No time pressure; can wait indefinitely.", "Routine; handle within the normal queue.", "Elevated; should be handled within the same week.", "Critical; requires action within the same day."],
# 			"instructions": "How quickly must a responder act on this alert?",
# 			"type": "score"
# 		}
# 	},
# 	"targets": {
# 		"credential_compromise": {
# 			"label": "true"
# 		},
# 		"disposition": {
# 			"label": "investigate"
# 		},
# 		"severity": {
# 			"label": "2"
# 		},
# 		"true_positive": {
# 			"label": "true"
# 		},
# 		"urgency": {
# 			"label": "2"
# 		}
# 	}
# }

#1、先经过encoded = self.encode(record) 处理数据，得到的结果为
# encoded = {
#     "messages": [system+user（state原文+schema（A/B/C/D：选项内容）+assistant（只有占位符）],   # ① 来自 compile_record

            # schema内容：
                        # credential_compromise: The evidence indicates a credential or account has been compromised.
                        # A = no: ...
                        # B = yes: ...
                        # disposition: How should this security alert be dispositioned?
                        # A = close_benign: ...
                        # ...
                        # urgency: How quickly must a responder act on this alert?
                        # A = 0: ... D = 3: ...
                #   assistant
                # {
                #     "credential_compromise": "decision",
                #     "disposition": "decision",
                #     "severity": "decision",
                #     "true_positive": "decision",
                #     "urgency": "decision"
                # }

#     "fields":   ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],   # ② 来自 compile_record

#     "options":  
            "options": {
                "team": [
                    ("A", "billing", "Payments and refunds"),
                    ("B", "delivery", "Shipping and delivery"),
                    ("C", "technical", "Product technical support"),
                ],
                "urgency": [
                    ("A", "0", "Low: general inquiry"),
                    ("B", "1", "Medium: an issue requiring follow-up"),
                    ("C", "2", "High: a financial issue such as a duplicate charge"),
                ],
                "refund_requested": [
                    ("A", "no", "The answer is no (negative, or disagree with the claim)."),
                    ("B", "yes", "The answer is yes (affirmative, or align with the claim)."),
                ],
            }
{credential_compromise:
                dict[选项，[(A，原文选择，原文选择解释)]]
                    #选项A 具体选项 选项解释，
（for循环取值这样取 for xuanxiang，xuanze，jieshi in options[filed]）
                    #     A   no:  The answer is no (negative, or disagree with the claim).
                    #     B   yes: The answer is yes (affirmative, or align with the claim).
                    # disposition:。。。。。},   # ③ 来自 compile_record，真实 dump 出来的   A/B/C/D：选项内容

#     "targets":  {
                    #   "credential_compromise": "B",   # "true" → 归一 "yes" → 下标1 → B，假如是false 则是下标0，A
                    #   "disposition": "C",             # "investigate" → 下标2 → C
                    #   "severity": "C",                # "2" → 下标2 → C
                    #   "true_positive": "B",           # "true" → "yes" → B
                    #   "urgency": "C"                  # "2" → 下标2 → C
                    # },   # ④ 来自 compile_record

#     "ids":      [
                #   # 前 30 个（system 指令开头）：
                #   248045, 8678, 198, 2523, 513, 264, 16097, 5307, 17313, 13,
                #   5272, 279, 1528, 321, 5307, 10485, 303, 279, 1156, 1876,
                #   310, 1236, 279, 10897, 10856, 13, 1690, 1396, 2002, 11,
                #   # …… 中间是 state 案情 + schema 选项 ……
                #   # 后 30 个（骨架尾部）：
                #   198, 262, 328, 74385, 763, 328, 61781, 487, 198, 262,
                #   328, 1802, 52339, 763, 328, 61781, 487, 198, 262, 328,
                #   5382, 2179, 763, 328, 61781, 1, 198, 92, 248046, 198,
                # ],   # ⑤ 来自 encode
                #还不是最终输入模型的，248045 / 248046 = <|im_start|> / <|im_end|>（chat template 的控制符）
                                   #这个需要替换 ---》 61781 = decision（占位符 marker）

#     "markers":  {
# #         "credential_compromise": 836,
# #         "disposition": 845,
# #         "severity": 853,
# #         "true_positive": 862,
# #         "urgency": 871,
# #     },   # ⑥ 来自 encode
# }







# encoded = self.encode(record)
# encoded: dict
# ├── messages: list[dict]   对话文本
# ├── fields: list[str]      问题字段名
# ├── options: dict          每个字段的选项及符号
# ├── targets: dict          正确答案符号
# ├── ids: list[int]         对话 token 序列
# └── markers: dict[str,int] 每个答案占位符的位置

# 举例encoded = {
#     "ids": [ ... 877 个 token id ... ],
#     "markers": {
#         "credential_compromise": 836,
#         "disposition": 845,
#         "severity": 853,
#         "true_positive": 862,
#         "urgency": 871,
#     },
#     "targets": {
#         "credential_compromise": "B",
#         "disposition": "C",
#         "severity": "C",
#         "true_positive": "B",
#         "urgency": "C",
#     },
#     "fields": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
#     "options": { ... },
# }


# decision 占位符是用来「标记位置」的，不是用来「当答案」的。
#模型最终要ids[:-1]和ids[1:]以及weight，ids[:-1]和ids[1:]都好确定，但是weight不好确定，
# 所以需要答案的具体位置，和代替符号A，B，C（实际训练后也是输出这些为结果），所以markers作用
# 作用1：是替换为abc，不在是true、选项，分数。
# 作用2：是告诉weight哪个位置的权重为1，好只计算这些loss



# 具体拆成两个原因：
#
# 1. 答案符号不能直接写进 assistant 骨架
#
# 如果一开始就在 assistant 里写 {"disposition": "C", ...}，那 ids 里 C 的位置已经是正确答案了。但这样的话：
#
# 训练时模型会在输入里直接看到答案，等于泄题；
# 而且你没法区分「这里是要监督的位置」和「别的地方恰好也出现字母 C」——state 描述里到处是单词和字符，C、B 这种单字符 token 在正文里满天飞。
# 2. decision 是唯一的「定位锚点」
# decision 这个词在 Qwen3.5 词表里是单个 token（id 61781）​，而且正常语料里几乎不会在 state/instructions 里作为独立 token 出现。所以：
# encode 阶段：在骨架 token 里扫 decision 的命中位置，就能精确知道「每个字段的答案应该落在哪」→ 得到 markers。
# build_sft_datum 阶段：拿 markers[field] - 1 直接算出 weight=1 的位置，把那里的 target 从 61781 覆盖成 B/C 的正确符号 id。
# 整个过程 decision 只扮演「占位 + 定位」两个角色，最后被覆盖掉，从不作为监督目标出现（这点上一条已验证：模型在答案位置看到的是前面的 " 引号，不是 decision）。


#实际训练输入-转tokenid之前：
# <|im_start|>system
# You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.
# <|im_end|>
# <|im_start|>user
# Return one answer for every field using the supplied answer symbols.
#
# ## State
# {
#   "alert": { "description": "...", "evidence": "...", "rule": "mass_download" },
#   "context": { "asset_criticality": "medium", "change_window_active": true, "source_on_allowlist": false },
#   "history": { "credential_rotation_days_ago": 3, "distinct_countries_30d": 4, "logins_30d": 688, "prior_alerts_90d": 0 },
#   "principal": { "mfa_enrolled": true, "name": "svc-analytics-473", "privileges": ["repo:read"], "type": "ci_runner" }
# }
# ## Decision schema
# credential_compromise: The evidence indicates a credential or account has been compromised.
#     A = no: ...
#     B = yes: ...
# disposition: How should this security alert be dispositioned?
#     A = close_benign: ...
#     B = contain: ...
#     C = investigate: ...
#     D = monitor: ...
# severity: How severe is the potential impact if this alert is real?
#     A = 0: ...  B = 1: ...  C = 2: ...  D = 3: ...  E = 4: ...
# true_positive: This alert reflects genuinely malicious or unauthorised activity.
#     A = no: ...
#     B = yes: ...
# urgency: How quickly must a responder act on this alert?
#     A = 0: ...  B = 1: ...  C = 2: ...  D = 3: ...
# <|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# {
#     "credential_compromise": "decision",
#     "disposition": "decision",
#     "severity": "decision",
#     "true_positive": "decision",
#     "urgency": "decision"
# }


#对比的target
# <|im_start|>system
# You are a careful decision assistant. ...
# <|im_end|>
# <|im_start|>user
# ... state + schema ...
# <|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# {
#     "credential_compromise": "B",      ← decision → B
#     "disposition": "C",                ← decision → C
#     "severity": "C",                   ← decision → C
#     "true_positive": "B",              ← decision → B
#     "urgency": "C"                     ← decision → C
# }
# <|im_end|>

# 位置:     ...  835         836           837         838   ...
# 模型读:   ...  引号"      decision      引号"        逗号,   ...
# 模型预测:      B          → 引号"      → 逗号,       → ...
# weight:        1.0 ✅      0.0 ❌        0.0 ❌       0.0 ❌

#为什么中间引号闭合不算loss
#见过，但不专门算这部分」 —— 精确地讲：
# 1、因为只有一次前向传播 不需要闭合 只用预测一个值

#为什么要用占位符号去训练，不就只用错位去训练？
#1、只用错位去训练，全文都监督：省事但把训练信号浪费在「学格式」上，决策能力被稀释；
#2、只想监督答案：得自己想办法在 token 流里定位「哪些是答案符号」，而 decision 占位符正是干这个的——用一个唯一 token 当坐标，让「只监督 5 个符号」这件事从「难」变「一行代码」。
占位符不是训练的必要条件，而是「稀疏监督」的工程实现手段。你不用它，训练照跑，只是没法那么干净地只盯答案符号。


encoded = encoder.encode(record, include_targets=False)
print(encoded["messages"])
print(encoded["ids"])
tokenizer = sampling_client.get_tokenizer()
ids = tokenizer.encode(
    tokenizer.apply_chat_template(
        encoded["messages"], chat_template=encoder.template, tokenize=False
    ),
    add_special_tokens=False,
)
print(ids)

# [{'role': 'system', 'content': 'You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.'}, {'role': 'user', 'content': 'Return one answer for every field using the supplied answer symbols.\n\n## State\n{\n  "customer_message": "I bought headphones yesterday and noticed today that I was charged twice. Please refund the extra charge as soon as possible."\n}\n## Decision schema\nteam: Which team should handle this ticket?\n    A = billing: Payments and refunds\n    B = delivery: Shipping and delivery\n    C = technical: Product technical support\nurgency: Choose the priority for handling this ticket.\n    A = 0: Low: general inquiry\n    B = 1: Medium: an issue requiring follow-up\n    C = 2: High: a financial issue such as a duplicate charge\nrefund_requested: Does the customer explicitly request a refund?\n    A = no: The answer is no (negative, or disagree with the claim).\n    B = yes: The answer is yes (affirmative, or align with the claim).'}, {'role': 'assistant', 'content': '{\n    "team": "decision",\n    "urgency": "decision",\n    "refund_requested": "decision"\n}'}]
[248045, 8678, 198, 2523, 513, 264, 16097, 5307, 17313, 13, 5272, 279, 1528, 321, 5307, 10485, 303, 279, 1156, 1876, 310, 1236, 279, 10897, 10856, 13, 1690, 1396, 2002, 11, 4992, 6681, 799, 4087, 7489, 318, 68, 1257, 13, 357, 11, 417, 11, 351, 11, 45347, 494, 1141, 9711, 2519, 321, 460, 799, 2610, 4566, 1576, 12366, 1754, 2002, 803, 310, 1141, 11543, 7489, 13, 5272, 279, 2002, 4874, 321, 17210, 6681, 430, 2574, 13, 3054, 524, 2830, 39492, 11, 70703, 11, 466, 4799, 1414, 13, 248046, 198, 248045, 846, 198, 5423, 799, 4087, 364, 1396, 2002, 1608, 279, 16713, 4087, 17210, 13, 271, 550, 3130, 198, 90, 198, 220, 328, 10726, 6260, 763, 328, 40, 10471, 41581, 13278, 321, 13293, 3242, 421, 353, 557, 11102, 10598, 13, 5044, 20325, 279, 4799, 6545, 430, 4970, 430, 3102, 1149, 198, 92, 198, 550, 39087, 10485, 198, 8923, 25, 15451, 2015, 1220, 3579, 411, 11390, 30, 198, 262, 357, 283, 32419, 25, 55935, 321, 70057, 198, 262, 417, 283, 9403, 25, 23206, 321, 9403, 198, 262, 351, 283, 10597, 25, 5466, 10597, 1761, 198, 5382, 2179, 25, 21513, 279, 10304, 364, 11256, 411, 11390, 13, 198, 262, 357, 283, 220, 15, 25, 11698, 25, 4437, 25066, 198, 262, 417, 283, 220, 16, 25, 23587, 25, 449, 4125, 21998, 1732, 5072, 198, 262, 351, 283, 220, 17, 25, 4962, 25, 264, 5713, 4125, 1680, 430, 264, 21814, 6545, 198, 66165, 70238, 25, 12193, 279, 5813, 20335, 1622, 264, 20325, 30, 198, 262, 357, 283, 874, 25, 561, 4087, 369, 874, 318, 40823, 11, 466, 27376, 440, 279, 3591, 553, 198, 262, 417, 283, 9542, 25, 561, 4087, 369, 9542, 318, 2562, 2760, 1340, 11, 466, 5117, 440, 279, 3591, 553, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 90, 198, 262, 328, 8923, 763, 328, 61781, 487, 198, 262, 328, 5382, 2179, 763, 328, 61781, 487, 198, 262, 328, 66165, 70238, 763, 328, 61781, 1, 198, 92, 248046, 198]
2026-11-26 11:33:18,190 | WARNING | modelscope_hub.download | Cannot confirm the cached file is for revision: master
[248045, 8678, 198, 2523, 513, 264, 16097, 5307, 17313, 13, 5272, 279, 1528, 321, 5307, 10485, 303, 279, 1156, 1876, 310, 1236, 279, 10897, 10856, 13, 1690, 1396, 2002, 11, 4992, 6681, 799, 4087, 7489, 318, 68, 1257, 13, 357, 11, 417, 11, 351, 11, 45347, 494, 1141, 9711, 2519, 321, 460, 799, 2610, 4566, 1576, 12366, 1754, 2002, 803, 310, 1141, 11543, 7489, 13, 5272, 279, 2002, 4874, 321, 17210, 6681, 430, 2574, 13, 3054, 524, 2830, 39492, 11, 70703, 11, 466, 4799, 1414, 13, 248046, 198, 248045, 846, 198, 5423, 799, 4087, 364, 1396, 2002, 1608, 279, 16713, 4087, 17210, 13, 271, 550, 3130, 198, 90, 198, 220, 328, 10726, 6260, 763, 328, 40, 10471, 41581, 13278, 321, 13293, 3242, 421, 353, 557, 11102, 10598, 13, 5044, 20325, 279, 4799, 6545, 430, 4970, 430, 3102, 1149, 198, 92, 198, 550, 39087, 10485, 198, 8923, 25, 15451, 2015, 1220, 3579, 411, 11390, 30, 198, 262, 357, 283, 32419, 25, 55935, 321, 70057, 198, 262, 417, 283, 9403, 25, 23206, 321, 9403, 198, 262, 351, 283, 10597, 25, 5466, 10597, 1761, 198, 5382, 2179, 25, 21513, 279, 10304, 364, 11256, 411, 11390, 13, 198, 262, 357, 283, 220, 15, 25, 11698, 25, 4437, 25066, 198, 262, 417, 283, 220, 16, 25, 23587, 25, 449, 4125, 21998, 1732, 5072, 198, 262, 351, 283, 220, 17, 25, 4962, 25, 264, 5713, 4125, 1680, 430, 264, 21814, 6545, 198, 66165, 70238, 25, 12193, 279, 5813, 20335, 1622, 264, 20325, 30, 198, 262, 357, 283, 874, 25, 561, 4087, 369, 874, 318, 40823, 11, 466, 27376, 440, 279, 3591, 553, 198, 262, 417, 283, 9542, 25, 561, 4087, 369, 9542, 318, 2562, 2760, 1340, 11, 466, 5117, 440, 279, 3591, 553, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271, 90, 198, 262, 328, 8923, 763, 328, 61781, 487, 198, 262, 328, 5382, 2179, 763, 328, 61781, 487, 198, 262, 328, 66165, 70238, 763, 328, 61781, 1, 198, 92, 248046, 198]


#推理过程
#一次前向，拿到每个marker位置的概率，然后每个位置取topk，然后过滤后，在转softmax，在取值
发生在「拿到 logprob」之后、「softmax」之前