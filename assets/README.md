# 发布资产

四份小型评估Parquet与manifest纳入源码。Student/Judge位于assets/models/，共同单头adv初始化位于assets/initialization/，权重和17k数据通过资产包交付，不进入普通Git。

baseline/correctness_only只需要Student、共同初始化和17k V2数据；rubric_joint另需Qwen3-8B Judge。sample模板中的Teacher按需另行提供；训练已有Teacher文本时不需要Teacher模型。

共同初始化从原runs/_initialization逐文件复制，不重新随机初始化。旧实验快照/缓存继续使用原路径。模型复制或外部路径由显式机器配置指定；完整发布包解开后不需要../model、../OPRD或历史runs。
