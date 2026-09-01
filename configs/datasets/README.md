# Dataset 配置

活动代码中的数据集 alias 必须在这里显式配置，不能写进 Python registry。

每个配置至少包含：

```json
{
  "schema_version": 1,
  "name": "example_train",
  "annotation_path": "/absolute/path/train.jsonl",
  "media_root": "/absolute/path/media",
  "split": "train"
}
```

规则：

- `annotation_path` 和 `media_root` 必须是运行环境中真实存在的绝对路径。
- train 与 validation 使用不同 alias 和不同 annotation 文件。
- 配置不含密码、token 或远端连接信息。
- production 运行必须记录配置文件 hash、数据文件 hash 和媒体 manifest hash。
- 本目录只提交模板；机器专属路径配置放在受控工作区，不提交到代码仓库。
