# 发布流程

1. 配置 public.json

gitmodule_updates 设置 gitmodule path 到外部 git 仓地址的映射
gitmodule_refs 设置 gitmodule path 到外部 git 仓 commit 的映射

2. 执行 public.py

脚本会自动创建 github-public 分支, 删除指定文件, 修改 gitmodule, 并把修改放到暂存区

3. 提交并 force push

