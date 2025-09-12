#!/bin/bash

echo "Git更改管理脚本"
echo "=================="

echo "当前stash状态:"
git stash list

echo ""
echo "可用的操作:"
echo "1. 查看stash内容: git stash show stash@{0}"
echo "2. 恢复修改: git stash pop"
echo "3. 应用修改但不删除stash: git stash apply"
echo "4. 删除stash: git stash drop"
echo "5. 清理所有stash: git stash clear"
echo ""
echo "当前git状态:"
git status

echo ""
echo "使用示例:"
echo "恢复修改: git stash pop"
echo "查看修改内容: git stash show -p stash@{0}"