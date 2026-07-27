# 108. 将有序数组转换为二叉搜索树

**难度：** 🟢 简单  
**链接：** [LeetCode](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)  
**标签：** 树, 二叉搜索树, 数组, 分治, 二叉树  
**分类：** 二叉树

---

## 题目描述

给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵 平衡 二叉搜索树。

 

**示例 1：**

![image](https://assets.leetcode.com/uploads/2021/02/18/btree1.jpg)

```
**输入：**nums = [-10,-3,0,5,9]
**输出：**[0,-3,9,-10,null,5]
**解释：**[0,-10,5,null,-3,null,9] 也将被视为正确答案：
![image](https://assets.leetcode.com/uploads/2021/02/18/btree2.jpg)
```

**示例 2：**

![image](https://assets.leetcode.com/uploads/2021/02/18/btree.jpg)

```
**输入：**nums = [1,3]
**输出：**[3,1]
**解释：**[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
```

 

**提示：**

	- `1 <= nums.length <= 10^(4)`

	- `-10^(4) <= nums[i] <= 10^(4)`

	- `nums` 按 **严格递增** 顺序排列
