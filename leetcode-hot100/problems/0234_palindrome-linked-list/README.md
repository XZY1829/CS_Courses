# 234. 回文链表

**难度：** 🟢 简单  
**链接：** [LeetCode](https://leetcode.cn/problems/palindrome-linked-list/)  
**标签：** 栈, 递归, 链表, 双指针  
**分类：** 链表

---

## 题目描述

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

![image](https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg)

```
**输入：**head = [1,2,2,1]
**输出：**true
```

**示例 2：**

![image](https://assets.leetcode.com/uploads/2021/03/03/pal2linked-list.jpg)

```
**输入：**head = [1,2]
**输出：**false
```

 

**提示：**

	- 链表中节点数目在范围`[1, 10^(5)]` 内

	- `0 <= Node.val <= 9`

 

**进阶：**你能否用 `O(n)` 时间复杂度和 `O(1)` 空间复杂度解决此题？
