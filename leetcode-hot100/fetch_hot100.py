#!/usr/bin/env python3
"""
LeetCode Hot 100 本地练习环境生成器
从 LeetCode CN GraphQL API 获取热题100的题目描述和C++初始代码
"""

import requests
import json
import os
import time
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent / "problems"
LEETCODE_GRAPHQL = "https://leetcode.cn/graphql/"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://leetcode.cn",
    "Origin": "https://leetcode.cn",
}

# ────────────────────────────────────────────────────────
# 硬编码的 Hot 100 题目列表（按力扣官方分类），作为 API 失败时的 fallback
# ────────────────────────────────────────────────────────
HOT100_FALLBACK = [
    # 哈希
    (1, "two-sum", "哈希"),
    (49, "group-anagrams", "哈希"),
    (128, "longest-consecutive-sequence", "哈希"),
    # 双指针
    (283, "move-zeroes", "双指针"),
    (11, "container-with-most-water", "双指针"),
    (15, "3sum", "双指针"),
    (42, "trapping-rain-water", "双指针"),
    # 滑动窗口
    (3, "longest-substring-without-repeating-characters", "滑动窗口"),
    (438, "find-all-anagrams-in-a-string", "滑动窗口"),
    # 子串
    (560, "subarray-sum-equals-k", "子串"),
    (239, "sliding-window-maximum", "子串"),
    (76, "minimum-window-substring", "子串"),
    # 普通数组
    (53, "maximum-subarray", "普通数组"),
    (56, "merge-intervals", "普通数组"),
    (189, "rotate-array", "普通数组"),
    (238, "product-of-array-except-self", "普通数组"),
    (41, "first-missing-positive", "普通数组"),
    # 矩阵
    (73, "set-matrix-zeroes", "矩阵"),
    (54, "spiral-matrix", "矩阵"),
    (48, "rotate-image", "矩阵"),
    (240, "search-a-2d-matrix-ii", "矩阵"),
    # 链表
    (160, "intersection-of-two-linked-lists", "链表"),
    (206, "reverse-linked-list", "链表"),
    (234, "palindrome-linked-list", "链表"),
    (141, "linked-list-cycle", "链表"),
    (142, "linked-list-cycle-ii", "链表"),
    (21, "merge-two-sorted-lists", "链表"),
    (2, "add-two-numbers", "链表"),
    (19, "remove-nth-node-from-end-of-list", "链表"),
    (24, "swap-nodes-in-pairs", "链表"),
    (25, "reverse-nodes-in-k-group", "链表"),
    (138, "copy-list-with-random-pointer", "链表"),
    (148, "sort-list", "链表"),
    (23, "merge-k-sorted-lists", "链表"),
    (146, "lru-cache", "链表"),
    # 二叉树
    (94, "binary-tree-inorder-traversal", "二叉树"),
    (104, "maximum-depth-of-binary-tree", "二叉树"),
    (226, "invert-binary-tree", "二叉树"),
    (101, "symmetric-tree", "二叉树"),
    (543, "diameter-of-binary-tree", "二叉树"),
    (102, "binary-tree-level-order-traversal", "二叉树"),
    (108, "convert-sorted-array-to-binary-search-tree", "二叉树"),
    (98, "validate-binary-search-tree", "二叉树"),
    (230, "kth-smallest-element-in-a-bst", "二叉树"),
    (199, "binary-tree-right-side-view", "二叉树"),
    (114, "flatten-binary-tree-to-linked-list", "二叉树"),
    (105, "construct-binary-tree-from-preorder-and-inorder-traversal", "二叉树"),
    (437, "path-sum-iii", "二叉树"),
    (236, "lowest-common-ancestor-of-a-binary-tree", "二叉树"),
    (124, "binary-tree-maximum-path-sum", "二叉树"),
    # 图论
    (200, "number-of-islands", "图论"),
    (994, "rotting-oranges", "图论"),
    (207, "course-schedule", "图论"),
    (208, "implement-trie-prefix-tree", "图论"),
    # 回溯
    (46, "permutations", "回溯"),
    (78, "subsets", "回溯"),
    (17, "letter-combinations-of-a-phone-number", "回溯"),
    (39, "combination-sum", "回溯"),
    (22, "generate-parentheses", "回溯"),
    (79, "word-search", "回溯"),
    (131, "palindrome-partitioning", "回溯"),
    (51, "n-queens", "回溯"),
    # 二分查找
    (35, "search-insert-position", "二分查找"),
    (74, "search-a-2d-matrix", "二分查找"),
    (34, "find-first-and-last-position-of-element-in-sorted-array", "二分查找"),
    (33, "search-in-rotated-sorted-array", "二分查找"),
    (153, "find-minimum-in-rotated-sorted-array", "二分查找"),
    (4, "median-of-two-sorted-arrays", "二分查找"),
    # 栈
    (20, "valid-parentheses", "栈"),
    (155, "min-stack", "栈"),
    (394, "decode-string", "栈"),
    (739, "daily-temperatures", "栈"),
    (84, "largest-rectangle-in-histogram", "栈"),
    # 堆
    (215, "kth-largest-element-in-an-array", "堆"),
    (347, "top-k-frequent-elements", "堆"),
    (295, "find-median-from-data-stream", "堆"),
    # 贪心算法
    (121, "best-time-to-buy-and-sell-stock", "贪心算法"),
    (55, "jump-game", "贪心算法"),
    (45, "jump-game-ii", "贪心算法"),
    (763, "partition-labels", "贪心算法"),
    # 动态规划
    (70, "climbing-stairs", "动态规划"),
    (118, "pascals-triangle", "动态规划"),
    (198, "house-robber", "动态规划"),
    (279, "perfect-squares", "动态规划"),
    (322, "coin-change", "动态规划"),
    (139, "word-break", "动态规划"),
    (300, "longest-increasing-subsequence", "动态规划"),
    (152, "maximum-product-subarray", "动态规划"),
    (416, "partition-equal-subset-sum", "动态规划"),
    (32, "longest-valid-parentheses", "动态规划"),
    # 多维动态规划
    (62, "unique-paths", "多维动态规划"),
    (64, "minimum-path-sum", "多维动态规划"),
    (5, "longest-palindromic-substring", "多维动态规划"),
    (1143, "longest-common-subsequence", "多维动态规划"),
    (72, "edit-distance", "多维动态规划"),
    # 技巧
    (136, "single-number", "技巧"),
    (169, "majority-element", "技巧"),
    (75, "sort-colors", "技巧"),
    (31, "next-permutation", "技巧"),
    (287, "find-the-duplicate-number", "技巧"),
]


def graphql_request(query, variables=None):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    for attempt in range(3):
        try:
            resp = requests.post(
                LEETCODE_GRAPHQL, json=payload, headers=HEADERS, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                return None
            return data.get("data")
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return None


def get_hot100_from_api():
    """尝试从力扣学习计划 API 获取 Hot 100 列表"""
    query = """
    query studyPlanV2Detail($planSlug: String!) {
        studyPlanV2Detail(planSlug: $planSlug) {
            slug
            name
            planSubGroups {
                slug
                name
                questions {
                    titleSlug
                    title
                    translatedTitle
                    frontendQuestionId
                    difficulty
                }
            }
        }
    }
    """
    data = graphql_request(query, {"planSlug": "top-100-liked"})
    if not data or not data.get("studyPlanV2Detail"):
        return None

    plan = data["studyPlanV2Detail"]
    problems = []
    for group in plan.get("planSubGroups", []):
        cat = group.get("name", "")
        for q in group.get("questions", []):
            problems.append(
                {
                    "id": int(q["frontendQuestionId"]),
                    "slug": q["titleSlug"],
                    "title": q.get("translatedTitle") or q["title"],
                    "difficulty": q.get("difficulty", ""),
                    "category": cat,
                }
            )
    return problems if problems else None


def get_hot100_problems():
    print("  尝试从力扣 API 获取 Hot 100 列表...", end=" ", flush=True)
    problems = get_hot100_from_api()
    if problems:
        print(f"成功 ({len(problems)} 题)")
        return problems

    print("失败，使用内置列表")
    return [
        {"id": pid, "slug": slug, "title": "", "difficulty": "", "category": cat}
        for pid, slug, cat in HOT100_FALLBACK
    ]


def get_problem_detail(title_slug):
    query = """
    query questionData($titleSlug: String!) {
        question(titleSlug: $titleSlug) {
            questionFrontendId
            titleSlug
            title
            translatedTitle
            translatedContent
            content
            difficulty
            codeSnippets {
                lang
                langSlug
                code
            }
            exampleTestcaseList
            topicTags {
                name
                translatedName
            }
        }
    }
    """
    data = graphql_request(query, {"titleSlug": title_slug})
    if data and data.get("question"):
        return data["question"]
    return None


def html_to_markdown(html):
    if not html:
        return ""
    text = html
    text = re.sub(r"<strong>(.*?)</strong>", r"**\1**", text, flags=re.S)
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.S)
    text = re.sub(r"<em>(.*?)</em>", r"*\1*", text, flags=re.S)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.S)
    text = re.sub(r"<pre>\s*", "\n```\n", text)
    text = re.sub(r"\s*</pre>", "\n```\n", text)
    text = re.sub(r"<ul>", "", text)
    text = re.sub(r"</ul>", "\n", text)
    text = re.sub(r"<ol>", "", text)
    text = re.sub(r"</ol>", "\n", text)
    text = re.sub(r"<li>\s*", "- ", text)
    text = re.sub(r"\s*</li>", "\n", text)
    text = re.sub(r"<p>", "\n", text)
    text = re.sub(r"</p>", "\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    for i in range(1, 7):
        text = re.sub(f"<h{i}>(.*?)</h{i}>", f'{"#" * i} \\1\n', text, flags=re.S)
    text = re.sub(r"<sup>(.*?)</sup>", r"^(\1)", text, flags=re.S)
    text = re.sub(r"<sub>(.*?)</sub>", r"_(\1)", text, flags=re.S)
    text = re.sub(r'<img[^>]*src="([^"]*)"[^>]*/?\s*>', r"![image](\1)", text)
    text = re.sub(r"<[^>]+>", "", text)
    for entity, char in [
        ("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
        ("&quot;", '"'), ("&#39;", "'"), ("&le;", "≤"), ("&ge;", "≥"),
        ("&times;", "×"), ("&minus;", "−"),
    ]:
        text = text.replace(entity, char)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


DIFF_BADGE = {"Easy": "🟢 简单", "Medium": "🟡 中等", "Hard": "🔴 困难"}


def generate_readme(detail, category=""):
    qid = detail["questionFrontendId"]
    title = detail.get("translatedTitle") or detail["title"]
    difficulty = detail.get("difficulty", "")
    tags = [
        t.get("translatedName") or t.get("name", "")
        for t in detail.get("topicTags", [])
        if t.get("translatedName") or t.get("name")
    ]
    desc = html_to_markdown(
        detail.get("translatedContent") or detail.get("content", "")
    )

    lines = [
        f"# {qid}. {title}",
        "",
        f"**难度：** {DIFF_BADGE.get(difficulty, difficulty)}  ",
        f"**链接：** [LeetCode](https://leetcode.cn/problems/{detail['titleSlug']}/)  ",
        f"**标签：** {', '.join(tags) if tags else '—'}  ",
    ]
    if category:
        lines.append(f"**分类：** {category}")
    lines += ["", "---", "", "## 题目描述", "", desc, ""]
    return "\n".join(lines)


COMMON_STRUCTS = {
    "ListNode": """
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
""",
    "TreeNode": """
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
""",
    "Node": """
class Node {
public:
    int val;
    Node* next;
    Node* random;
    Node(int _val) : val(_val), next(nullptr), random(nullptr) {}
};
""",
}


def detect_class_name(cpp_code):
    """从代码片段中提取主类名（Solution / LRUCache / Trie 等）"""
    m = re.search(r"^class\s+(\w+)\s*\{", cpp_code, re.MULTILINE)
    return m.group(1) if m else "Solution"


def generate_solution_cpp(detail):
    cpp_code = ""
    for snippet in detail.get("codeSnippets", []):
        if snippet["langSlug"] == "cpp":
            cpp_code = snippet["code"]
            break

    if not cpp_code:
        cpp_code = "// 此题无 C++ 代码模板\n"

    qid = detail["questionFrontendId"]
    title = detail.get("translatedTitle") or detail["title"]
    slug = detail["titleSlug"]

    structs_needed = []
    for name, code in COMMON_STRUCTS.items():
        if re.search(rf"\b{name}\b", cpp_code):
            structs_needed.append(code)

    cls = detect_class_name(cpp_code)

    parts = [
        f"// {qid}. {title}",
        f"// https://leetcode.cn/problems/{slug}/",
        "",
        "#include <iostream>",
        "#include <vector>",
        "#include <string>",
        "#include <unordered_map>",
        "#include <unordered_set>",
        "#include <algorithm>",
        "#include <stack>",
        "#include <queue>",
        "#include <climits>",
        "#include <numeric>",
        "using namespace std;",
        "",
    ]
    for s in structs_needed:
        parts.append(s)

    parts.append(cpp_code)
    parts.append("")
    parts.append("int main() {")
    if cls == "Solution":
        parts.append(f"    {cls} sol;")
    else:
        parts.append(f"    // {cls} obj(...);")
    parts.append("    // TODO: 编写测试用例")
    parts.append('    cout << "Tests passed!" << endl;')
    parts.append("    return 0;")
    parts.append("}")
    parts.append("")
    return "\n".join(parts)


def main():
    print("=" * 60)
    print("  LeetCode Hot 100 — 本地练习环境生成器")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 获取题目列表
    print("\n[1/3] 获取题目列表...")
    problems = get_hot100_problems()
    print(f"  共 {len(problems)} 道题\n")

    # 2. 逐题拉取
    print("[2/3] 获取题目详情并生成文件...")
    success, skipped, failed = 0, 0, []

    for i, prob in enumerate(problems):
        qid = str(prob["id"]).zfill(4)
        slug = prob["slug"]
        label = prob.get("title") or slug
        category = prob.get("category", "")

        prob_dir = BASE_DIR / f"{qid}_{slug}"
        readme_path = prob_dir / "README.md"
        solution_path = prob_dir / "solution.cpp"

        if readme_path.exists() and solution_path.exists():
            print(f"  [{i+1:3d}/{len(problems)}] {qid} {label} — 跳过（已存在）")
            skipped += 1
            success += 1
            continue

        print(f"  [{i+1:3d}/{len(problems)}] {qid} {label} ...", end=" ", flush=True)

        detail = get_problem_detail(slug)
        if not detail:
            print("✗")
            failed.append(f"{qid} {slug}")
            time.sleep(1)
            continue

        if not label or label == slug:
            label = detail.get("translatedTitle") or detail.get("title", slug)

        prob_dir.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(generate_readme(detail, category), encoding="utf-8")
        solution_path.write_text(generate_solution_cpp(detail), encoding="utf-8")

        print("✓")
        success += 1
        time.sleep(0.6)

    # 3. 汇总
    print(f"\n[3/3] 完成!")
    print(f"  成功: {success}  (其中跳过已存在: {skipped})")
    if failed:
        print(f"  失败: {len(failed)}")
        for name in failed:
            print(f"    - {name}")
        print("  可重新运行此脚本重试失败项")
    print(f"\n  文件位置: {BASE_DIR.resolve()}")


if __name__ == "__main__":
    main()
