#include <iostream>
#include <string>
#include <vector>

using namespace std;

class TrieNode
{
    friend class Trie;

public:
    TrieNode(char ch)
    {
        this->val = ch;
    };

    TrieNode(char ch, const bool end)
    {
        this->val = ch;
        is_end = end;
    };

protected:
    char val;
    vector<TrieNode*> children;
    bool is_end = false;
};

class Trie
{
public:
    Trie()
    {
        root = new TrieNode('0');
    }

    void insert(string word)
    {
        TrieNode* cur = root;
        for (int i = 0; i < word.length(); i++)
        {
            int flag = 0;
            for (TrieNode* node : cur->children)
            {
                if (word[i] == node->val)
                {
                    cur = node;
                    if (i == word.length() - 1)
                    {
                        cur->is_end = true;
                    }
                    flag = 1;
                    break;
                }
            }
            if (flag == 1)
            {
                continue;
            }
            const char ch = word[i];
            if (i == word.length() - 1)
            {
                cur->children.push_back(new TrieNode(ch, true));
            }
            else
            {
                cur->children.push_back(new TrieNode(ch));
            }
            cur = cur->children.back();
        }
    }

    bool search(string word)
    {
        TrieNode* cur = root;
        for (int i = 0; i < word.length(); i++)
        {
            int flag = 0;
            for (const auto node : cur->children)
            {
                if (word[i] == node->val)
                {
                    cur = node;
                    flag = 1;
                    break;
                }
            }
            if (flag == 0)
            {
                return false;
            }
        }
        return cur->is_end;
    }

    bool startsWith(string prefix)
    {
        TrieNode* cur = root;
        for (int i = 0; i < prefix.length(); i++)
        {
            int flag = 0;
            for (const auto node : cur->children)
            {
                if (prefix[i] == node->val)
                {
                    cur = node;
                    flag = 1;
                    break;
                }
            }
            if (flag == 0)
            {
                return false;
            }
        }
        return true;
    }

private:
    TrieNode* root;
};


int main()
{
    Trie* trie = new Trie();
    trie->insert("apple");
    trie->insert("app");
    cout << trie->search("app");

    return 0;
}
