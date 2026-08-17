#include <iostream>
#include <list>
#include <unordered_map>

using namespace std;

class LRUCache
{
private:
    int capacity;
    list<pair<int, int>> cache_list;
    unordered_map<int, list<pair<int, int>>::iterator> key_to_iter;

public:
    LRUCache(int capacity) : capacity(capacity)
    {
    }

    int get(int key)
    {
        auto hash_it = key_to_iter.find(key);
        if (hash_it == key_to_iter.end())
        {
            return -1;
        }
        auto list_it = hash_it->second;
        cache_list.splice(cache_list.begin(), cache_list, list_it);
        return list_it->second;
    }

    void put(int key, int value)
    {
        auto hash_it = key_to_iter.find(key);
        if (hash_it != key_to_iter.end())
        {
            auto list_it = hash_it->second;
            list_it->second = value;
            cache_list.splice(cache_list.begin(), cache_list, list_it);
            return;
        }
        cache_list.emplace_front(key, value);
        key_to_iter[key] = cache_list.begin();
        if (cache_list.size() > capacity)
        {
            key_to_iter.erase(cache_list.back().first);
            cache_list.pop_back();
        }
    }
};

int main()
{
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
