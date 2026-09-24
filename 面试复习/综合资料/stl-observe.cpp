#include <cassert>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

// Define TRACE_THROWING_MOVE to compare relocation with a throwing move signature.
#ifdef TRACE_THROWING_MOVE
constexpr bool kNothrowMove = false;
#else
constexpr bool kNothrowMove = true;
#endif

struct Trace {
    static int next_id;
    int id;
    int value;

    explicit Trace(int v) : id(++next_id), value(v) {
        log("construct");
    }

    Trace(const Trace& other) : id(++next_id), value(other.value) {
        log("copy-construct", &other);
    }

    Trace(Trace&& other) noexcept(kNothrowMove)
        : id(++next_id), value(other.value) {
        other.value = -1;
        log("move-construct", &other);
    }

    Trace& operator=(const Trace& other) {
        if (this != &other) {
            value = other.value;
        }
        log("copy-assign", &other);
        return *this;
    }

    Trace& operator=(Trace&& other) noexcept(kNothrowMove) {
        if (this != &other) {
            value = other.value;
            other.value = -1;
        }
        log("move-assign", &other);
        return *this;
    }

    ~Trace() {
        log("destroy");
    }

    void log(const char* operation, const Trace* source = nullptr) const {
        std::cout << operation << " id=" << id << " value=" << value
                  << " this=" << static_cast<const void*>(this);
        if (source) {
            std::cout << " source-id=" << source->id
                      << " source=" << static_cast<const void*>(source);
        }
        std::cout << '\n';
    }
};

int Trace::next_id = 0;

void checkpoint(const char* phase, const std::vector<Trace>& v) {
    // BREAKPOINT: every vector operation has completed when this line is reached.
    std::cout << "\nCHECKPOINT " << phase << '\n';
    std::cout << "object=" << static_cast<const void*>(&v)
              << " data=" << static_cast<const void*>(v.data())
              << " size=" << v.size() << " capacity=" << v.capacity()
              << " sizeof(vector)=" << sizeof(v)
              << " sizeof(Trace)=" << sizeof(Trace) << '\n';
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << "  [" << i << "] id=" << v[i].id
                  << " value=" << v[i].value
                  << " address=" << static_cast<const void*>(&v[i]) << '\n';
    }
}

int main() {
    std::cout << "move noexcept=" << kNothrowMove << '\n';
    {
        std::vector<Trace> v;
        checkpoint("0: empty", v);

        v.reserve(2);
        assert(v.empty() && v.capacity() >= 2);
        checkpoint("1: reserve(2), no elements constructed", v);

        v.emplace_back(10);
        checkpoint("2: emplace_back(10), direct construction", v);

        v.push_back(Trace(20));
        checkpoint("3: push_back(temporary), then temporary destroyed", v);

        // Force reallocation without relying on an implementation's growth factor.
        v.reserve(v.capacity() + 1);
        assert(v.size() == 2 && v[0].value == 10 && v[1].value == 20);
        checkpoint("4: forced reallocation", v);

        // Separate capacity growth from resize's element construction.
        v.reserve(8);
        checkpoint("5: reserve(8), still two elements", v);

        {
            Trace fill(7);
            v.resize(8, fill);
            assert(v.size() == 8 && fill.value == 7);
            for (std::size_t i = 2; i < v.size(); ++i) {
                assert(v[i].value == 7);
            }
            checkpoint("6: resize(8, fill), six copies", v);
        }

        v.erase(v.begin() + 1);
        assert(v.size() == 7 && v[0].value == 10 && v[1].value == 7);
        checkpoint("7: erase second element, shift by assignment", v);

        const auto saved_capacity = v.capacity();
        v.clear();
        assert(v.empty() && v.capacity() == saved_capacity);
        checkpoint("8: clear, elements destroyed but capacity retained", v);

        v.emplace_back(99);
        assert(v.size() == 1 && v.capacity() == saved_capacity);
        checkpoint("9: reuse retained storage", v);
    }
    std::cout << "\nDONE: vector and remaining element destroyed\n";
}
