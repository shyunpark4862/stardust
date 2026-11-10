// A reference-counted block of memory that column data lives in
// (ARCHITECTURE.md §2.1). Columns share Buffers: a slice, a Series taken out
// of a DataFrame and the DataFrame itself all point at the same Buffer, and
// Copy-on-Write (M4) asks the reference count whether a write must copy.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace sd {

class Buffer {
public:
    // One cache line on Apple M4 (ARCHITECTURE.md §0). Aligning the start of
    // every allocation to a cache line means a SIMD loop never straddles two
    // lines at its first load.
    static constexpr std::size_t kAlignment = 128;

    // Fresh memory owned by this Buffer. `zeroed` uses calloc: for large
    // sizes the OS hands out pages that are zero-filled lazily on first touch.
    static std::shared_ptr<Buffer> allocate(std::int64_t nbytes, bool zeroed);

    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    std::byte* data() const { return data_; }
    std::int64_t size() const { return size_; }

    struct Key {  // lets make_shared call the constructor; nobody else can
        explicit Key() = default;
    };
    explicit Buffer(Key) {}

private:
    void* raw_ = nullptr;  // pointer returned by malloc/calloc
    std::byte* data_ = nullptr;
    std::int64_t size_ = 0;
};

}  // namespace sd
