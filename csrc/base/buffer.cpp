#include "base/buffer.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

#include <sys/sysctl.h>

namespace sd {

// Bytes of physical memory (hw.memsize), read once; no limit if it cannot be read.
std::uint64_t physical_memory() {
    static const std::uint64_t bytes = [] {
        std::uint64_t v = 0;
        std::size_t len = sizeof v;
        if (sysctlbyname("hw.memsize", &v, &len, nullptr, 0) != 0 || v == 0) return ~std::uint64_t{0};
        return v;
    }();
    return bytes;
}

namespace {

// std::bad_alloc (MemoryError in Python) that says why.
class TooLarge : public std::bad_alloc {
public:
    explicit TooLarge(std::int64_t nbytes)
        : msg_("cannot allocate " + std::to_string(nbytes) + " bytes: more than this machine's physical memory") {}
    const char* what() const noexcept override { return msg_.c_str(); }

private:
    std::string msg_;
};

// The blocks that hold a Buffer and its reference count (one size: the
// shared_ptr control block allocate_shared makes), kept per thread for reuse.
// A tiny `x + y` spent ~15 of its ~80 ns in malloc and free of that block; a
// thread's own list needs no lock, and a block freed on another thread simply
// joins that thread's list (the memory is malloc's either way). At most
// kCached blocks per thread (~35 KB); the rest go back to free().
struct BlockCache {
    static constexpr int kCached = 64;
    void* blocks[kCached];
    int count;  // -1 once the thread's flusher ran: no caching any more
};
thread_local BlockCache t_cache{};  // trivially destructible: usable to the thread's very end

// Frees the cached blocks when the thread ends (pool workers on a resize).
struct CacheFlusher {
    ~CacheFlusher() {
        for (int i = 0; i < t_cache.count; ++i) std::free(t_cache.blocks[i]);
        t_cache.count = -1;
    }
};
thread_local CacheFlusher t_flusher;

template <typename T>
struct CachedAlloc {
    using value_type = T;
    CachedAlloc() = default;
    template <typename U>
    CachedAlloc(const CachedAlloc<U>&) {}
    T* allocate(std::size_t n) {
        static_assert(alignof(T) <= 16);  // what malloc guarantees on macOS
        if (n == 1 && t_cache.count > 0) return static_cast<T*>(t_cache.blocks[--t_cache.count]);
        if (n == 1 && t_cache.count == 0) (void)&t_flusher;  // registers the flusher on first use
        void* p = std::malloc(n * sizeof(T));
        if (p == nullptr) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t n) {
        if (n == 1 && t_cache.count >= 0 && t_cache.count < BlockCache::kCached) {
            if (t_cache.count == 0) (void)&t_flusher;
            t_cache.blocks[t_cache.count++] = p;
            return;
        }
        std::free(p);
    }
    template <typename U>
    bool operator==(const CachedAlloc<U>&) const {
        return true;
    }
};

std::shared_ptr<Buffer> make_buffer() { return std::allocate_shared<Buffer>(CachedAlloc<Buffer>{}, Buffer::Key{}); }

}  // namespace

std::shared_ptr<Buffer> Buffer::allocate(std::int64_t nbytes, bool zeroed) {
    auto buf = make_buffer();  // the Buffer and its reference count, together
    buf->size_ = nbytes;
    if (static_cast<std::size_t>(nbytes) <= kInline) {
        const auto addr = reinterpret_cast<std::uintptr_t>(buf->inline_);
        buf->data_ = buf->inline_ + (((addr + kAlignment - 1) & ~(std::uintptr_t{kAlignment} - 1)) - addr);
        if (zeroed) std::memset(buf->data_, 0, static_cast<std::size_t>(nbytes));
        return buf;
    }
    // A request larger than the machine's physical memory can never be filled: the
    // system would grant it (address space only) and kill the process while it is written
    // (rng.permutation(2**40)). MemoryError instead (user decision 2026-11-18). A request
    // that is smaller but exceeds what is free is left to the system.
    if (static_cast<std::uint64_t>(nbytes) > physical_memory()) throw TooLarge(nbytes);
    // Over-allocate by one alignment unit and round the start up, so the
    // calloc fast path above still applies (aligned_alloc has no calloc twin).
    const std::size_t request = static_cast<std::size_t>(nbytes) + kAlignment;
    void* raw = zeroed ? std::calloc(1, request) : std::malloc(request);
    if (raw == nullptr) throw std::bad_alloc();
    auto addr = reinterpret_cast<std::uintptr_t>(raw);
    auto aligned = (addr + kAlignment - 1) & ~(std::uintptr_t{kAlignment} - 1);
    buf->raw_ = raw;
    buf->data_ = reinterpret_cast<std::byte*>(aligned);
    return buf;
}

std::shared_ptr<Buffer> Buffer::adopt(const Py_buffer& view) {
    auto buf = make_buffer();
    buf->external_ = true;
    buf->view_ = view;
    buf->data_ = static_cast<std::byte*>(view.buf);
    buf->size_ = view.len;
    return buf;
}

Buffer::~Buffer() {
    if (external_) {
        // The last owner may be dropped where the GIL is not held (later,
        // worker threads); releasing a Py_buffer touches Python objects.
        PyGILState_STATE gil = PyGILState_Ensure();
        PyBuffer_Release(&view_);
        PyGILState_Release(gil);
    } else {
        std::free(raw_);  // null for inline data: a no-op
    }
}

}  // namespace sd
