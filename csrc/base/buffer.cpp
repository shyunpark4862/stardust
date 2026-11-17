#include "base/buffer.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include <sys/sysctl.h>

namespace sd {

namespace {

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

// std::bad_alloc (MemoryError in Python) that says why.
class TooLarge : public std::bad_alloc {
public:
    explicit TooLarge(std::int64_t nbytes)
        : msg_("cannot allocate " + std::to_string(nbytes) + " bytes: more than this machine's physical memory") {}
    const char* what() const noexcept override { return msg_.c_str(); }

private:
    std::string msg_;
};

}  // namespace

std::shared_ptr<Buffer> Buffer::allocate(std::int64_t nbytes, bool zeroed) {
    auto buf = std::make_shared<Buffer>(Key{});  // the Buffer and its reference count, together
    buf->size_ = nbytes;
    if (static_cast<std::size_t>(nbytes) <= kInline) {
        buf->data_ = buf->inline_;
        if (zeroed) std::memset(buf->inline_, 0, static_cast<std::size_t>(nbytes));
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
    auto buf = std::make_shared<Buffer>(Key{});
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
