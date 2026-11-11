#include "core/buffer.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>

namespace sd {

std::shared_ptr<Buffer> Buffer::allocate(std::int64_t nbytes, bool zeroed) {
    auto buf = std::make_shared<Buffer>(Key{});  // the Buffer and its reference count, together
    // Over-allocate by one alignment unit and round the start up, so the
    // calloc fast path still applies (aligned_alloc has no calloc twin).
    // A zero-byte Buffer still gets a real, aligned pointer.
    const std::size_t request = static_cast<std::size_t>(nbytes) + kAlignment;
    void* raw = zeroed ? std::calloc(1, request) : std::malloc(request);
    if (raw == nullptr) throw std::bad_alloc();
    auto addr = reinterpret_cast<std::uintptr_t>(raw);
    auto aligned = (addr + kAlignment - 1) & ~(std::uintptr_t{kAlignment} - 1);
    buf->raw_ = raw;
    buf->data_ = reinterpret_cast<std::byte*>(aligned);
    buf->size_ = nbytes;
    return buf;
}

Buffer::~Buffer() { std::free(raw_); }

}  // namespace sd
