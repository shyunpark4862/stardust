#include "bind/errors.hpp"

#include "linalg/errors.hpp"

#include <cstring>
#include <stdexcept>

namespace sd {

std::string with_name(const char* func, const char* msg) {
    if (func == nullptr || *func == '\0') return msg;
    const std::string prefix = std::string(func) + (func[0] == '.' ? ": " : "(): ");
    if (std::strncmp(msg, prefix.c_str(), prefix.size()) == 0) return msg;
    return prefix + msg;
}

void rethrow_named(const char* func) {
    try {
        throw;
    } catch (const nb::builtin_exception& e) {
        switch (e.type()) {
            case nb::exception_type::type_error:
            case nb::exception_type::value_error:
            case nb::exception_type::index_error:
            case nb::exception_type::key_error:
                throw nb::builtin_exception(e.type(), with_name(func, e.what()).c_str());
            default:
                throw;  // stop_iteration, next_overload, ...: control flow, not messages
        }
    } catch (const linalg::LinAlgError& e) {
        throw linalg::LinAlgError(with_name(func, e.what()));
    } catch (const std::overflow_error& e) {
        throw std::overflow_error(with_name(func, e.what()));
    }
}

}  // namespace sd
