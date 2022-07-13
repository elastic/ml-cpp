/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_core_CContainerPrinter_h
#define INCLUDED_ml_core_CContainerPrinter_h

#include <core/CNonInstantiatable.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <boost/log/sources/record_ostream.hpp>
#include <boost/numeric/conversion/bounds.hpp>

#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>

namespace ml {
namespace core {
class CStoredStringPtr;

namespace printer_detail {

using true_ = std::true_type;
using false_ = std::false_type;

//! \name Check for nested typedef "const_iterator".
//@{
template<typename, typename = void>
struct has_const_iterator : false_ {};
template<typename T>
struct has_const_iterator<T, std::void_t<typename T::const_iterator>> : true_ {};
//@}

//! \name Check for member const function print.
//@{
template<typename, typename = void>
struct has_member_print_function : false_ {};
// clang-format off
template<typename T>
struct has_member_print_function<
        T, std::enable_if_t<std::is_same_v<decltype(&T::print), std::string (T::*)() const>>
    > : true_ {};
// clang-format on
//@}

//! \name Check if a type is printable to an ostream.
//@{
template<typename, typename = void>
struct is_printable : false_ {};
// clang-format off
template<typename T>
struct is_printable<
        T, std::enable_if_t<std::is_same_v<decltype(std::cout << std::declval<T>()), std::ostream&>>
    > : true_ {};
// clang-format on
//@}

//! \name Container Element Printer
//!
//! Template and partial specializations which selects how to print an
//! object which isn't a collection. This can use one of CStringUtils,
//! a std::ostringstream object or a nested print function.
//@{
template<bool>
class CElementPrinter {};
template<>
class CElementPrinter<false> {
public:
    template<typename T>
    static std::string print(const T& value) {
        // Use CStringUtils if possible: it's much faster but only supports
        // fundamental types.
        return print_(value, std::is_arithmetic<T>{});
    }

private:
    //! CStringUtil implementation with bounds checking.
    template<typename T>
    inline static std::string print_(T value, true_ /*is arithmetic*/) {
        // For signed types only.
        if (value != T(0) && value == boost::numeric::bounds<T>::lowest()) {
            return "\"min\"";
        }
        if (value == boost::numeric::bounds<T>::highest()) {
            return "\"max\"";
        }
        return CStringUtils::typeToStringPretty(value);
    }

    //! CStringUtil implementation.
    inline static std::string print_(bool value, true_ /*is arithmetic*/) {
        return CStringUtils::typeToStringPretty(value);
    }

    //! std::ostringstream implementation.
    template<typename T>
    inline static std::string print_(const T& value, false_ /*is arithmetic*/) {
        std::ostringstream result;
        result << value;
        return result.str();
    }
};
template<>
class CElementPrinter<true> {
public:
    template<typename T>
    static std::string print(const T& value) {
        return value.print();
    }
};
//@}

//! \name Print Forwarder
//!
//! Template and partial specializations to forward print requests to either
//! CElementPrinter or a specified printer.
//@{
template<typename PRINTER, bool USE_PRINTER>
class CPrinterForwarder {};
template<typename PRINTER>
class CPrinterForwarder<PRINTER, false> {
public:
    template<typename T>
    static std::string print(const T& value) {
        return CElementPrinter<has_member_print_function<T>::value>::print(value);
    }
};
template<typename PRINTER>
class CPrinterForwarder<PRINTER, true> {
public:
    template<typename T>
    static std::string print(const T& value) {
        return PRINTER::print(value.begin(), value.end());
    }
};
//@}

//! Extracts a std::ostream from \p stream.
template<typename STREAM>
STREAM& stream(STREAM& stream) {
    return stream;
}
//! Extracts a std::ostream from \p stream.
template<typename CHAR_T>
std::ostream& stream(boost::log::basic_record_ostream<CHAR_T>& stream) {
    return stream.stream();
}
}

//! \brief Prints STL compliant container objects and iterator ranges.
//!
//! DESCRIPTION:\n
//! This will print most sorts of containers in a human readable format.
//! For example, the following code\n
//! \code{.cpp}
//!   std::map<int, int> mymap;
//!   std::cout << "mymap = " << CContainerPrinter::print(mymap) << std::endl;
//!   mymap.insert(std::make_pair(1,2));
//!   mymap.insert(std::make_pair(2,5));
//!   std::cout << "mymap = " << CContainerPrinter::print(mymap) << std::endl;
//!
//!   std::vector<std::shared_ptr<int> > myvec(3, std::shared_ptr<int>(new int(1));
//!   std::cout << "myvec = " << CContainerPrinter::print(myvec) << std::endl;
//!
//!   std::list<std::pair<double, double>* > mylist;
//!   std::pair<double, double> p(1.1, 3.2);
//!   mylist.push_back(&p);
//!   mylist.push_back(nullptr);
//!   std::cout << "mylist = " << CContainerPrinter::print(mylist) << std::endl;
//! \endcode
//!
//! produces the following output\n
//! \code
//!   mymap = []
//!   mymap = [(1, 2), (2, 5)]
//!   myvec = [1, 1, 1]
//!   mylist = [(1.1, 3.2), "null"]
//! \endcode
//!
//! It works with most types of containers and also C-style arrays,
//! iterator ranges and pairs. It will generally dereference pointer
//! types. It also handles containers of containers (checks if they
//! have a nested const_iterator typedef).
//!
//! IMPLEMENTATION:\n
//! This is implemented using CStringUtils if possible and otherwise
//! std::ostringstream. It doesn't attempt to be high performance and
//! is primarily intended for testing and debugging.
class CORE_EXPORT CContainerPrinter : private CNonInstantiatable {
private:
    static const std::string NULL_STR;
    static constexpr std::size_t SIZE_OF_ELEMENT{20};

    //! Print a container element for debug.
    template<typename T>
    static std::string printElement(const T& value) {
        using namespace printer_detail;
        using Forwarder = CPrinterForwarder<CContainerPrinter, has_const_iterator<T>::value>;
        return Forwarder::print(value);
    }

    //! Unwrap any wrapped references.
    template<typename T>
    static std::string printElement(std::reference_wrapper<T> ref) {
        return printElement(ref.get());
    }

    //! Print const raw pointer.
    template<typename T>
    static std::string printElement(const T* value) {
        return value == nullptr ? NULL_STR : printElement(*value);
    }

    //! Print a raw pointer.
    template<typename T>
    static std::string printElement(T* value) {
        return value == nullptr ? NULL_STR : printElement(*value);
    }

    //! Print a std::unique_ptr.
    template<typename T>
    static std::string printElement(const std::unique_ptr<T>& value) {
        return value == nullptr ? NULL_STR : printElement(*value);
    }

    //! Print a std::shared_pointer.
    template<typename T>
    static std::string printElement(const std::shared_ptr<T>& value) {
        return value == nullptr ? NULL_STR : printElement(*value);
    }

    // If you find yourself using some different smart pointer and
    // it isn't printing please feel free to add an overload here.

    //! Print an optional.
    template<typename T>
    static std::string printElement(const std::optional<T>& value) {
        return value == std::nullopt ? NULL_STR : printElement(*value);
    }

    //! Print a std::pair.
    template<typename U, typename V>
    static std::string printElement(const std::pair<U, V>& value) {
        std::string result;
        result.reserve(2 * (SIZE_OF_ELEMENT + 1) + 2);
        result += "(";
        result += printElement(value.first);
        result += ", ";
        result += printElement(value.second);
        result += ")";
        return result;
    }

    //! Print a std::tuple.
    template<typename... ARGS>
    static std::string printElement(const std::tuple<ARGS...>& value) {
        std::string result;
        result.reserve((SIZE_OF_ELEMENT + 1) * sizeof...(ARGS) + 2);
        std::apply(
            [&result](ARGS const&... args) {
                result += '(';
                std::size_t n{0};
                ((result += printElement(args) + (++n != sizeof...(ARGS) ? ", " : "")), ...);
                result += ')';
            },
            value);
        return result;
    }

    //! Print a string.
    static const std::string& printElement(const std::string& value) {
        return value;
    }

    //! Print a CStoredStringPtr.
    static const std::string& printElement(const CStoredStringPtr& value);

public:
    //! Fallback print.
    template<typename T>
    static auto print(const T& t) -> decltype(printElement(t)) {
        return printElement(t);
    }

    //! Print a range of values as defined by a start and end iterator. This
    //! assumes that ITR is a forward iterator, i.e. it must implement prefix
    //! ++ and * operators.
    template<typename ITR>
    static std::string print(ITR begin, ITR end) {
        std::string result;
        result.reserve((SIZE_OF_ELEMENT + 1) * std::distance(begin, end) + 2);
        result += "[";
        if (begin != end) {
            for (;;) {
                result += printElement(*begin);
                if (++begin == end) {
                    break;
                }
                result += ", ";
            }
        }
        result += "]";
        return result;
    }

    //! Specialization for C-style arrays.
    template<typename T, std::size_t SIZE>
    static std::string print(const T (&array)[SIZE]) {
        return print(std::begin(array), std::end(array));
    }
};

//! \brief A stream manipulator which enables printing of STL compliant containers.
//!
//! DESCRIPTION:\n
//! Example usage\n
//! \code{.cpp}
//!   std::vector<double> myvec{1.0, 2.0, 3.0};
//!   std::cout << CPrintContainers{} << "myvec = " << myvec << std::endl;
//! \endcode
//!
//! IMPLEMENTATION:\n
//! Libraries have a tendency to overload operator<< for std::ostream (for
//! example PyTorch does this) for STL types. In such cases, we get a violation
//! of the ODR if we do the same. However, we can simply wrap the std::ostream
//! reference and overload operator<< for CPrintContainers to achieve the same
//! effect. This avoids any chance of a collision definitions.
class CPrintContainers {
public:
    void attach(std::ostream& stream) const { m_Stream = &stream; }
    template<typename T>
    void print(T&& t, printer_detail::true_) {
        (*m_Stream) << t;
    }
    template<typename T>
    void print(T&& t, printer_detail::false_) {
        (*m_Stream) << CContainerPrinter::print(t);
    }

private:
    mutable std::ostream* m_Stream{nullptr};
};

//! Convert a STREAM object \p s into a wrapped stream which prints containers.
template<typename STREAM>
CPrintContainers operator<<(STREAM& s, CPrintContainers printer) {
    printer.attach(printer_detail::stream(s));
    return printer;
}
//! Convert a temporary STREAM object \p s into a wrapped stream which prints containers.
template<typename STREAM>
CPrintContainers operator<<(STREAM&& s, CPrintContainers printer) {
    printer.attach(printer_detail::stream(s));
    return printer;
}
//! Print \p t with \p printer.
template<typename T>
CPrintContainers operator<<(CPrintContainers printer, T&& t) {
    printer.print(t, printer_detail::is_printable<T>{});
    return printer;
}
//! Print an array \p t with \p printer.
template<typename T, std::size_t SIZE>
CPrintContainers operator<<(CPrintContainers printer, const T (&t)[SIZE]) {
    printer.print(CContainerPrinter::print(t), printer_detail::true_{});
    return printer;
}
//! Print an array \p t with \p printer.
template<typename T, std::size_t SIZE>
CPrintContainers operator<<(CPrintContainers printer, T (&t)[SIZE]) {
    printer.print(CContainerPrinter::print(t), printer_detail::true_{});
    return printer;
}
//! Print a character array \p t with \p printer.
template<std::size_t SIZE>
CPrintContainers operator<<(CPrintContainers printer, const char (&t)[SIZE]) {
    printer.print(t, printer_detail::true_{});
    return printer;
}
//! Print a character array \p t with \p printer.
template<std::size_t SIZE>
CPrintContainers operator<<(CPrintContainers printer, char (&t)[SIZE]) {
    printer.print(t, printer_detail::true_{});
    return printer;
}
}
}

#endif // INCLUDED_ml_core_CContainerPrinter_h
