/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_core_CContainerPrinter_h
#define INCLUDED_ml_core_CContainerPrinter_h

#include <core/CNonInstantiatable.h>
#include <core/CStoredStringPtr.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/optional/optional_fwd.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

#include <memory>
#include <sstream>
#include <string>

namespace ml {
namespace core {

namespace printer_detail {

typedef boost::true_type true_;
typedef boost::false_type false_;

//! Auxiliary type used by has_const_iterator to test for a nested
//! typedef.
template <typename T, typename R = void> struct enable_if_has { typedef R type; };

//! Auxiliary type used by has_print_function to test for a nested
//! member function.
template <typename T, T, typename R = void> struct enable_if_is { typedef R type; };

//! \name Check For Nested "const_iterator"
//!
//! Template and partial specialization which identify whether a
//! type has a nested constant iterator.
//!
//! \note Partial specializations can't be nested classes.
//! \note Uses SFINAE to check for nested typedef.
//! \note Uses the enable_if trick to get around the restriction that
//! "A partially specialized non-type argument expression shall not
//! involve a template parameter of the partial specialization except
//! when the argument expression is a simple identifier" (see section
//! 14.5.4/9 of the standard).
//@{
template <typename T, typename ENABLE = void> struct has_const_iterator { typedef false_ value; };

template <typename T>
struct has_const_iterator<T, typename enable_if_has<typename T::const_iterator>::type> {
    typedef true_ value;
};
//@}

//! \name Check for Member Function Called Print
//!
//! Template and partial specialization which identify whether a
//! type has a constant member function returning a string called
//! print.
//!
//! \note Partial specializations can't be nested classes.
//! \note Uses SFINAE to check for member function.
//! \note Uses the enable_if trick to get around the restriction that
//! "A partially specialized non-type argument expression shall not
//! involve a template parameter of the partial specialization except
//! when the argument expression is a simple identifier" (see section
//! 14.5.4/9 of the standard).
//@{
template <typename T, typename U = void> struct has_print_function { typedef false_ value; };

template <typename T>
struct has_print_function<T,
                          typename enable_if_is<std::string (T::*)(void) const, &T::print>::type> {
    typedef true_ value;
};
//@}

//! \name Container Leaf Printer
//!
//! Template and partial specializations which select to print
//! functionality for an object which isn't a collection, i.e.
//! a leaf of a composite container. This can either use a nested
//! print function or a std::ostringstream.
//!
//! \note Partial specializations can't be nested classes.
//@{
template <typename SELECTOR> class CLeafPrinter {};
template <> class CLeafPrinter<false_> {
public:
    static std::string print(const std::string &value) { return value; }

    template <typename T> static std::string print(const T &value) {
        // Use CStringUtils if possible: it's much faster but
        // only supports fundamental types.
        return print_(value, typename boost::is_arithmetic<T>::type());
    }

private:
    //! Fast CStringUtil implementation with bounds checking.
    template <typename T> inline static std::string print_(T value, true_ /*is arithmetic*/) {
        // For signed types only.
        if (value != T(0) && value == boost::numeric::bounds<T>::lowest()) {
            return "\"min\"";
        }
        if (value == boost::numeric::bounds<T>::highest()) {
            return "\"max\"";
        }
        return CStringUtils::typeToStringPretty(value);
    }

    //! Fast CStringUtil implementation.
    inline static std::string print_(bool value, true_ /*is arithmetic*/) {
        return CStringUtils::typeToStringPretty(value);
    }

    //! Slow std::ostringstream stream implementation.
    template <typename T>
    inline static std::string print_(const T &value, false_ /*is arithmetic*/) {
        std::ostringstream result;
        result << value;
        return result.str();
    }
};
template <> class CLeafPrinter<true_> {
public:
    template <typename T> static std::string print(const T &value) { return value.print(); }
};
//@}

//! \name Container Node Printer
//!
//! Template and partial specializations which select to forward
//! printing to the specified printer or print the value using
//! either a member print function or with a std::ostringstream.
//!
//! \note Partial specializations can't be nested classes.
//@{
template <typename SELECTOR, typename PRINTER> class CNodePrinter {};
template <typename PRINTER> class CNodePrinter<false_, PRINTER> {
public:
    template <typename T> static std::string print(const T &value) {
        return CLeafPrinter<typename has_print_function<T>::value>::print(value);
    }
};
template <typename PRINTER> class CNodePrinter<true_, PRINTER> {
public:
    template <typename T> static std::string print(const T &value) { return PRINTER::print(value); }
};
//@}
}

//! \brief Prints STL compliant container objects and iterator ranges.
//!
//! DESCRIPTION:\n
//! This will print most sorts of containers in a human readable format.
//! For example, the following code\n
//! \code
//!   std::map<int, int> mymap;
//!   std::cout << "mymap = " << CContainerPrinter::print(mymap) << std::endl;
//!   mymap.insert(std::make_pair(1,2));
//!   mymap.insert(std::make_pair(2,5));
//!   std::cout << "mymap = " << CContainerPrinter::print(mymap) << std::endl;
//!
//!   std::vector<boost::shared_ptr<int> > myvec(3, boost::shared_ptr<int>(new int(1));
//!   std::cout << "myvec = " << CContainerPrinter::print(myvec) << std::endl;
//!
//!   std::list<std::pair<double, double>* > mylist;
//!   std::pair<double, double> p(1.1, 3.2);
//!   mylist.push_back(&p);
//!   mylist.push_back(0);
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
//! std::ostringstream (which is slooow). However, it makes too much
//! use of std::ostringstream and isn't too careful about copying
//! strings to be really high performance and so this functionality
//! is primarily intended for testing and debugging.
class CORE_EXPORT CContainerPrinter : private CNonInstantiatable {
private:
    //! Print a non associative container element for debug.
    template <typename T> static std::string printElement(const T &value) {
        using namespace printer_detail;
        typedef typename boost::unwrap_reference<T>::type U;
        typedef CNodePrinter<typename has_const_iterator<U>::value, CContainerPrinter> Printer;
        return Printer::print(boost::unwrap_ref(value));
    }

    //! Print a non associative element pointer to const for debug.
    template <typename T> static std::string printElement(const T *value) {
        if (value == 0) {
            return "\"null\"";
        }
        std::ostringstream result;
        result << printElement(boost::unwrap_ref(*value));
        return result.str();
    }

    //! Print a non associative element pointer for debug.
    template <typename T> static std::string printElement(T *value) {
        if (value == 0) {
            return "\"null\"";
        }
        std::ostringstream result;
        result << printElement(boost::unwrap_ref(*value));
        return result.str();
    }

    //! Print a std::auto_ptr.
    template <typename T> static std::string printElement(const std::auto_ptr<T> &value) {
        if (value.get() == 0) {
            return "\"null\"";
        }
        std::ostringstream result;
        result << printElement(*value);
        return result.str();
    }

    //! Print a CStoredStringPtr
    static std::string printElement(const CStoredStringPtr &value) {
        if (value == nullptr) {
            return "\"null\"";
        }
        return *value;
    }

    //! Print a boost::shared_pointer.
    template <typename T> static std::string printElement(const boost::shared_ptr<T> &value) {
        if (value == boost::shared_ptr<T>()) {
            return "\"null\"";
        }
        std::ostringstream result;
        result << printElement(*value);
        return result.str();
    }

    // If you find yourself using some different smart pointer and
    // it isn't printing please feel free to add an overload here.

    //! Print a non associative (boost) optional element for debug.
    template <typename T> static std::string printElement(const boost::optional<T> &value) {
        if (!value) {
            return "\"null\"";
        }
        std::ostringstream result;
        result << printElement(boost::unwrap_ref(*value));
        return result.str();
    }

    //! Print an associative container element for debug.
    template <typename U, typename V>
    static std::string printElement(const std::pair<U, V> &value) {
        std::ostringstream result;
        result << "(" << printElement(boost::unwrap_ref(value.first)) << ", "
               << printElement(boost::unwrap_ref(value.second)) << ")";
        return result.str();
    }

    //! Print a string for debug (otherwise we split them into their
    //! component characters since they have iterators).
    static std::string printElement(const std::string &value) { return value; }

public:
    //! Function object wrapper around printElement for use with STL.
    class CElementPrinter {
    public:
        template <typename T> std::string operator()(const T &value) { return printElement(value); }
    };

    //! Print a range of values as defined by a start and end iterator
    //! for debug. This assumes that ITR is a forward iterator, i.e.
    //! it must implement prefix ++ and * operators.
    template <typename ITR> static std::string print(ITR begin, ITR end) {
        std::ostringstream result;

        result << "[";
        if (begin != end) {
            for (;;) {
                result << printElement(*begin);
                if (++begin == end) {
                    break;
                }
                result << ", ";
            }
        }
        result << "]";

        return result.str();
    }

    //! Print a STL compliant container for debug.
    template <typename CONTAINER> static std::string print(const CONTAINER &container) {
        return print(boost::unwrap_ref(container).begin(), boost::unwrap_ref(container).end());
    }

    //! Specialization for arrays.
    template <typename T, std::size_t SIZE> static std::string print(const T (&array)[SIZE]) {
        return print(array, array + SIZE);
    }

    //! Print a pair for debug.
    template <typename U, typename V> static std::string print(const std::pair<U, V> &value) {
        return printElement(value);
    }

    //! Print an optional value for debug.
    template <typename T> static std::string print(const boost::optional<T> &value) {
        return printElement(value);
    }
};
}
}
#endif// INCLUDED_ml_core_CContainerPrinter_h
