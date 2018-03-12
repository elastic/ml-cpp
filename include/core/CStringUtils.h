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
#ifndef INCLUDED_ml_core_CStringUtils_h
#define INCLUDED_ml_core_CStringUtils_h

#include <core/CIEEE754.h>
#include <core/CNonInstantiatable.h>
#include <core/ImportExport.h>

#include <locale>
#include <string>
#include <vector>


namespace ml {
namespace core {


//! \brief
//! A holder of string utility methods.
//!
//! DESCRIPTION:\n
//! A holder of string utility methods.
//!
//! IMPLEMENTATION DECISIONS:\n
//!
class CORE_EXPORT CStringUtils : private CNonInstantiatable {
    public:
        //! We should only have one definition of whitespace across the whole
        //! product - this definition matches what ::isspace() considers as
        //! whitespace in the "C" locale
        static const std::string WHITESPACE_CHARS;

    public:
        typedef std::vector<std::string>    TStrVec;
        typedef TStrVec::iterator           TStrVecItr;
        typedef TStrVec::const_iterator     TStrVecCItr;

    public:
        //! If \p c is the start of a UTF-8 character, return the number of
        //! bytes in the whole character.  Otherwise (i.e. it's a continuation
        //! character) return -1.
        static int utf8ByteType(char c);

        //! Convert a type to a string
        template<typename T>
        static std::string typeToString(const T &type) {
            return CStringUtils::_typeToString(type);
        }

        //! Convert a double to a pretty string (single precision using %g formatting).
        static std::string typeToStringPretty(double d);

        //! For types other than double, use the default conversions
        template<typename T>
        static std::string typeToStringPretty(const T &type) {
            return CStringUtils::_typeToString(type);
        }

        //! Convert a double to a string with the specified precision
        static std::string typeToStringPrecise(double d,
                                               CIEEE754::EPrecision precision);

        //! For types other than double, default conversions are precise
        template<typename T>
        static std::string typeToStringPrecise(const T &type,
                                               CIEEE754::EPrecision /*precision*/) {
            return CStringUtils::_typeToString(type);
        }

        //! Convert a string to a type
        template<typename T>
        static bool stringToType(const std::string &str, T &ret) {
            return CStringUtils::_stringToType(false, str, ret);
        }

        //! Convert a string to a type, and don't print an
        //! error message if the conversion fails
        template<typename T>
        static bool stringToTypeSilent(const std::string &str, T &ret) {
            return CStringUtils::_stringToType(true, str, ret);
        }

        //! Joins the strings in the container with the \p delimiter.
        //! CONTAINER must be a container of std::string.
        template<typename CONTAINER>
        static std::string join(const CONTAINER &strings, const std::string &delimiter) {
            if (strings.empty()) {
                return std::string();
            }
            std::size_t requiredSpace = computeStringLength(strings.begin(), strings.end());
            requiredSpace += (strings.size() - 1) * delimiter.length();
            if (requiredSpace == 0) {
                return std::string();
            }
            std::string output;
            output.reserve(requiredSpace);
            CStringUtils::join(strings.begin(), strings.end(), delimiter, output);
            return output;
        }

        //! Joins the strings in the range with the \p delimiter.
        //! ITR must be a forward iterator that dereferences to std::string.
        template<typename ITR>
        static void join(ITR begin,
                         ITR end,
                         const std::string &delimiter,
                         std::string &output) {
            if (begin == end) {
                return;
            }
            for (;;) {
                output += *begin;
                if (++begin == end) {
                    break;
                }
                output += delimiter;
            }
        }

        //! Convert a string to lower case
        static std::string toLower(std::string str);

        //! Convert a string to upper case
        static std::string toUpper(std::string str);

        //! How many times does word occur in str?
        static size_t numMatches(const std::string &str,
                                 const std::string &word);

        //! Trim whitespace characters from the beginning and end of a string
        static void trimWhitespace(std::string &str);

        //! Trim certain characters from the beginning and end of a string
        static void trim(const std::string &toTrim,
                         std::string &str);

        //! Replace adjacent whitespace characters with single spaces
        static std::string normaliseWhitespace(const std::string &str);

        //! Find and replace a string within another string
        static size_t replace(const std::string &from,
                              const std::string &to,
                              std::string &str);

        //! Find and replace the first occurrence (only) of a string within
        //! another string
        static size_t replaceFirst(const std::string &from,
                                   const std::string &to,
                                   std::string &str);

        //! Escape a specified set of characters in a string
        static void escape(char escape,
                           const std::string &toEscape,
                           std::string &str);

        //! Remove a given escape character from a string
        static void unEscape(char escape, std::string &str);

        //! Tokenise a std::string based on a delimiter.
        //! This does NOT behave like strtok - it matches
        //! the entire delimiter not just characters in it
        static void tokenise(const std::string &delim,
                             const std::string &str,
                             TStrVec &tokens,
                             std::string &remainder);

        //! Find the longest common substring of two strings
        static std::string longestCommonSubstr(const std::string &str1,
                                               const std::string &str2);

        //! Find the longest common subsequence of two strings
        static std::string longestCommonSubsequence(const std::string &str1,
                                                    const std::string &str2);

        //! Convert between wide and narrow strings.
        //! There's currently no clever processing here for character set
        //! conversion, so for non-ASCII characters the results won't be great.
        //! TODO - Use a string library (e.g. ICU) to add support for sensible
        //! conversion between different character sets.
        static std::string wideToNarrow(const std::wstring &wideStr);
        static std::wstring narrowToWide(const std::string &narrowStr);

        //! Get a locale object for character transformations
        //! TODO - remove when we switch to a character conversion library
        //! (e.g. ICU)
        static const std::locale &locale(void);

    private:
        //! Internal calls for public templated methods
        //! Important: These are implemented in terms of the built-in
        //! types.  The public templated methods will call the correct
        //! one based on the actual underlying type for a given typedef.
        //! For example, suppose time_t is a long on a particular
        //! platform.  The user calls typeToString passing a time_t
        //! without caring what the underlying type is.  Then the
        //! compiler calls _typeToString(long) having translated the
        //! typedef to its actual underlying type.  But at no point
        //! did the user have to know the underlying type.
        //! In almost every other part of the code base, the built-in
        //! types should not be used, as they restrict the ease with
        //! which we could switch between 32 bit and 64 bit compilation.
        //! Instead typedefs like uint##_t, int##_t, size_t, etc. should
        //! be used.  But because these methods are called from a
        //! templated wrapper and selected by the compiler they're a
        //! special case.

        static std::string _typeToString(const unsigned long long &);
        static std::string _typeToString(const unsigned long &);
        static std::string _typeToString(const unsigned int &);
        static std::string _typeToString(const unsigned short &);
        static std::string _typeToString(const long long &);
        static std::string _typeToString(const long &);
        static std::string _typeToString(const int &);
        static std::string _typeToString(const short &);
        static std::string _typeToString(const bool &);

        //! There's a function for double, but not float as we want to
        //! discourage the use of float.
        static std::string _typeToString(const double &);

        static std::string _typeToString(const char *);
        static std::string _typeToString(const char &);

        //! This one seems silly, but it allows generic methods to be written
        //! more easily
        static const std::string &_typeToString(const std::string &str);

        static bool _stringToType(bool silent, const std::string &, unsigned long long &);
        static bool _stringToType(bool silent, const std::string &, unsigned long &);
        static bool _stringToType(bool silent, const std::string &, unsigned int &);
        static bool _stringToType(bool silent, const std::string &, unsigned short &);
        static bool _stringToType(bool silent, const std::string &, long long &);
        static bool _stringToType(bool silent, const std::string &, long &);
        static bool _stringToType(bool silent, const std::string &, int &);
        static bool _stringToType(bool silent, const std::string &, short &);

        //! This bool converter accepts true/false and yes/no as well as
        //! numeric values
        static bool _stringToType(bool silent, const std::string &, bool &);

        //! There's a function for double, but not float as we want to
        //! discourage the use of float.
        static bool _stringToType(bool silent, const std::string &, double &);

        static bool _stringToType(bool silent, const std::string &, char &);

        //! This one seems silly, but it allows generic methods to be written
        //! more easily
        static bool _stringToType(bool, const std::string &, std::string &);

        template<typename ITR>
        static std::size_t computeStringLength(ITR begin, ITR end) {
            std::size_t length(0);
            while (begin != end) {
                length += begin->length();
                ++begin;
            }
            return length;
        }
};

//! Macro to convert a pre-processor symbol to a string constant - has to be
//! done in a macro unfortunately as the # operator is only recognised by the
//! pre-processor.
#define STRINGIFY_MACRO(str) (#str)


}
}

#endif // INCLUDED_ml_core_CStringUtils_h

