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

#ifndef INCLUDED_ml_core_CPersistUtils_h
#define INCLUDED_ml_core_CPersistUtils_h

#include <core/CFloatStorage.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <boost/array.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/unordered/unordered_map_fwd.hpp>
#include <boost/unordered/unordered_set_fwd.hpp>
#include <boost/bind.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace ml {
namespace core {

namespace persist_utils_detail {

const std::string FIRST_TAG("a");
const std::string SECOND_TAG("b");
const std::string MAP_TAG("c");
const std::string SIZE_TAG("d");

template<typename T>
struct remove_const {
    typedef typename boost::remove_const<T>::type type;
};

template<typename U, typename V>
struct remove_const<std::pair<U, V> > {
    typedef std::pair<typename remove_const<U>::type,
                      typename remove_const<V>::type> type;
};

//! Template specialisation utility classes for selecting various
//! approaches for persisting and restoring objects.
class BasicPersist {
};
class ContainerPersist {
};
class MemberPersist {
};
class MemberToDelimited {
};
class BasicRestore {
};
class ContainerRestore {
};
class MemberRestore {
};
class MemberFromDelimited {
};

//! Auxiliary type used by has_const_iterator to test for a nested
//! typedef.
template<typename T, typename R = void>
struct enable_if {
    typedef R type;
};

//! Auxiliary type used by has_persist_function to test for a nested
//! member function.
template<typename T, T, typename R = void>
struct enable_if_is {
    typedef R type;
};


//! \name Class used to select appropriate persist implementation
//! for containers.
//@{
template<typename T, typename ITR = void>
struct persist_container_selector {
    typedef BasicPersist value;
};
template<typename T>
struct persist_container_selector<T, typename enable_if<typename T::const_iterator>::type> {
    typedef ContainerPersist value;
};
//@}

//! \name Class used to select appropriate persist implementation.
//@{
template<typename T, typename ENABLE = void>
struct persist_selector {
    typedef typename persist_container_selector<T>::value value;
};
template<typename T>
struct persist_selector<T, typename enable_if_is<void (T::*)(CStatePersistInserter &) const, &T::acceptPersistInserter>::type> {
    typedef MemberPersist value;
};
template<typename T>
struct persist_selector<T, typename enable_if_is<std::string (T::*)(void) const, &T::toDelimited>::type> {
    typedef MemberToDelimited value;
};
//@}

//! Detail of the persist class selected by the object features
template<typename SELECTOR> class CPersisterImpl {
};

//! Convenience function to select implementation.
template<typename T>
bool persist(const std::string &tag, const T &target, CStatePersistInserter &inserter) {
    CPersisterImpl<typename persist_selector<T>::value>::dispatch(tag, target, inserter);
    return true;
}


//! \name Class used to select appropriate restore implementation
//! for containers.
//@{
template<typename T, typename ITR = void>
struct restore_container_selector {
    typedef BasicRestore value;
};
template<typename T>
struct restore_container_selector<T, typename enable_if<typename T::const_iterator>::type> {
    typedef ContainerRestore value;
};
//@}

//! \name Class used to select appropriate restore implementation.
//@{
template<typename T, typename ENABLE = void>
struct restore_selector {
    typedef typename restore_container_selector<T>::value value;
};
template<typename T>
struct restore_selector<T, typename enable_if_is<bool (T::*)(CStateRestoreTraverser &), &T::acceptRestoreTraverser>::type> {
    typedef MemberRestore value;
};
template<typename T>
struct restore_selector<T, typename enable_if_is<bool (T::*)(const std::string &), &T::fromDelimited>::type> {
    typedef MemberFromDelimited value;
};
//@}

//! Detail of the restorer implementation based on object features
template<typename SELECTOR> class CRestorerImpl {
};

//! Convenience function to select implementation.
template<typename T>
bool restore(const std::string &tag, T &target, CStateRestoreTraverser &traverser) {
    return CRestorerImpl<typename restore_selector<T>::value>::dispatch(tag, target, traverser);
}

//! Template specialisation utility classes for determining whether
//! or not a collection has a void rehash(size_t) method and failing
//! that a void reserve(size_t) method (boost unordered_* have both
//! and we prefer rehashing in this case).
class CanReserve {
};

//! \name Class used to select appropriate reserve implementation.
//!
//! Template and partial specialization which chooses between our
//! various reserve implementations.
//@{
template<typename T, typename ENABLE = void>
struct reserve_selector {
    typedef ENABLE value;
};
template<typename T>
struct reserve_selector<T, typename enable_if_is<void (T::*)(std::size_t), &T::reserve>::type> {
    typedef CanReserve value;
};
//@}

//! \brief Implementation of the pre-allocation class for objects
//! which don't support pre-allocation - i.e. do nothing.
template<typename SELECTOR> class CReserveImpl {
    public:
        template<typename T>
        static void dispatch(const T &, std::size_t) {
        }
};

//! \brief Implementation of the pre-allocation class for objects
//! which have a void reserve(size_t) method.
template<>
class CReserveImpl<CanReserve> {
    public:
        template<typename T>
        static void dispatch(T &t, std::size_t amount) {
            t.reserve(amount);
        }
};

//! Helper function to select the correct reserver class based
//! on template specialisation selection
template<typename T>
void reserve(T &t, std::size_t amount) {
    CReserveImpl<typename reserve_selector<T>::value>::dispatch(t, amount);
}

} // persist_utils_detail::


//! \brief A class of persistence patterns.
//!
//! DESCRIPTION:\n
//! Various patterns come up repeatedly in persistence. For example,
//! it is much more efficient to persist containers of basic types
//! to a single string rather than as separate elements in the state
//! document. This also means that we can reserve vectors on restore.
//!
//! Any new patterns should be added here.
class CORE_EXPORT CPersistUtils {
    public:
        static const char DELIMITER;
        static const char PAIR_DELIMITER;

    public:
        //! \brief Utility to convert a built in type to a string
        //! using CStringUtils functions.
        class CORE_EXPORT CBuiltinToString {
            public:
                CBuiltinToString(const char pairDelimiter) :
                    m_PairDelimiter(pairDelimiter) {
                }

                std::string operator()(double value) const {
                    return CStringUtils::typeToStringPrecise(value, CIEEE754::E_SinglePrecision);
                }

                template<typename T>
                std::string operator()(T value) const {
                    return CStringUtils::typeToString(value);
                }

                std::string operator()(int8_t value) const {
                    return CStringUtils::typeToString(static_cast<int>(value));
                }

                std::string operator()(uint8_t value) const {
                    return CStringUtils::typeToString(static_cast<unsigned int>(value));
                }

                std::string operator()(int16_t value) const {
                    return CStringUtils::typeToString(static_cast<int>(value));
                }

                std::string operator()(uint16_t value) const {
                    return CStringUtils::typeToString(static_cast<unsigned int>(value));
                }

                std::string operator()(CFloatStorage value) const {
                    return value.toString();
                }

                template<typename U, typename V>
                std::string operator()(const std::pair<U, V> &value) const {
                    return this->operator()(value.first)
                           + m_PairDelimiter
                           + this->operator()(value.second);
                }

            private:
                char m_PairDelimiter;
        };

        //! \brief Utility to convert a string to a built in type
        //! using CStringUtils functions.
        class CORE_EXPORT CBuiltinFromString {
            public:
                CBuiltinFromString(const char pairDelimiter) :
                    m_PairDelimiter(pairDelimiter) {
                    m_Token.reserve(15);
                }

                template<typename T>
                bool operator()(const std::string &token, T &value) const {
                    return CStringUtils::stringToType(token, value);
                }

                bool operator()(const std::string &token, int8_t &value) const {
                    int value_;
                    if (CStringUtils::stringToType(token, value_)) {
                        value = static_cast<int8_t>(value_);
                        return true;
                    }
                    return false;
                }

                bool operator()(const std::string &token, uint8_t &value) const {
                    unsigned int value_;
                    if (CStringUtils::stringToType(token, value_)) {
                        value = static_cast<uint8_t>(value_);
                        return true;
                    }
                    return false;
                }

                bool operator()(const std::string &token, int16_t &value) const {
                    int value_;
                    if (CStringUtils::stringToType(token, value_)) {
                        value = static_cast<int16_t>(value_);
                        return true;
                    }
                    return false;
                }

                bool operator()(const std::string &token, uint16_t &value) const {
                    unsigned int value_;
                    if (CStringUtils::stringToType(token, value_)) {
                        value = static_cast<uint16_t>(value_);
                        return true;
                    }
                    return false;
                }

                bool operator()(const std::string &token, CFloatStorage &value) const {
                    return value.fromString(token);
                }

                template<typename U, typename V>
                bool operator()(const std::string &token, std::pair<U, V> &value) const {
                    std::size_t delimPos(token.find(m_PairDelimiter));
                    if (delimPos == std::string::npos) {
                        return false;
                    }
                    m_Token.assign(token, 0, delimPos);
                    if (!this->operator()(m_Token, value.first)) {
                        return false;
                    }
                    m_Token.assign(token, delimPos + 1, token.length() - delimPos);
                    return this->operator()(m_Token, value.second);
                }

            private:
                char                m_PairDelimiter;
                mutable std::string m_Token;
        };

        //! Entry method for objects being restored
        template<typename T>
        static bool restore(const std::string &tag,
                            T &collection,
                            CStateRestoreTraverser &traverser) {
            return persist_utils_detail::restore(tag, collection, traverser);
        }

        //! Entry method for objects being persisted
        template<typename T>
        static bool persist(const std::string &tag,
                            const T &collection,
                            CStatePersistInserter &inserter) {
            return persist_utils_detail::persist(tag, collection, inserter);
        }

        //! Wrapper for containers of built in types.
        template<typename CONTAINER>
        static std::string toString(const CONTAINER &collection,
                                    const char delimiter = DELIMITER,
                                    const char pairDelimiter = PAIR_DELIMITER) {
            CBuiltinToString f(pairDelimiter);
            return toString(collection, f, delimiter);
        }

        //! Convert a collection to a string.
        //!
        //! \param[in] collection The collection to persist.
        //! \param[in] stringFunc The function used to persist
        //! elements of the collection.
        //! \param[in] delimiter The delimiter used to separate
        //! elements.
        //! \note This should use RVO so just return the string.
        template<typename CONTAINER, typename F>
        static std::string toString(const CONTAINER &collection,
                                    const F &stringFunc,
                                    const char delimiter = DELIMITER) {
            if (collection.empty()) {
                return std::string();
            }
            auto begin = collection.begin();
            auto end = collection.end();
            return toString(begin, end, stringFunc, delimiter);
        }

        //! Wrapper for containers of built in types.
        template<typename ITR>
        static std::string toString(ITR &begin, ITR &end,
                                    const char delimiter = DELIMITER,
                                    const char pairDelimiter = PAIR_DELIMITER) {
            CBuiltinToString f(pairDelimiter);
            return toString(begin, end, f, delimiter);
        }

        //! Convert the range between 2 iterators to a string.
        //!
        //! \param[in,out] begin The iterator at the start of the range.
        //! This will be equal to end when the function returns
        //! \param[in] end The iterator at the end of the range
        //! \param[in] stringFunc The function used to persist
        //! elements of the collection.
        //! \param[in] delimiter The delimiter used to separate
        //! elements.
        //! \note This should use RVO so just return the string.
        template<typename ITR, typename F>
        static std::string toString(ITR &begin, ITR &end,
                                    const F &stringFunc,
                                    const char delimiter = DELIMITER) {
            std::string result = stringFunc(*begin++);
            for (/**/; begin != end; ++begin) {
                result += delimiter;
                result += stringFunc(*begin);
            }
            return result;
        }

        //! Wrapper for arrays of built in types.
        template<typename T, std::size_t N>
        static bool fromString(const std::string &state,
                               boost::array<T, N> &collection,
                               const char delimiter = DELIMITER,
                               const char pairDelimiter = PAIR_DELIMITER) {
            CBuiltinFromString f(pairDelimiter);
            return fromString(state, f, collection, delimiter);
        }

        //! Wrapper for containers of built in types.
        template<typename CONTAINER>
        static bool fromString(const std::string &state,
                               CONTAINER &collection,
                               const char delimiter = DELIMITER,
                               const char pairDelimiter = PAIR_DELIMITER,
                               bool append = false) {
            CBuiltinFromString f(pairDelimiter);
            return fromString(state, f, collection, delimiter, append);
        }

        //! Wrapper for ranges of built in types.
        template<typename ITR>
        static bool fromString(const std::string &state,
                               ITR begin,
                               ITR end,
                               const char delimiter = DELIMITER,
                               const char pairDelimiter = PAIR_DELIMITER) {
            CBuiltinFromString f(pairDelimiter);
            return fromString(state, f, begin, end, delimiter);
        }

        //! Restore a vector from a string created by toString.
        //!
        //! \param[in] state The string description of the
        //! collection.
        //! \param[in] stringFunc The function used to restore
        //! elements of the collection.
        //! \param[out] collection Filled in with the elements
        //! extracted from \p state.
        //! \param[in] delimiter The delimiter used to separate
        //! elements.
        //! \param[in] append If true append the results to the
        //! collection otherwise it is cleared first
        //! \return True if there was no error parsing \p state
        //! and false otherwise. If the state cannot be parsed
        //! then an empty collection is returned.
        //! \note T must have a default constructor.
        //! \note The delimiter must match the delimiter used
        //! for persistence.
        //! \tparam F Expected to have the signature:
        //! \code
        //! bool (const std::string &, T &)
        //! \endcode
        template<typename T, typename F>
        static bool fromString(const std::string &state,
                               const F &stringFunc,
                               std::vector<T> &collection,
                               const char delimiter = DELIMITER,
                               const bool append = false) {
            if (!append) {
                collection.clear();
            }

            if (state.empty()) {
                return true;
            }

            collection.reserve(std::count(state.begin(), state.end(), delimiter) + 1);

            if (fromString<T>(state, delimiter, stringFunc,
                              std::back_inserter(collection)) == false) {
                collection.clear();
                return false;
            }
            return true;
        }

        //! Restore a boost::array from a string created by toString.
        //!
        //! \param[in] state The string description of the
        //! collection.
        //! \param[in] stringFunc The function used to restore
        //! elements of the collection.
        //! \param[out] collection Filled in with the elements
        //! extracted from \p state.
        //! \param[in] delimiter The delimiter used to separate
        //! elements.
        //! \return True if there was no error parsing \p state
        //! and it contained exactly N elements and false otherwise.
        //! \note The delimiter must match the delimiter used
        //! for persistence.
        //! \tparam F Expected to have the signature:
        //! \code
        //! bool (const std::string &, T &)
        //! \endcode
        template<typename T, std::size_t N, typename F>
        static bool fromString(const std::string &state,
                               const F &stringFunc,
                               boost::array<T, N> &collection,
                               const char delimiter = DELIMITER) {
            if (state.empty()) {
                LOG_ERROR("Unexpected number of elements 0"
                          << ", expected " << N);
                return false;
            }

            std::size_t n = std::count(state.begin(), state.end(), delimiter) + 1;
            if (n != N) {
                LOG_ERROR("Unexpected number of elements " << n
                                                           << ", expected " << N);
                return false;
            }

            return fromString<T>(state, delimiter, stringFunc, collection.begin());
        }

        //! Restore a container from a string created by toString.
        //!
        //! \param[in] state The string description of the
        //! collection.
        //! \param[in] stringFunc The function used to restore
        //! elements of the collection.
        //! \param[out] collection Filled in with the elements
        //! extracted from \p state.
        //! \param[in] delimiter The delimiter used to separate
        //! elements.
        //! \param[in] append If true append the results to the
        //! collection otherwise it is cleared first
        //! \return True if there was no error parsing \p state
        //! and false otherwise. If the state cannot be parsed
        //! then an empty collection is returned.
        //! \note The container value type must have a default
        //! constructor.
        //! \note The delimiter must match the delimiter used
        //! for persistence.
        //! \tparam F Expected to have the signature:
        //! \code{.cpp}
        //! bool (const std::string &, CONTAINER::value_type &)
        //! \endcode
        template<typename CONTAINER, typename F>
        static bool fromString(const std::string &state,
                               const F &stringFunc,
                               CONTAINER &collection,
                               const char delimiter = DELIMITER,
                               bool append = false) {
            typedef typename persist_utils_detail::remove_const<typename CONTAINER::value_type>::type T;

            if (!append) {
                collection.clear();
            }

            if (state.empty()) {
                return true;
            }

            if (fromString<T>(state, delimiter, stringFunc,
                              std::inserter(collection, collection.end())) == false) {
                collection.clear();
                return false;
            }
            return true;
        }

        //! Restore a range from a string created by toString.
        //!
        //! \param[in] state The string description of the range.
        //! \param[in] stringFunc The function used to restore
        //! elements of the range.
        //! \param[out] begin Filled in with the elements
        //! extracted from \p state.
        //! \param[in] end The end of the range into which to
        //! restore the elements.
        //! \param[in] delimiter The delimiter used to separate
        //! elements.
        //! \return True if there was no error parsing \p state
        //! and false otherwise.
        //! \note The container value type must have a default
        //! constructor.
        //! \note The delimiter must match the delimiter used
        //! for persistence.
        //! \tparam F Expected to have the signature:
        //! \code{.cpp}
        //! bool (const std::string &, CONTAINER::value_type &)
        //! \endcode
        template<typename ITR, typename F>
        static bool fromString(const std::string &state,
                               const F &stringFunc,
                               ITR begin, ITR end,
                               const char delimiter = DELIMITER) {

            if (state.empty()) {
                return true;
            }

            std::size_t n = std::count(state.begin(), state.end(), delimiter) + 1;
            std::size_t N = std::distance(begin, end);
            if (n != N) {
                LOG_ERROR("Unexpected number of elements " << n
                                                           << ", expected " << N);
                return false;
            }

            return fromString<typename std::iterator_traits<ITR>::value_type>(
                state, delimiter, stringFunc, begin);
        }

    private:
        //! Restores to an insertion iterator.
        template<typename T, typename F, typename ITR>
        static bool fromString(const std::string &state,
                               const char delimiter,
                               const F &stringFunc,
                               ITR inserter) {
            std::size_t delimPos = state.find(delimiter);
            if (delimPos == std::string::npos) {
                T element;
                if (stringFunc(state, element) == false) {
                    LOG_ERROR("Invalid state " << state);
                    return false;
                }
                *inserter = element;
                ++inserter;
                return true;
            }

            // Reuse this same string to avoid as many allocations
            // as possible.
            //
            // The reservation is 15 because for string implementations
            // using the short string optimisation we don't want to
            // cause an unnecessary allocation.
            std::string token;
            token.reserve(15);
            token.assign(state, 0, delimPos);
            {
                T element;
                if (stringFunc(token, element) == false) {
                    LOG_ERROR("Invalid element 0 : element " << token
                                                             << " in " << state);
                    return false;
                }
                *inserter = element;
                ++inserter;
            }

            std::size_t i = 1u;
            std::size_t lastDelimPos(delimPos);
            while (lastDelimPos != std::string::npos) {
                delimPos = state.find(delimiter, lastDelimPos + 1);
                if (delimPos == std::string::npos) {
                    token.assign(state, lastDelimPos + 1, state.length() - lastDelimPos);
                } else {
                    token.assign(state, lastDelimPos + 1, delimPos - lastDelimPos - 1);
                }

                T element;
                if (stringFunc(token, element) == false) {
                    LOG_ERROR("Invalid element " << i
                                                 << " : element " << token
                                                 << " in " << state);
                    return false;
                }
                *inserter = element;

                ++i;
                lastDelimPos = delimPos;
                ++inserter;
            }

            return true;
        }
};


namespace persist_utils_detail {

//! Basic persist functionality implementation, for PODs or pairs
template<>
class CPersisterImpl<BasicPersist> {
    public:
        template<typename T>
        static void dispatch(const std::string &tag,
                             const T &t,
                             CStatePersistInserter &inserter) {
            CPersistUtils::CBuiltinToString toString(CPersistUtils::PAIR_DELIMITER);
            inserter.insertValue(tag, toString(t));
        }

        template<typename A, typename B>
        static void dispatch(const std::string &tag,
                             const std::pair<A, B> &t,
                             CStatePersistInserter &inserter) {
            inserter.insertLevel(tag, boost::bind(&newLevel<A, B>, boost::cref(t), _1));
        }

    private:
        template<typename A, typename B>
        static void newLevel(const std::pair<A, B> &t,
                             CStatePersistInserter &inserter) {
            persist(FIRST_TAG, t.first, inserter);
            persist(SECOND_TAG, t.second, inserter);
        }
};

//! Persister class for containers. If contained types are PODs
//! they are written as a delimited string, or strings written
//! as straight strings, else added as a new level and re-dispatched
template<>
class CPersisterImpl<ContainerPersist> {
    public:
        template<typename T>
        static void dispatch(const std::string &tag,
                             const T &container,
                             CStatePersistInserter &inserter) {
            doInsert(tag,
                     container,
                     inserter,
                     boost::integral_constant<bool, boost::is_arithmetic<typename T::value_type>::value>(),
                     boost::false_type());
        }

        //! Specialisation for boost::unordered_set which orders values.
        template<typename T, typename H, typename P, typename A>
        static void dispatch(const std::string &tag,
                             const boost::unordered_set<T, H, P, A> &container,
                             CStatePersistInserter &inserter) {
            typedef typename std::vector<T>                                   TVec;
            typedef typename boost::unordered_set<T, H, P, A>::const_iterator TCItr;
            typedef typename std::vector<TCItr>                               TCItrVec;

            if (boost::is_arithmetic<T>::value) {
                TVec values(container.begin(), container.end());
                std::sort(values.begin(), values.end());
                doInsert(tag, values, inserter, boost::true_type(), boost::false_type());
            } else {
                TCItrVec iterators;
                iterators.reserve(container.size());
                for (TCItr i = container.begin(); i != container.end(); ++i) {
                    iterators.push_back(i);
                }

                // Sort the values to ensure consistent persist state.
                std::sort(iterators.begin(), iterators.end(),
                          [](TCItr lhs, TCItr rhs) {
                            return *lhs < *rhs;
                        });
                doInsert(tag, iterators, inserter, boost::false_type(), boost::true_type());
            }
        }

        //! Specialisation for boost::unordered_map which orders values.
        template<typename K, typename V, typename H, typename P, typename A>
        static void dispatch(const std::string &tag,
                             const boost::unordered_map<K, V, H, P, A> &container,
                             CStatePersistInserter &inserter) {
            typedef typename boost::unordered_map<K, V, H, P, A>::const_iterator TCItr;
            typedef typename std::vector<TCItr>                                  TCItrVec;

            TCItrVec iterators;
            iterators.reserve(container.size());
            for (TCItr i = container.begin(); i != container.end(); ++i) {
                iterators.push_back(i);
            }

            // Sort the keys to ensure consistent persist state.
            std::sort(iterators.begin(), iterators.end(),
                      [](TCItr lhs, TCItr rhs) {
                        return lhs->first < rhs->first;
                    });
            doInsert(tag, iterators, inserter, boost::false_type(), boost::true_type());
        }

        //! Specialisation for std::string, which has iterators but doesn't need
        //! to be split up into individual characters
        static void dispatch(const std::string &tag,
                             const std::string &str,
                             CStatePersistInserter &inserter) {
            inserter.insertValue(tag, str);
        }

    private:
        //! Handle the case of a built-in type.
        //!
        //! \note Type T is not an iterator
        template<typename T>
        static void doInsert(const std::string &tag,
                             const T &container,
                             CStatePersistInserter &inserter,
                             boost::true_type,
                             boost::false_type) {
            inserter.insertValue(tag, CPersistUtils::toString(container));
        }

        //! Handle the case for a non-built-in type, which will be added
        //! as a new level.
        //!
        //! \note Type T is not an iterator
        template<typename T>
        static void doInsert(const std::string &tag,
                             const T &container,
                             CStatePersistInserter &inserter,
                             boost::false_type,
                             boost::false_type) {
            typedef typename T::const_iterator TCItr;
            inserter.insertLevel(tag, boost::bind(&newLevel<TCItr>,
                                                  container.begin(), container.end(), container.size(), _1));
        }

        //! Handle the case for a non-built-in type, which will be added
        //! as a new level.
        //!
        //! \note Type T is an iterator
        template<typename T>
        static void doInsert(const std::string &tag,
                             const T &t,
                             CStatePersistInserter &inserter,
                             boost::false_type,
                             boost::true_type) {
            typedef boost::indirect_iterator<typename T::const_iterator> TCItr;
            inserter.insertLevel(tag, boost::bind(&newLevel<TCItr>,
                                                  TCItr(t.begin()), TCItr(t.end()), t.size(), _1));
        }

        //! Dispatch a collection of items
        //!
        //! \note The container size is added to allow the restorer to
        //! pre-size the new container if appropriate
        template<typename ITR>
        static void newLevel(ITR begin,
                             ITR end,
                             std::size_t size,
                             CStatePersistInserter &inserter) {
            inserter.insertValue(SIZE_TAG, size);
            for (; begin != end; ++begin) {
                persist(FIRST_TAG, *begin, inserter);
            }
        }
};

//! \brief Persister for objects which have an acceptPersistInserter method.
template<>
class CPersisterImpl<MemberPersist> {
    public:
        template<typename T>
        static void dispatch(const std::string &tag,
                             const T &t,
                             CStatePersistInserter &inserter) {
            inserter.insertLevel(tag, boost::bind(&newLevel<T>, boost::cref(t), _1));
        }

    private:
        template<typename T>
        static void newLevel(const T &t, CStatePersistInserter &inserter) {
            t.acceptPersistInserter(inserter);
        }
};

//! \brief Persister for objects which have a toDelimited method.
template<>
class CPersisterImpl<MemberToDelimited> {
    public:
        template<typename T>
        static void dispatch(const std::string &tag,
                             const T &t,
                             CStatePersistInserter &inserter) {
            inserter.insertValue(tag, t.toDelimited());
        }
};


//! \brief Restorer class for PODs and pairs.
template<>
class CRestorerImpl<BasicRestore> {
    public:
        template<typename T>
        static bool dispatch(const std::string &tag,
                             T &t,
                             CStateRestoreTraverser &traverser) {
            bool ret = true;
            if (traverser.name() == tag) {
                CPersistUtils::CBuiltinFromString stringFunc(CPersistUtils::PAIR_DELIMITER);
                ret = stringFunc(traverser.value(), t);
            }
            return ret;
        }

        template<typename A, typename B>
        static bool dispatch(const std::string &tag,
                             std::pair<A, B> &t,
                             CStateRestoreTraverser &traverser) {
            bool ret = true;
            if (traverser.name() == tag) {
                if (!traverser.hasSubLevel()) {
                    LOG_ERROR("SubLevel mismatch in restore, at " << traverser.name());
                    return false;
                }
                ret = traverser.traverseSubLevel(boost::bind(&newLevel<A, B>, boost::ref(t), _1));
            }
            return ret;
        }

    private:
        template<typename A, typename B>
        static bool newLevel(std::pair<A, B> &t,
                             CStateRestoreTraverser &traverser) {
            if (traverser.name() != FIRST_TAG) {
                LOG_ERROR("Tag mismatch at " << traverser.name() << ", expected " << FIRST_TAG);
                return false;
            }
            if (!restore(FIRST_TAG, t.first, traverser)) {
                LOG_ERROR("Restore error at " << traverser.name() << ": " << traverser.value());
                return false;
            }
            if (!traverser.next()) {
                LOG_ERROR("Restore error at " << traverser.name() << ": " << traverser.value());
                return false;
            }
            if (traverser.name() != SECOND_TAG) {
                LOG_ERROR("Tag mismatch at " << traverser.name() << ", expected " << SECOND_TAG);
                return false;
            }
            if (!restore(SECOND_TAG, t.second, traverser)) {
                LOG_ERROR("Restore error at " << traverser.name() << ": " << traverser.value());
                return false;
            }
            return true;
        }
};

//! \brief Restorer class for collections of items.
template<>
class CRestorerImpl<ContainerRestore> {
    public:
        template<typename T>
        static bool dispatch(const std::string &tag,
                             T &container,
                             CStateRestoreTraverser &traverser) {
            return doTraverse(tag,
                              container,
                              traverser,
                              boost::integral_constant<bool, boost::is_arithmetic<typename T::value_type>::value>());
        }

        //! Specialisation for std::string, which has iterators but doesn't
        //! need to be split up into individual characters
        static bool dispatch(const std::string &tag,
                             std::string &str,
                             CStateRestoreTraverser &traverser) {
            if (traverser.name() == tag) {
                str = traverser.value();
            }
            return true;
        }

    private:
        struct SSubLevel {
            template<typename T>
            bool operator()(T &container, CStateRestoreTraverser &traverser) {
                typedef typename remove_const<typename T::value_type>::type TValueType;
                do {
                    if (traverser.name() == SIZE_TAG) {
                        std::size_t size = 0;
                        if (!core::CStringUtils::stringToType(traverser.value(), size)) {
                            LOG_WARN("Failed to determine size: " << traverser.value());
                        } else {
                            reserve(container, size);
                        }
                    } else {
                        TValueType value;
                        if (!restore(FIRST_TAG, value, traverser)) {
                            LOG_ERROR("Restoration error at " << traverser.name());
                            return false;
                        }
                        container.insert(container.end(), value);
                    }
                } while (traverser.next());
                return true;
            }

            template<typename T, std::size_t N>
            bool operator()(boost::array<T, N> &container, CStateRestoreTraverser &traverser) {
                typedef typename remove_const<T>::type TValueType;
                typename boost::array<T, N>::iterator i = container.begin();
                do {
                    TValueType value;
                    if (traverser.name() == FIRST_TAG) {
                        if (!restore(FIRST_TAG, value, traverser)) {
                            LOG_ERROR("Restoration error at " << traverser.name());
                            return false;
                        }
                        *(i++) = value;
                    }
                } while (traverser.next());
                return true;
            }
        };

    private:
        template<typename T>
        static bool doTraverse(const std::string &tag,
                               T &container,
                               CStateRestoreTraverser &traverser,
                               boost::true_type) {
            bool ret = true;
            if (traverser.name() == tag) {
                ret = CPersistUtils::fromString(traverser.value(), container);
            }
            return ret;
        }

        template<typename T>
        static bool doTraverse(const std::string &tag,
                               T &container,
                               CStateRestoreTraverser &traverser,
                               boost::false_type) {
            bool ret = true;
            if (traverser.name() == tag) {
                if (!traverser.hasSubLevel()) {
                    LOG_ERROR("SubLevel mismatch in restore, at " << traverser.name());
                    return false;
                }
                ret = traverser.traverseSubLevel(boost::bind<bool>(SSubLevel(), boost::ref(container), _1));
            }
            return ret;
        }
};

//! \brief Restorer for objects which has an acceptRestoreTraverser method.
template<>
class CRestorerImpl<MemberRestore> {
    public:
        template<typename T>
        static bool dispatch(const std::string &tag,
                             T &t,
                             CStateRestoreTraverser &traverser) {
            bool ret = true;
            if (traverser.name() == tag) {
                if (!traverser.hasSubLevel()) {
                    LOG_ERROR("SubLevel mismatch in restore, at " << traverser.name());
                    return false;
                }
                ret = traverser.traverseSubLevel(boost::bind(&subLevel<T>, boost::ref(t), _1));
            }
            return ret;
        }

    private:
        template<typename T>
        static bool subLevel(T &t, CStateRestoreTraverser &traverser) {
            return t.acceptRestoreTraverser(traverser);
        }
};


//! \brief Restorer for objects which have a fromDelimited method.
template<>
class CRestorerImpl<MemberFromDelimited> {
    public:
        template<typename T>
        static bool dispatch(const std::string & /*tag*/,
                             T &t,
                             CStateRestoreTraverser &traverser) {
            return t.fromDelimited(traverser.value());
        }
};

} // persist_utils_detail::

}
}

#endif // INCLUDED_ml_core_CPersistUtils_h
