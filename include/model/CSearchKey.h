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

#ifndef INCLUDED_ml_model_CSearchKey_h
#define INCLUDED_ml_model_CSearchKey_h

#include <core/CHashing.h>
#include <core/CStoredStringPtr.h>

#include <maths/COrderings.h>

#include <model/FunctionTypes.h>
#include <model/ImportExport.h>

#include <boost/ref.hpp>

#include <iosfwd>
#include <string>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {

//! \brief
//! Associative store key for simple searches.
//!
//! DESCRIPTION:\n
//! The syntax for specifying a simple search is along the lines
//! of:
//!
//! count
//! count by status
//! freq_rare(uri_path) by clientip
//!
//! More generically:
//!
//! fieldname
//! fieldname by byfieldname
//! function
//! function by byfieldname
//! function(fieldname) by byfieldname
//!
//! The syntax that doesn't explicitly specify a function implies a
//! function, so there is always a function. However, the fieldname
//! and "by" fieldname are not always required.
//!
//! In addition, its possible to have a partitioning field, such that
//! a completely different analysis is done for different subsets of
//! the input data.
//!
//! This class simply groups the 5 pieces of information together,
//! reducing the risk of missing places where changes are required
//! if the syntax for specifying an anomaly detector is expanded in
//! the future.
//!
//! IMPLEMENTATION DECISIONS:\n
//! It is assumed that validation of combinations of function,
//! fieldname and "by" fieldname has been done by other code.  This
//! class is intended purely to store the information and be used as
//! a key in associative containers.
//!
class MODEL_EXPORT CSearchKey {
public:
    typedef std::vector<std::string> TStrVec;
    typedef std::vector<core::CStoredStringPtr> TStoredStringPtrVec;

    //! The type of a search key which mixes in the partition field
    //! value.
    typedef std::pair<std::string, CSearchKey> TStrKeyPr;

    //! The type of a constant reference string search key pair.
    //!
    //! \note This is intended for map lookups when one doesn't want
    //! to copy the strings.
    typedef std::pair<boost::reference_wrapper<const std::string>,
                      boost::reference_wrapper<const CSearchKey>>
        TStrCRefKeyCRefPr;

public:
    //! If the "by" field name is "count" then the key represents
    //! a simple count detector
    static const std::string COUNT_NAME;

    //! Character used to delimit the "cue" representation of the key
    static const char CUE_DELIMITER;

    //! An empty string.
    static const std::string EMPTY_STRING;

public:
    //! Construct with an over field and a partitioning field
    //!
    //! \note Use the pass-by-value-and-swap trick to improve performance
    //! when the arguments are temporaries.
    explicit CSearchKey(int identifier = 0,
                        function_t::EFunction function = function_t::E_IndividualCount,
                        bool useNull = false,
                        model_t::EExcludeFrequent excludeFrequent = model_t::E_XF_None,
                        std::string fieldName = EMPTY_STRING,
                        std::string byFieldName = EMPTY_STRING,
                        std::string overFieldName = EMPTY_STRING,
                        std::string partitionFieldName = EMPTY_STRING,
                        const TStrVec& influenceFieldNames = TStrVec());

    //! Create the key from part of an state document.
    //!
    //! \param[in,out] traverser A state document traverser.
    //! \param[out] successful Set to true if the state could be fully
    //! deserialised and false otherwise.
    CSearchKey(core::CStateRestoreTraverser& traverser, bool& successful);

private:
    //! Initialise by traversing a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

public:
    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Efficiently swap the contents of two objects of this class.
    void swap(CSearchKey& other);

    //! Check if this and \p rhs are equal.
    bool operator==(const CSearchKey& rhs) const;

    //! Check if this is less than \p rhs.
    bool operator<(const CSearchKey& rhs) const;

    //! Get an identifier for this search.
    int identifier(void) const;

    //! Get the unique simple counting search key.
    //!
    //! Definition: the function is individual count and the "by"
    //! field name is "count".
    static const CSearchKey& simpleCountKey(void);

    //! Does this key represent a simple counting search?
    bool isSimpleCount(void) const;

    //! Do the function and by field name identify a simple
    //! counting search.
    static bool isSimpleCount(function_t::EFunction function, const std::string& byFieldName);

    //! Is the function type for use with the individual models?
    bool isMetric(void) const;

    //! Is the function type for use with the population models?
    bool isPopulation(void) const;

    //! Create a "cue" suitable to be used in persisted state.
    std::string toCue(void) const;

    //! Debug representation.  Note that operator<<() is more efficient than
    //! generating this debug string and immediately outputting it to a
    //! stream.
    std::string debug(void) const;

    //! Get the function.
    function_t::EFunction function(void) const;

    //! Get whether to use null field values.
    bool useNull(void) const;

    //! Get the ExcludeFrequent setting
    model_t::EExcludeFrequent excludeFrequent(void) const;

    //! Check if there is a field called \p name.
    bool hasField(const std::string& name) const;

    //! Get the value field name.
    const std::string& fieldName(void) const;

    //! Get the by field name.
    const std::string& byFieldName(void) const;

    //! Get the over field name.
    const std::string& overFieldName(void) const;

    //! Get the partition field name.
    const std::string& partitionFieldName(void) const;

    //! Get the influence field names.
    const TStoredStringPtrVec& influenceFieldNames(void) const;

    //! Get a hash of the contents of this key.
    uint64_t hash(void) const;

private:
    int m_Identifier;
    function_t::EFunction m_Function;
    bool m_UseNull;
    model_t::EExcludeFrequent m_ExcludeFrequent;
    core::CStoredStringPtr m_FieldName;
    core::CStoredStringPtr m_ByFieldName;
    core::CStoredStringPtr m_OverFieldName;
    core::CStoredStringPtr m_PartitionFieldName;
    TStoredStringPtrVec m_InfluenceFieldNames;

    //! Used for efficient comparison.
    mutable uint64_t m_Hash;

    // For debug output
    friend MODEL_EXPORT std::ostream& operator<<(std::ostream&, const CSearchKey&);
};

MODEL_EXPORT
std::ostream& operator<<(std::ostream& strm, const CSearchKey& key);

//! Hashes a (string, search key) pair.
class CStrKeyPrHash {
public:
    std::size_t operator()(const CSearchKey::TStrKeyPr& key) const { return this->hash(key); }
    std::size_t operator()(const CSearchKey::TStrCRefKeyCRefPr& key) const {
        return this->hash(key);
    }

private:
    template<typename T>
    std::size_t hash(const T& key) const {
        core::CHashing::CSafeMurmurHash2String64 stringHasher;
        uint64_t result = stringHasher(boost::unwrap_ref(key.first));
        core::CHashing::hashCombine(boost::unwrap_ref(key.second).hash(), result);
        return static_cast<std::size_t>(result);
    }
};

//! Checks if two (string, search key) pairs are equal.
class CStrKeyPrEqual {
public:
    bool operator()(const CSearchKey::TStrKeyPr& lhs, const CSearchKey::TStrKeyPr& rhs) const {
        return this->equal(lhs, rhs);
    }
    bool operator()(const CSearchKey::TStrCRefKeyCRefPr& lhs,
                    const CSearchKey::TStrKeyPr& rhs) const {
        return this->equal(lhs, rhs);
    }
    bool operator()(const CSearchKey::TStrKeyPr& lhs,
                    const CSearchKey::TStrCRefKeyCRefPr& rhs) const {
        return this->equal(lhs, rhs);
    }
    bool operator()(const CSearchKey::TStrCRefKeyCRefPr& lhs,
                    const CSearchKey::TStrCRefKeyCRefPr& rhs) const {
        return this->equal(lhs, rhs);
    }

private:
    template<typename U, typename V>
    bool equal(const U& lhs, const V& rhs) const {
        return boost::unwrap_ref(lhs.second) == boost::unwrap_ref(rhs.second) &&
               boost::unwrap_ref(lhs.first) == boost::unwrap_ref(rhs.first);
    }
};
}
}

#endif // INCLUDED_ml_model_CSearchKey_h
