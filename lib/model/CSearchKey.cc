/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CSearchKey.h>

#include <core/CContainerPrinter.h>
#include <core/CFunctional.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/CChecksum.h>

#include <model/CStringStore.h>

#include <boost/bind.hpp>

#include <algorithm>
#include <functional>
#include <ostream>
#include <sstream>

namespace ml {
namespace model {

namespace {

// CSearchKey
const std::string FUNCTION_NAME_TAG("a");
const std::string USE_NULL_TAG("b");
const std::string FIELD_NAME_TAG("c");
const std::string BY_FIELD_NAME_TAG("d");
const std::string OVER_FIELD_NAME_TAG("e");
const std::string PARTITION_FIELD_NAME_TAG("f");
const std::string EXCLUDE_FREQUENT_TAG("g");
const std::string INFLUENCE_FIELD_NAME_TAG("h");
const std::string IDENTIFIER_TAG("i");

// AggregateSearchKey
const std::string KEY_TAG("a");

const std::string EMPTY_STRING;
}

// Initialise statics
const std::string CSearchKey::COUNT_NAME("count");
const char CSearchKey::CUE_DELIMITER('/');
const std::string CSearchKey::EMPTY_STRING;

CSearchKey::CSearchKey(int identifier,
                       function_t::EFunction function,
                       bool useNull,
                       model_t::EExcludeFrequent excludeFrequent,
                       std::string fieldName,
                       std::string byFieldName,
                       std::string overFieldName,
                       std::string partitionFieldName,
                       const TStrVec& influenceFieldNames)
    : m_Identifier(identifier), m_Function(function), m_UseNull(useNull),
      m_ExcludeFrequent(excludeFrequent), m_Hash(0) {
    m_FieldName = CStringStore::names().get(fieldName);
    m_ByFieldName = CStringStore::names().get(byFieldName);
    m_OverFieldName = CStringStore::names().get(overFieldName);
    m_PartitionFieldName = CStringStore::names().get(partitionFieldName);
    for (TStrVec::const_iterator i = influenceFieldNames.begin();
         i != influenceFieldNames.end(); ++i) {
        m_InfluenceFieldNames.push_back(CStringStore::influencers().get(*i));
    }
}

CSearchKey::CSearchKey(core::CStateRestoreTraverser& traverser, bool& successful)
    : m_Identifier(0), m_Function(function_t::E_IndividualCount),
      m_UseNull(false), m_ExcludeFrequent(model_t::E_XF_None), m_Hash(0) {
    successful = traverser.traverseSubLevel(
        std::bind(&CSearchKey::acceptRestoreTraverser, this, std::placeholders::_1));
}

bool CSearchKey::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == IDENTIFIER_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_Identifier) == false) {
                LOG_ERROR(<< "Invalid identifier in " << traverser.value());
                return false;
            }
        } else if (name == FUNCTION_NAME_TAG) {
            int function(-1);
            if (core::CStringUtils::stringToType(traverser.value(), function) == false ||
                function < 0) {
                LOG_ERROR(<< "Invalid function in " << traverser.value());
                return false;
            }
            m_Function = static_cast<function_t::EFunction>(function);
        } else if (name == USE_NULL_TAG) {
            int useNull(-1);
            if (core::CStringUtils::stringToType(traverser.value(), useNull) == false) {
                LOG_ERROR(<< "Invalid use null flag in " << traverser.value());
                return false;
            }
            m_UseNull = (useNull != 0);
        } else if (name == EXCLUDE_FREQUENT_TAG) {
            int excludeFrequent(-1);
            if ((core::CStringUtils::stringToType(traverser.value(), excludeFrequent) == false) ||
                (excludeFrequent < 0)) {
                LOG_ERROR(<< "Invalid excludeFrequent flag in " << traverser.value());
                return false;
            }
            m_ExcludeFrequent = static_cast<model_t::EExcludeFrequent>(excludeFrequent);
        } else if (name == FIELD_NAME_TAG) {
            m_FieldName = CStringStore::names().get(traverser.value());
        } else if (name == BY_FIELD_NAME_TAG) {
            m_ByFieldName = CStringStore::names().get(traverser.value());
        } else if (name == OVER_FIELD_NAME_TAG) {
            m_OverFieldName = CStringStore::names().get(traverser.value());
        } else if (name == PARTITION_FIELD_NAME_TAG) {
            m_PartitionFieldName = CStringStore::names().get(traverser.value());
        } else if (name == INFLUENCE_FIELD_NAME_TAG) {
            m_InfluenceFieldNames.push_back(
                CStringStore::influencers().get(traverser.value()));
        }
    } while (traverser.next());

    return true;
}

void CSearchKey::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(IDENTIFIER_TAG, m_Identifier);
    inserter.insertValue(FUNCTION_NAME_TAG, static_cast<int>(m_Function));
    inserter.insertValue(USE_NULL_TAG, static_cast<int>(m_UseNull));
    inserter.insertValue(EXCLUDE_FREQUENT_TAG, static_cast<int>(m_ExcludeFrequent));
    inserter.insertValue(FIELD_NAME_TAG, *m_FieldName);
    inserter.insertValue(BY_FIELD_NAME_TAG, *m_ByFieldName);
    inserter.insertValue(OVER_FIELD_NAME_TAG, *m_OverFieldName);
    inserter.insertValue(PARTITION_FIELD_NAME_TAG, *m_PartitionFieldName);
    for (std::size_t i = 0u; i < m_InfluenceFieldNames.size(); ++i) {
        inserter.insertValue(INFLUENCE_FIELD_NAME_TAG, *m_InfluenceFieldNames[i]);
    }
}

void CSearchKey::swap(CSearchKey& other) {
    std::swap(m_Identifier, other.m_Identifier);
    std::swap(m_Function, other.m_Function);
    std::swap(m_UseNull, other.m_UseNull);
    std::swap(m_ExcludeFrequent, other.m_ExcludeFrequent);
    m_FieldName.swap(other.m_FieldName);
    m_ByFieldName.swap(other.m_ByFieldName);
    m_OverFieldName.swap(other.m_OverFieldName);
    m_PartitionFieldName.swap(other.m_PartitionFieldName);
    m_InfluenceFieldNames.swap(other.m_InfluenceFieldNames);
    std::swap(m_Hash, other.m_Hash);
}

bool CSearchKey::operator==(const CSearchKey& rhs) const {
    using TStrEqualTo = std::equal_to<std::string>;

    return this->hash() == rhs.hash() && m_Identifier == rhs.m_Identifier &&
           m_Function == rhs.m_Function && m_UseNull == rhs.m_UseNull &&
           m_ExcludeFrequent == rhs.m_ExcludeFrequent &&
           m_FieldName == rhs.m_FieldName && m_ByFieldName == rhs.m_ByFieldName &&
           m_OverFieldName == rhs.m_OverFieldName &&
           m_PartitionFieldName == rhs.m_PartitionFieldName &&
           m_InfluenceFieldNames.size() == rhs.m_InfluenceFieldNames.size()
           // Compare dereferenced strings rather than pointers as there's a
           // (small) possibility that the string store will not always return
           // the same pointer for the same string
           && std::equal(m_InfluenceFieldNames.begin(), m_InfluenceFieldNames.end(),
                         rhs.m_InfluenceFieldNames.begin(),
                         core::CFunctional::SDereference<TStrEqualTo>());
}

bool CSearchKey::operator<(const CSearchKey& rhs) const {
    // We rely on simple count to come before other detectors when we sort
    if (this->isSimpleCount() != rhs.isSimpleCount()) {
        return this->isSimpleCount() ? true : false;
    }

    if (this->hash() == rhs.hash()) {
        if (m_Identifier == rhs.m_Identifier) {
            if (m_Function == rhs.m_Function) {
                if (m_UseNull == rhs.m_UseNull) {
                    if (m_ExcludeFrequent == rhs.m_ExcludeFrequent) {
                        // Use compare() to calculate equality and less than in one call
                        int comp(m_FieldName->compare(*rhs.m_FieldName));
                        if (comp != 0) {
                            return comp < 0;
                        }

                        comp = m_ByFieldName->compare(*rhs.m_ByFieldName);
                        if (comp != 0) {
                            return comp < 0;
                        }

                        comp = m_OverFieldName->compare(*rhs.m_OverFieldName);
                        if (comp != 0) {
                            return comp < 0;
                        }

                        if (m_InfluenceFieldNames.size() <
                            rhs.m_InfluenceFieldNames.size()) {
                            return true;
                        }
                        if (m_InfluenceFieldNames.size() >
                            rhs.m_InfluenceFieldNames.size()) {
                            return false;
                        }
                        for (std::size_t i = 0u; i < m_InfluenceFieldNames.size(); ++i) {
                            comp = m_InfluenceFieldNames[i]->compare(
                                *rhs.m_InfluenceFieldNames[i]);
                            if (comp != 0) {
                                return comp < 0;
                            }
                        }

                        return m_PartitionFieldName < rhs.m_PartitionFieldName;
                    }

                    return m_ExcludeFrequent < rhs.m_ExcludeFrequent;
                }

                return m_UseNull < rhs.m_UseNull;
            }

            return m_Function < rhs.m_Function;
        }

        return m_Identifier < rhs.m_Identifier;
    }

    return this->hash() < rhs.hash();
}

namespace {

// This is keyed on a 'by' field name of 'count', which isn't allowed
// in a real field config, as it doesn't make sense.
const CSearchKey SIMPLE_COUNT_KEY(0, // identifier
                                  function_t::E_IndividualCount,
                                  true,
                                  model_t::E_XF_None,
                                  EMPTY_STRING,
                                  CSearchKey::COUNT_NAME);
}

const CSearchKey& CSearchKey::simpleCountKey() {
    return SIMPLE_COUNT_KEY;
}

bool CSearchKey::isSimpleCount() const {
    return isSimpleCount(m_Function, *m_ByFieldName);
}

bool CSearchKey::isSimpleCount(function_t::EFunction function, const std::string& byFieldName) {
    return function == function_t::E_IndividualCount && byFieldName == COUNT_NAME;
}

bool CSearchKey::isMetric() const {
    return function_t::isMetric(m_Function);
}

bool CSearchKey::isPopulation() const {
    return function_t::isPopulation(m_Function);
}

std::string CSearchKey::toCue() const {
    std::string cue;
    cue.reserve(64 + // hopefully covers function description and slashes
                m_FieldName->length() + m_ByFieldName->length() +
                m_OverFieldName->length() + m_PartitionFieldName->length());
    cue += function_t::print(m_Function);
    cue += CUE_DELIMITER;
    cue += m_UseNull ? '1' : '0';
    cue += CUE_DELIMITER;
    cue += core::CStringUtils::typeToString(static_cast<int>(m_ExcludeFrequent));
    cue += CUE_DELIMITER;
    cue += *m_FieldName;
    cue += CUE_DELIMITER;
    cue += *m_ByFieldName;
    if (!m_OverFieldName->empty() || !m_PartitionFieldName->empty()) {
        cue += CUE_DELIMITER;
        cue += *m_OverFieldName;
        if (!m_PartitionFieldName->empty()) {
            cue += CUE_DELIMITER;
            cue += *m_PartitionFieldName;
        }
    }
    return cue;
}

std::string CSearchKey::debug() const {
    std::ostringstream strm;
    strm << *this;
    return strm.str();
}

int CSearchKey::identifier() const {
    return m_Identifier;
}

function_t::EFunction CSearchKey::function() const {
    return m_Function;
}

bool CSearchKey::useNull() const {
    return m_UseNull;
}

model_t::EExcludeFrequent CSearchKey::excludeFrequent() const {
    return m_ExcludeFrequent;
}

bool CSearchKey::hasField(const std::string& name) const {
    return *m_PartitionFieldName == name || *m_OverFieldName == name ||
           *m_ByFieldName == name || *m_FieldName == name;
}

const std::string& CSearchKey::fieldName() const {
    return *m_FieldName;
}

const std::string& CSearchKey::byFieldName() const {
    return *m_ByFieldName;
}

const std::string& CSearchKey::overFieldName() const {
    return *m_OverFieldName;
}

const std::string& CSearchKey::partitionFieldName() const {
    return *m_PartitionFieldName;
}

const CSearchKey::TStoredStringPtrVec& CSearchKey::influenceFieldNames() const {
    return m_InfluenceFieldNames;
}

uint64_t CSearchKey::hash() const {
    if (m_Hash != 0) {
        return m_Hash;
    }
    m_Hash = m_UseNull ? 1 : 0;
    m_Hash = 4 * m_Hash + static_cast<uint64_t>(m_ExcludeFrequent);
    m_Hash = core::CHashing::hashCombine(m_Hash, static_cast<uint64_t>(m_Identifier));
    m_Hash = core::CHashing::hashCombine(m_Hash, static_cast<uint64_t>(m_Function));
    m_Hash = maths::CChecksum::calculate(m_Hash, *m_FieldName);
    m_Hash = maths::CChecksum::calculate(m_Hash, *m_ByFieldName);
    m_Hash = maths::CChecksum::calculate(m_Hash, *m_OverFieldName);
    m_Hash = maths::CChecksum::calculate(m_Hash, *m_PartitionFieldName);
    m_Hash = maths::CChecksum::calculate(m_Hash, m_InfluenceFieldNames);
    m_Hash = std::max(m_Hash, uint64_t(1));
    return m_Hash;
}

std::ostream& operator<<(std::ostream& strm, const CSearchKey& key) {
    // The format for this is very similar to the format used by toCue() at the
    // time of writing.  However, do NOT combine the code because the intention
    // is to simplify toCue() in the future.
    strm << key.m_Identifier << "==" << function_t::print(key.m_Function) << '/'
         << (key.m_UseNull ? '1' : '0') << '/' << static_cast<int>(key.m_ExcludeFrequent)
         << '/' << *key.m_FieldName << '/' << *key.m_ByFieldName << '/'
         << *key.m_OverFieldName << '/' << *key.m_PartitionFieldName << '/';

    for (size_t i = 0; i < key.m_InfluenceFieldNames.size(); ++i) {
        if (i > 0) {
            strm << ',';
        }
        strm << *key.m_InfluenceFieldNames[i];
    }

    return strm;
}
}
}
