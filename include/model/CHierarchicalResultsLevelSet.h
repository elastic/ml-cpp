/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CHierarchicalResultsLevelSet_h
#define INCLUDED_ml_model_CHierarchicalResultsLevelSet_h

#include <core/CCompressedDictionary.h>

#include <maths/CChecksum.h>
#include <maths/COrderings.h>

#include <model/CHierarchicalResults.h>

#include <stdint.h>

namespace ml {
namespace model {

//! \brief Defines a set of objects with one set for each distinct level
//! in the hierarchical results.
//!
//! DESCRIPTION:\n
//! This holds a templated set type stored for each distinct level in the
//! hierarchical results. These are leaf, person, partition, influencer,
//! influencer bucket and bucket. This is used as the base type for our
//! stateful visitors, specifically, the results aggregator and normalizer.
//!
//! IMPLEMENTATION:\n
//! Common functionality for maintaining and accessing state has been
//! factored into this class. As such the functionality implemented by
//! this class is entirely protected and intended only for use by the
//! internals of the concrete implementations. Creation of set elements
//! is the responsibility of a factory object which is passed in by
//! the calling code to increase generality. The factory must have
//! a make function return T by value and taking the strings identifying
//! the level. T must have a clear function and propagateForwardByTime
//! functions.
template<typename T>
class CHierarchicalResultsLevelSet : public CHierarchicalResultsVisitor {
protected:
    using Type = T;
    using TTypePtrVec = std::vector<Type*>;
    using TDictionary = core::CCompressedDictionary<1>;
    using TWord = TDictionary::CWord;
    using TWordTypePr = std::pair<TWord, T>;
    using TWordTypePrVec = std::vector<TWordTypePr>;
    using TWordTypePrVecItr = typename TWordTypePrVec::iterator;
    using TWordTypePrVecCItr = typename TWordTypePrVec::const_iterator;

protected:
    explicit CHierarchicalResultsLevelSet(const T& bucketElement)
        : m_BucketElement(bucketElement) {}

    //! Get the root unique element.
    const T& bucketElement() const { return m_BucketElement; }
    //! Get a writable root unique element.
    T& bucketElement() { return m_BucketElement; }

    //! Get an influencer bucket element for \p influencerFieldName.
    //!
    //! \note Returns NULL if there isn't a matching one.
    const T* influencerBucketElement(const std::string& influencerFieldName) const {
        return element(m_InfluencerBucketSet, influencerFieldName);
    }

    //! Get an influencer element for \p influencerFieldName.
    //!
    //! \note Returns NULL if there isn't a matching one.
    const T* influencerElement(const std::string& influencerFieldName) const {
        return element(m_InfluencerSet, influencerFieldName);
    }

    //! Get a partition element for \p partitionFieldName.
    //!
    //! \note Returns NULL if there isn't a matching one.
    const T* partitionElement(const std::string& partitionFieldName) const {
        return element(m_PartitionSet, partitionFieldName);
    }

    //! Get a person element.
    //!
    //! \note Returns NULL if there isn't a matching one.
    const T* personElement(const std::string& partitionFieldName,
                           const std::string& personFieldName) const {
        TWord word = ms_Dictionary.word(partitionFieldName, personFieldName);
        TWordTypePrVecCItr i = element(m_PersonSet, word);
        return (i != m_PersonSet.end() && i->first == word) ? &i->second : nullptr;
    }

    //! Get a leaf element.
    //!
    //! \note Returns NULL if there isn't a matching one.
    const T* leafElement(const std::string& partitionFieldName,
                         const std::string& personFieldName,
                         const std::string& functionName,
                         const std::string& valueFieldName) const {
        TWord word = ms_Dictionary.word(partitionFieldName, personFieldName,
                                        functionName, valueFieldName);
        TWordTypePrVecCItr i = element(m_LeafSet, word);
        return (i != m_LeafSet.end() && i->first == word) ? &i->second : nullptr;
    }

    //! Get the influencer bucket set.
    const TWordTypePrVec& influencerBucketSet() const {
        return m_InfluencerBucketSet;
    }
    //! Get a writable influencer bucket set.
    TWordTypePrVec& influencerBucketSet() { return m_InfluencerBucketSet; }

    //! Get the influencer set.
    const TWordTypePrVec& influencerSet() const { return m_InfluencerSet; }
    //! Get a writable influencer set.
    TWordTypePrVec& influencerSet() { return m_InfluencerSet; }

    //! Get the partition set.
    const TWordTypePrVec& partitionSet() const { return m_PartitionSet; }
    //! Get a writable partition set.
    TWordTypePrVec& partitionSet() { return m_PartitionSet; }

    //! Get the person set.
    const TWordTypePrVec& personSet() const { return m_PersonSet; }
    //! Get a writable person set.
    TWordTypePrVec& personSet() { return m_PersonSet; }

    //! Get the leaf set.
    const TWordTypePrVec& leafSet() const { return m_LeafSet; }
    //! Get a writable leaf set.
    TWordTypePrVec& leafSet() { return m_LeafSet; }

    //! Clear all the sets.
    void clear() {
        m_BucketElement.clear();
        m_InfluencerBucketSet.clear();
        m_InfluencerSet.clear();
        m_PartitionSet.clear();
        m_PersonSet.clear();
        m_LeafSet.clear();
    }

    //! Sort all the sets.
    void sort() {
        sort(m_InfluencerBucketSet);
        sort(m_InfluencerSet);
        sort(m_PartitionSet);
        sort(m_PersonSet);
        sort(m_LeafSet);
    }

    //! Age the level set elements.
    template<typename F>
    void age(F doAge) {
        doAge(m_BucketElement);
        age(m_InfluencerBucketSet, doAge);
        age(m_InfluencerSet, doAge);
        age(m_PartitionSet, doAge);
        age(m_PersonSet, doAge);
        age(m_LeafSet, doAge);
    }

    //! Get and possibly add a normalizer for \p node.
    template<typename FACTORY>
    void elements(const TNode& node, bool pivot, const FACTORY& factory, TTypePtrVec& result) {
        result.clear();
        if (this->isSimpleCount(node)) {
            return;
        }

        if (pivot && this->isRoot(node)) {
            TWord word = ms_Dictionary.word(*node.s_Spec.s_PersonFieldName);
            TWordTypePrVecItr i = element(m_InfluencerBucketSet, word);
            if (i == m_InfluencerBucketSet.end() || i->first != word) {
                i = m_InfluencerBucketSet.emplace(i, word, factory.make(node, pivot));
            }
            result.push_back(&i->second);
            return;
        }
        if (pivot && !this->isRoot(node)) {
            TWord word = ms_Dictionary.word(*node.s_Spec.s_PersonFieldName);
            TWordTypePrVecItr i = element(m_InfluencerSet, word);
            if (i == m_InfluencerSet.end() || i->first != word) {
                i = m_InfluencerSet.emplace(i, word, factory.make(node, pivot));
            }
            result.push_back(&i->second);
            return;
        }

        if (this->isLeaf(node)) {
            TWord word = ms_Dictionary.word(
                *node.s_Spec.s_PartitionFieldName, *node.s_Spec.s_PersonFieldName,
                *node.s_Spec.s_FunctionName, *node.s_Spec.s_ValueFieldName);
            TWordTypePrVecItr i = element(m_LeafSet, word);
            if (i == m_LeafSet.end() || i->first != word) {
                i = m_LeafSet.emplace(i, word, factory.make(node, pivot));
            }
            result.push_back(&i->second);
        }
        if (this->isPerson(node)) {
            TWord word = ms_Dictionary.word(*node.s_Spec.s_PartitionFieldName,
                                            *node.s_Spec.s_PersonFieldName);
            TWordTypePrVecItr i = element(m_PersonSet, word);
            if (i == m_PersonSet.end() || i->first != word) {
                i = m_PersonSet.emplace(i, word, factory.make(node, pivot));
            }
            result.push_back(&i->second);
        }
        if (this->isPartition(node)) {
            TWord word = ms_Dictionary.word(*node.s_Spec.s_PartitionFieldName);

            TWordTypePrVecItr i = element(m_PartitionSet, word);
            if (i == m_PartitionSet.end() || i->first != word) {
                i = m_PartitionSet.emplace(i, word, factory.make(node, pivot));
            }
            result.push_back(&i->second);
        }
        if (this->isRoot(node)) {
            result.push_back(&m_BucketElement);
        }
    }

    //! Get a checksum of the set data.
    uint64_t checksum(uint64_t seed) const {
        seed = maths::CChecksum::calculate(seed, m_BucketElement);
        seed = maths::CChecksum::calculate(seed, m_InfluencerBucketSet);
        seed = maths::CChecksum::calculate(seed, m_InfluencerSet);
        seed = maths::CChecksum::calculate(seed, m_PartitionSet);
        seed = maths::CChecksum::calculate(seed, m_PersonSet);
        return maths::CChecksum::calculate(seed, m_LeafSet);
    }

private:
    //! Get an element of \p set by name.
    static const T* element(const TWordTypePrVec& set, const std::string& name) {
        TWord word = ms_Dictionary.word(name);
        TWordTypePrVecCItr i = element(set, word);
        return (i != set.end() && i->first == word) ? &i->second : nullptr;
    }

    //! Get the element corresponding to \p word if it exists
    //! and return the end iterator otherwise.
    static TWordTypePrVecCItr element(const TWordTypePrVec& set, const TWord& word) {
        return element(const_cast<TWordTypePrVec&>(set), word);
    }

    //! Get the element corresponding to \p word if it exists
    //! and return the end iterator otherwise.
    static TWordTypePrVecItr element(TWordTypePrVec& set, const TWord& word) {
        return std::lower_bound(set.begin(), set.end(), word,
                                maths::COrderings::SFirstLess());
    }

    //! Sort \p set on its key.
    static void sort(TWordTypePrVec& set) {
        std::sort(set.begin(), set.end(), maths::COrderings::SFirstLess());
    }

    //! Propagate the set elements forwards by \p time.
    template<typename F>
    static void age(TWordTypePrVec& set, F doAge) {
        for (std::size_t i = 0; i < set.size(); ++i) {
            doAge(set[i].second);
        }
    }

private:
    //! The word dictionary. This is static on the assumption that the
    //! methods of this class will not be used before main() runs or
    //! after it returns.
    static TDictionary ms_Dictionary;

private:
    //! The value for the bucket.
    T m_BucketElement;

    //! The container for named influencer buckets.
    TWordTypePrVec m_InfluencerBucketSet;

    //! The container for named influencers.
    TWordTypePrVec m_InfluencerSet;

    //! The container for named partitions.
    TWordTypePrVec m_PartitionSet;

    //! The container for named people.
    TWordTypePrVec m_PersonSet;

    //! The container for leaves comprising distinct named
    //! (partition, person) field name pairs.
    TWordTypePrVec m_LeafSet;
};

template<typename T>
typename CHierarchicalResultsLevelSet<T>::TDictionary CHierarchicalResultsLevelSet<T>::ms_Dictionary;
}
}

#endif // INCLUDED_ml_model_CHierarchicalResultsLevelSet_h
