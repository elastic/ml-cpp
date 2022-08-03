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

#ifndef INCLUDED_ml_core_CCompressedLfuCache_h
#define INCLUDED_ml_core_CCompressedLfuCache_h

#include <core/CCompressedDictionary.h>
#include <core/CLogger.h>
#include <core/CMemoryDefStd.h>
#include <core/CPersistUtils.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <boost/unordered_map.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>

namespace ml {
namespace core {
namespace compressed_lfu_cache_detail {

//! \brief Implements a memory limited least frequently used cache.
//!
//! IMPLEMENTATION DECISIONS:\n
//! To make the implementation less complex we assume cached values are small
//! compared to the memory budget.
//
//! In order to estimate item counts we use the Space-Saving algorithm described
//! in "Efficient Computation of Frequent and Top-k Elements in Data Streams",
//! Metwally et al.
//!
//! The thread safety is delegated to the derived types which must implement the
//! read and write guards. We do not assume that any locks we take are recursive
//! so have the policy that:
//!   -# All public member functions implement taking read and write locks
//!      unless commented otherwise,
//!   -# No internal member functions take any locks,
//!   -# No member functions calls a public member function.
//!
//! \tparam KEY The key type which must support conversion to CCompressedDictionary::CWord.
//! \tparam VALUE The value type which should implement memory estimation support.
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
class CCompressedLfuCache {
public:
    using TDictionary = CCompressedDictionary<COMPRESSED_KEY_BITS / 64>;
    using TCompressedKey = typename TDictionary::CWord;
    using TCompressKey = std::function<TCompressedKey(const TDictionary&, const KEY&)>;
    using TComputeValueCallback = std::function<VALUE(KEY)>;
    using TReadValueCallback = std::function<void(const VALUE&, bool)>;

public:
    // Construction is only exposed to derived types.
    virtual ~CCompressedLfuCache() = default;
    CCompressedLfuCache(const CCompressedLfuCache&) = delete;
    CCompressedLfuCache& operator=(const CCompressedLfuCache&) = delete;

    //! Resize the cache so that it uses no more than \p maximumMemory.
    //!
    //! \note This simply updates the limit if \p maximumMemory is larger than the
    //! current memory usage. Otherwise, it discards items.
    void resize(std::size_t maximumMemory) {

        this->guardWrite(DONT_TIME_OUT, [&] {
            m_MaximumMemory = maximumMemory;
            this->computeMemoryParameters();

            // Ensure the cache is consuming no more than the current memory limit.
            while (this->memoryUsage() > m_MaximumMemory) {
                auto itemToEvict = m_ItemStats.begin();
                m_RemovedCount += itemToEvict->count();
                this->removeFromCache(itemToEvict);
                this->maybeReallocateCache();
            }
        });
    }

    //! Lookup an item with \p key in the cache or else fall back to computing.
    //!
    //! \param[in] key The item key.
    //! \param[in] computeValue Computes the value in the case of a cache miss.
    //! \param[in] readValue Processes the value.
    //! \return True if there was a cache hit.
    //! \note For thread safe implementations we support timing out waiting for
    //! a lock since we can make progress even we skip reading or writing from
    //! the cache. This allows the caller to control the maximum latency on top
    //! of \p computeValue.
    //! \note Reading the result takes place under a lock so cut the work done
    //! in \p readValue to the minimum possible.
    //! \warning The value reference is only guaranteed to survive for the duration
    //! of \p readValue.
    //! \warning \p key is passed by value so don't forget to move it into this
    //! function if possible.
    //! \warning \p computeValue and \p readValue must not call functions on the
    //! cache.
    bool lookup(KEY key,
                const TComputeValueCallback& computeValue,
                const TReadValueCallback& readValue) {

        auto compressedKey = m_CompressKey(m_Dictionary, key);

        if (this->guardRead(TIME_OUT, [&] {
                ++m_NumberLookups;
                auto hit = this->hit(compressedKey);
                if (hit != nullptr) {
                    ++m_NumberHits;
                    readValue(*hit, true);
                    return true;
                }
                return false;
            })) {
            if (this->guardWrite(TIME_OUT, [&] {
                    this->incrementCount(compressedKey);
                }) == false) {
                ++m_LostCount;
            }
            return true;
        }

        auto value = computeValue(std::move(key));

        std::size_t itemMemoryUsage{memory::dynamicSize(value)};

        if (this->guardWrite(TIME_OUT, [&] {
                // It is possible that two values with the same key check the cache
                // before either takes the write lock. So check if this is already
                // in the cache before going any further.
                if (m_ItemCache.find(compressedKey) != m_ItemCache.end()) {
                    readValue(value, true);
                    this->incrementCount(compressedKey);
                    return;
                }

                // Update the item counts using the Space-Saving algorithm.
                std::size_t count{1};
                std::size_t lastEvictedCount{0};
                while (this->full(itemMemoryUsage)) {
                    auto itemToEvict = m_ItemStats.begin();
                    // It's possible that the cache is empty yet isn't big
                    // enough to hold this new item.
                    if (itemToEvict == m_ItemStats.end()) {
                        readValue(value, false);
                        return;
                    }
                    m_RemovedCount += lastEvictedCount;
                    lastEvictedCount = itemToEvict->count();
                    this->removeFromCache(itemToEvict);
                }
                readValue(this->insert(compressedKey, value, itemMemoryUsage,
                                       count + lastEvictedCount),
                          false);
            }) == false) {
            ++m_LostCount;
        }

        return false;
    }

    void clear() {
        this->guardWrite(DONT_TIME_OUT, [&] {
            m_NumberLookups = 0;
            m_NumberHits = 0;
            m_LostCount = 0;
            m_RemovedCount = 0;
            m_ItemsMemoryUsage = 0;
            m_ItemCache.clear();
            m_ItemStats.clear();
        });
    }

    //! Get the proportion of requests which result in a hit.
    double hitFraction() const {
        return std::min(static_cast<double>(m_NumberHits.load()) /
                            static_cast<double>(m_NumberLookups.load()),
                        1.0);
    }

    //! Get the number of items currently stored in the cache.
    std::size_t size() const {
        std::size_t result{std::numeric_limits<std::size_t>::max()};
        this->guardRead(DONT_TIME_OUT, [&] {
            result = m_ItemCache.size();
            return true;
        });
        return result;
    }

    //! Get the stats for item with \p key.
    //!
    //! \return (memory usage, hit count).
    //! \note Returns (0, 0) if the item is not in the cache.
    std::pair<std::size_t, std::uint64_t> stats(const KEY& key) const {
        std::pair<std::size_t, std::uint64_t> result{0, 0};
        auto compressedKey = m_CompressKey(m_Dictionary, key);
        this->guardRead(DONT_TIME_OUT, [&] {
            auto item = m_ItemCache.find(compressedKey);
            if (item != m_ItemCache.end()) {
                result = std::make_pair(item->second.stats()->memoryUsage(),
                                        item->second.stats()->count());
            }
            return true;
        });
        return result;
    }

    //! Get the amount of memory the cache is consuming.
    std::size_t memoryUsage() const {
        std::size_t result{std::numeric_limits<std::size_t>::max()};
        this->guardRead(DONT_TIME_OUT, [&] {
            result = this->unguardedMemoryUsage();
            return true;
        });
        return result;
    }

    //! Check cache invariants.
    //!
    //! \return True if all cache invariants hold.
    bool checkInvariants() const {

        bool result{true};

        this->guardRead(DONT_TIME_OUT, [&] {
            if (m_ItemCache.size() != m_ItemStats.size()) {
                LOG_ERROR(<< "Size mismatch " << m_ItemCache.size() << " vs "
                          << m_ItemStats.size() << ".");
                result = false;
            }

            std::size_t itemsMemoryUsage{0};
            std::uint64_t totalCount{m_RemovedCount};

            for (auto stats = m_ItemStats.begin(); stats != m_ItemStats.end(); ++stats) {
                auto item = m_ItemCache.find(stats->key());
                if (item == m_ItemCache.end()) {
                    LOG_ERROR(<< "Missing item '" << stats->key().print() << "'.");
                    result = false;
                } else if (stats != item->second.stats()) {
                    LOG_ERROR(<< "Link mismatch for '" << stats->key().print() << "'.");
                    result = false;
                }
                itemsMemoryUsage += stats->memoryUsage();
                totalCount += stats->count();
            }

            if (itemsMemoryUsage != m_ItemsMemoryUsage) {
                LOG_ERROR(<< "Memory mismatch " << itemsMemoryUsage << " vs "
                          << m_ItemsMemoryUsage << ".");
                result = false;
            }
            // We may be missing counts if a single addition caused multiple
            // evictions, or if updates can time out
            if (m_NumberLookups - m_LostCount != totalCount) {
                LOG_ERROR(<< "Count mismatch " << m_NumberLookups.load() << " (less "
                          << m_LostCount.load() << " lost) vs " << totalCount << ".");
                result = false;
            }
            return true;
        });

        return result;
    }

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(CStatePersistInserter& inserter) const {
        this->guardRead(DONT_TIME_OUT, [&] {
            CPersistUtils::persist(BUCKET_COUNT_SEQUENCE_TAG, m_BucketCountSequence, inserter);
            CPersistUtils::persist(ITEM_CACHE_TAG, m_ItemCache, inserter);
            CPersistUtils::persist(ITEMS_MEMORY_USAGE_TAG, m_ItemsMemoryUsage, inserter);
            CPersistUtils::persist(ITEM_STATS_TAG, m_ItemStats, inserter);
            CPersistUtils::persist(NUMBER_HITS_TAG, m_NumberHits, inserter);
            CPersistUtils::persist(NUMBER_LOOKUPS_TAG, m_NumberLookups, inserter);
            return true;
        });
    }

    //! Populate the object from serialized data.
    //!
    //! \warning This is not thread safe since we expect to restore before the
    //! cache is being used.
    bool acceptRestoreTraverser(CStateRestoreTraverser& traverser) {
        do {
            const std::string& name{traverser.name()};
            RESTORE_WITH_UTILS(BUCKET_COUNT_SEQUENCE_TAG, m_BucketCountSequence)
            RESTORE_WITH_UTILS(ITEM_CACHE_TAG, m_ItemCache)
            RESTORE_WITH_UTILS(ITEMS_MEMORY_USAGE_TAG, m_ItemsMemoryUsage)
            RESTORE_WITH_UTILS(ITEM_STATS_TAG, m_ItemStats)
            RESTORE_WITH_UTILS(NUMBER_HITS_TAG, m_NumberHits)
            RESTORE_WITH_UTILS(NUMBER_LOOKUPS_TAG, m_NumberLookups)
        } while (traverser.next());

        for (auto stats = m_ItemStats.begin(); stats != m_ItemStats.end(); ++stats) {
            m_ItemCache[stats->key()].stats(stats);
        }

        this->checkRestoredInvariants();

        return true;
    }

protected:
    using TReadFunc = std::function<bool()>;
    using TWriteFunc = std::function<void()>;

protected:
    CCompressedLfuCache(std::size_t maximumMemory, TCompressKey compressKey)
        : m_MaximumMemory{maximumMemory}, m_CompressKey{std::move(compressKey)} {
        this->computeMemoryParameters();
    }

private:
    //! \brief A cache item hit frequency and memory usage.
    class CCacheItemStats {
    public:
        static constexpr bool dynamicSizeAlwaysZero() { return true; }

    public:
        CCacheItemStats() = default;
        CCacheItemStats(const TCompressedKey& key, std::size_t memoryUsage, std::size_t count)
            : m_Key{key}, m_MemoryUsage{memoryUsage}, m_Count{count} {}

        bool operator<(const CCacheItemStats& rhs) const {
            // Break ties by _increasing_ memory usage so we evict largest items first.
            return m_Count < rhs.count() ||
                   (m_Count == rhs.count() && m_MemoryUsage > rhs.memoryUsage());
        }

        TCompressedKey key() const { return m_Key; }
        std::size_t memoryUsage() const { return m_MemoryUsage; }
        std::size_t count() const { return m_Count; }
        void count(std::size_t count) { m_Count = count; }

        void acceptPersistInserter(CStatePersistInserter& inserter) const {
            CPersistUtils::persist(COUNT_TAG, m_Count, inserter);
            CPersistUtils::persist(KEY_TAG, m_Key, inserter);
            CPersistUtils::persist(MEMORY_USAGE_TAG, m_MemoryUsage, inserter);
        }

        bool acceptRestoreTraverser(CStateRestoreTraverser& traverser) {
            do {
                const std::string& name{traverser.name()};
                RESTORE_WITH_UTILS(COUNT_TAG, m_Count)
                RESTORE_WITH_UTILS(KEY_TAG, m_Key)
                RESTORE_WITH_UTILS(MEMORY_USAGE_TAG, m_MemoryUsage)
            } while (traverser.next());
            return true;
        }

    private:
        static const std::string COUNT_TAG;
        static const std::string KEY_TAG;
        static const std::string MEMORY_USAGE_TAG;

    private:
        TCompressedKey m_Key;
        std::size_t m_MemoryUsage{0};
        std::uint64_t m_Count{0};
    };

    using TCacheItemStatsSet = std::multiset<CCacheItemStats>;
    using TCacheItemStatsSetItr = typename TCacheItemStatsSet::iterator;

    //! \brief A cache item wrapper which holds a reference to its stats.
    class CCacheItem {
    public:
        //! We purposely disable direct memory estimation for cache items because
        //! we want to avoid quadratic complexity using core::memory::dynamicSize.
        //! Instead we estimate memory usage for each item we add and remove from
        //! the cache and maintain a running total.
        static constexpr bool dynamicSizeAlwaysZero() { return true; }

    public:
        CCacheItem() = default;
        explicit CCacheItem(VALUE value) : m_Value{std::move(value)} {}

        const VALUE& value() const { return m_Value; }
        TCacheItemStatsSetItr stats() const { return m_Stats; }
        void stats(TCacheItemStatsSetItr stats) { m_Stats = stats; }

        void acceptPersistInserter(CStatePersistInserter& inserter) const {
            CPersistUtils::persist(VALUE_TAG, m_Value, inserter);
        }

        bool acceptRestoreTraverser(CStateRestoreTraverser& traverser) {
            do {
                const std::string& name{traverser.name()};
                RESTORE_WITH_UTILS(VALUE_TAG, m_Value)
            } while (traverser.next());
            return true;
        }

    private:
        static const std::string VALUE_TAG;

    private:
        VALUE m_Value;
        TCacheItemStatsSetItr m_Stats;
    };

    using TSizeVec = std::vector<std::size_t>;
    using TCompressedKeyCacheItemUMap = typename TDictionary::template TWordTUMap<CCacheItem>;
    using TCompressedKeyCacheItemUMapCItr = typename TCompressedKeyCacheItemUMap::const_iterator;

    static constexpr bool TIME_OUT{true};
    static constexpr bool DONT_TIME_OUT{false};

private:
    virtual bool guardRead(bool timeOut, TReadFunc func) const = 0;
    virtual bool guardWrite(bool timeOut, TWriteFunc func) = 0;
    virtual bool updatesCanTimeOut() const = 0;

    void maybeReallocateCache() {
        auto shrunk = std::lower_bound(m_BucketCountSequence.begin(),
                                       m_BucketCountSequence.end(),
                                       m_ItemCache.bucket_count());
        if (shrunk != m_BucketCountSequence.begin()) {
            --shrunk;
            double shrunkLoadFactor{static_cast<double>(m_ItemCache.bucket_count()) /
                                    static_cast<double>(*shrunk) *
                                    static_cast<double>(m_ItemCache.load_factor())};
            if (shrunkLoadFactor < static_cast<double>(m_ItemCache.max_load_factor())) {
                // The only way we can actually reclaim memory is to copy
                // the contents of the cache into a new map.
                TCompressedKeyCacheItemUMap copy{*shrunk - 1};
                for (auto & [ compressedKey, value ] : m_ItemCache) {
                    copy.emplace(compressedKey, std::move(value));
                }
                m_ItemCache.swap(copy);
            }
        }
    }

    const VALUE* hit(const TCompressedKey& key) const {
        auto item = m_ItemCache.find(key);
        return item != m_ItemCache.end() ? &item->second.value() : nullptr;
    }

    void incrementCount(const TCompressedKey& key) {
        auto item = m_ItemCache.find(key);
        if (item != m_ItemCache.end()) {
            auto node = m_ItemStats.extract(item->second.stats());
            node.value().count(node.value().count() + 1);
            item->second.stats(m_ItemStats.insert(std::move(node)));
        }
    }

    void removeFromCache(TCacheItemStatsSetItr itemToEvict) {
        m_ItemCache.erase(itemToEvict->key());
        m_ItemsMemoryUsage -= itemToEvict->memoryUsage();
        m_ItemStats.erase(itemToEvict);
    }

    const VALUE& insert(const TCompressedKey& key,
                        VALUE value,
                        std::size_t itemMemoryUsage,
                        std::size_t count) {
        m_ItemsMemoryUsage += itemMemoryUsage;
        auto item = m_ItemCache.emplace(key, std::move(value)).first;
        auto stats = m_ItemStats.emplace_hint(m_ItemStats.begin(), key,
                                              itemMemoryUsage, count);
        item->second.stats(stats);
        return item->second.value();
    }

    bool full(std::size_t itemMemoryUsage) const {
        std::size_t memory{this->unguardedMemoryUsage() + itemMemoryUsage +
                           sizeof(typename TCompressedKeyCacheItemUMap::value_type) +
                           memory::storageNodeOverhead(m_ItemCache) +
                           sizeof(typename TCacheItemStatsSet::value_type) +
                           memory::storageNodeOverhead(m_ItemStats)};
        if (this->needToResizeItemCache()) {
            memory += static_cast<std::size_t>(
                (static_cast<double>(this->nextItemCacheBucketCount()) /
                     static_cast<double>(m_ItemCache.bucket_count()) -
                 1.0) *
                static_cast<double>(memory::dynamicSize(m_ItemCache) -
                                    m_ItemCache.size() *
                                        sizeof(typename TCompressedKeyCacheItemUMap::value_type)));
        }
        return memory > m_MaximumMemory;
    }

    std::size_t unguardedMemoryUsage() const {
        return m_ItemsMemoryUsage + // overheads
               memory::dynamicSize(m_ItemCache) + memory::dynamicSize(m_ItemStats) +
               memory::dynamicSize(m_BucketCountSequence);
    }

    bool needToResizeItemCache() const {
        return static_cast<double>(m_ItemCache.load_factor()) +
                   1.0 / static_cast<double>(m_ItemCache.bucket_count()) >
               static_cast<double>(m_ItemCache.max_load_factor());
    }

    std::size_t nextItemCacheBucketCount() const {
        return *std::upper_bound(m_BucketCountSequence.begin(),
                                 m_BucketCountSequence.end(), m_ItemCache.bucket_count());
    }

    void computeMemoryParameters() {
        // We expect resizing to cycle through a fixed sequence of bucket counts.
        // This deduces and caches that sequence.

        typename TDictionary::template TWordTUMap<std::size_t> test;

        m_BucketCountSequence.clear();
        m_BucketCountSequence.push_back(test.bucket_count());

        // Ignore heap memory allocated by VALUE since we only need a lower bound.
        std::size_t memoryPerItem{2 * sizeof(TCompressedKey) +
                                  sizeof(std::uint64_t) + sizeof(VALUE)};
        for (std::size_t i = 0; memoryPerItem * i < m_MaximumMemory; ++i) {
            auto translator = m_Dictionary.translator();
            translator.add(i);
            test.emplace(translator.word(), i);
            if (test.bucket_count() != m_BucketCountSequence.back()) {
                m_BucketCountSequence.push_back(test.bucket_count());
            }
        }

        LOG_TRACE(<< "bucket sequence = " << m_BucketCountSequence);
    }

    void checkRestoredInvariants() const {
        VIOLATES_INVARIANT(m_ItemCache.size(), !=, m_ItemStats.size());
        for (const auto & [ key, item ] : m_ItemCache) {
            VIOLATES_INVARIANT_NO_EVALUATION(item.stats(), ==, m_ItemStats.end());
            VIOLATES_INVARIANT_NO_EVALUATION(item.stats(), ==, TCacheItemStatsSetItr{});
            VIOLATES_INVARIANT_NO_EVALUATION(key, !=, item.stats()->key());
        }
        for (const auto& stats : m_ItemStats) {
            VIOLATES_INVARIANT_NO_EVALUATION(m_ItemCache.find(stats.key()), ==,
                                             m_ItemCache.end());
        }
    }

private:
    static const std::string BUCKET_COUNT_SEQUENCE_TAG;
    static const std::string ITEM_CACHE_TAG;
    static const std::string ITEMS_MEMORY_USAGE_TAG;
    static const std::string ITEM_STATS_TAG;
    static const std::string NUMBER_HITS_TAG;
    static const std::string NUMBER_LOOKUPS_TAG;

private:
    // Align the lower bits to prevent false sharing with other members. This
    // should switch to std::hardware_destructive_interference_size when it's
    // fully supported.
    alignas(64) std::atomic<std::uint64_t> m_NumberLookups{0};
    alignas(64) std::atomic<std::uint64_t> m_NumberHits{0};
    //! We can lose count if we fail to obtain a write lock within the timeout
    //! period when trying to increment cache item counts.
    alignas(64) std::atomic<std::uint64_t> m_LostCount{0};
    std::uint64_t m_RemovedCount{0};
    std::size_t m_MaximumMemory{0};
    std::size_t m_ItemsMemoryUsage{0};
    TCompressKey m_CompressKey;
    TDictionary m_Dictionary;
    TCompressedKeyCacheItemUMap m_ItemCache;
    TCacheItemStatsSet m_ItemStats;
    TSizeVec m_BucketCountSequence;
};

// clang-format off
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::BUCKET_COUNT_SEQUENCE_TAG{"bucket_count_sequence"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::ITEM_CACHE_TAG{"item_cache"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::ITEMS_MEMORY_USAGE_TAG{"items_memory_usage"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::ITEM_STATS_TAG{"item_stats"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::NUMBER_HITS_TAG{"number_hits"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::NUMBER_LOOKUPS_TAG{"number_lookups"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::CCacheItem::VALUE_TAG{"value"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::CCacheItemStats::COUNT_TAG{"count"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::CCacheItemStats::KEY_TAG{"key"};
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS>
const std::string CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>::CCacheItemStats::MEMORY_USAGE_TAG{"memory_usage"};
// clang-format on
}

//! \brief A thread safe memory limited least frequently used cache.
//!
//! DESCRIPTION:\n
//! This provides a memory limited least frequently used cache. The maximum memory
//! the cache will consume is supplied as a parameter. It is optimized for the case
//! that the cache keys are large. We use a specified size hash (in number of bits)
//! for the keys. Using sufficient bits means the chance of a collision is effectively
//! zero for reasonably sized caches (for more details see CCompressedDictionary).
//! For this reason, no safe guards are provided in the case of a hash collision.
//!
//! Example usage caching the result of expensive operation and writing to stdout:
//! \code{.cpp}
//! double myExpensiveOperation(std::vector<double> input);
//! ...
//! core::CConcurrentCompressedLfuCache<std::vector<double>, double> cache{
//!     64 * 1024,
//!     [](const auto& dictionary, const auto& key) {
//!         auto translator = dictionary.translator();
//!         translator.add(key);
//!         return translator.word();
//!     }};
//! for (auto input : inputs) {
//!    core::async(
//!        core::defaultAsyncExecutor(),
//!        [&] {
//!            cache.lookup(
//!                std::move(input),
//!                [](auto input_) { return myExpensiveOperation(std::move(input_)); },
//!                [](const auto& result) { std::cout << result << std::endl; });
//!        });
//! }
//! \endcode
//!
//! IMPLEMENTATION DECISIONS:\n
//! \see compressed_lfu_cache_detail::CCompressedLfuCache.
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS = 128>
class CConcurrentCompressedLfuCache final
    : public compressed_lfu_cache_detail::CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS> {
public:
    using TBase = compressed_lfu_cache_detail::CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>;
    using TDictionary = typename TBase::TDictionary;
    using TCompressKey = typename TBase::TCompressKey;

public:
    //! \param[in] maximumMemory The maximum memory the cache will consume in bytes.
    //! \param[in] maximumWait The maximum time to wait at a critical section.
    //! \param[in] compressKey Compresses the key. \see CCompressedDictionary for
    //! details.
    CConcurrentCompressedLfuCache(std::size_t maximumMemory,
                                  std::chrono::milliseconds maximumWait,
                                  TCompressKey compressKey)
        : TBase{maximumMemory, compressKey}, m_MaximumWait{maximumWait} {}

private:
    using TReadFunc = typename TBase::TReadFunc;
    using TWriteFunc = typename TBase::TWriteFunc;

private:
    bool guardRead(bool timeOut, TReadFunc func) const override {
        if (timeOut) {
            std::shared_lock<std::shared_timed_mutex> lock{m_Mutex, m_MaximumWait};
            return lock.owns_lock() && func();
        }
        std::shared_lock<std::shared_timed_mutex> lock{m_Mutex};
        return func();
    }

    bool guardWrite(bool timeOut, TWriteFunc func) override {
        if (timeOut) {
            std::unique_lock<std::shared_timed_mutex> lock{m_Mutex, m_MaximumWait};
            if (lock.owns_lock()) {
                func();
                return true;
            }
            return false;
        }
        std::unique_lock<std::shared_timed_mutex> lock{m_Mutex};
        func();
        return true;
    }

    bool updatesCanTimeOut() const override { return true; }

private:
    std::chrono::milliseconds m_MaximumWait;
    mutable std::shared_timed_mutex m_Mutex;
};

//! \brief A memory limited least frequently used cache.
//!
//! DESCRIPTION:\n
//! This version is provided to remove all overheads from locking and unlocking
//! mutexes if it is known that the cache is only called from a single thread.
//! Otherwise, its behaviour is identical to CConcurrentCompressedLfuCache.
//!
//! Example usage caching the result of expensive operation and writing to stdout:
//! \code{.cpp}
//! double myExpensiveOperation(std::vector<double> input);
//! ...
//! CCompressedLfuCache<std::vector<double>, double> cache{
//!     64 * 1024,
//!     [](const auto& dictionary, const auto& key) {
//!         auto translator = dictionary.translator();
//!         translator.add(key);
//!         return translator.word();
//!     }};
//! for (auto input : inputs) {
//!    cache.lookup(
//!        std::move(input),
//!        [](auto input_) { return myExpensiveOperation(std::move(input_)); },
//!        [](const auto& result) { std::cout << result << std::endl; });
//! }
//! \endcode
//!
//! IMPLEMETATION:
//! \see compressed_lfu_cache_detail::CCompressedLfuCache.
template<typename KEY, typename VALUE, std::size_t COMPRESSED_KEY_BITS = 128>
class CCompressedLfuCache final
    : public compressed_lfu_cache_detail::CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS> {
public:
    using TBase = compressed_lfu_cache_detail::CCompressedLfuCache<KEY, VALUE, COMPRESSED_KEY_BITS>;
    using TDictionary = typename TBase::TDictionary;
    using TCompressKey = typename TBase::TCompressKey;

public:
    //! \param[in] maximumMemory The maximum memory the cache will consume in bytes.
    //! \param[in] compressKey Compresses the key. \see CCompressedDictionary for details.
    CCompressedLfuCache(std::size_t maximumMemory, TCompressKey compressKey)
        : TBase{maximumMemory, compressKey} {}

private:
    using TReadFunc = typename TBase::TReadFunc;
    using TWriteFunc = typename TBase::TWriteFunc;

private:
    bool guardRead(bool, TReadFunc func) const override { return func(); }
    bool guardWrite(bool, TWriteFunc func) override {
        func();
        return true;
    }
    bool updatesCanTimeOut() const override { return false; }
};
}
}

#endif // INCLUDED_ml_core_CCompressedLfuCache_h
