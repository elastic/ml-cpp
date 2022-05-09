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

#include <maths/time_series/CCountMinSketch.h>

#include <core/CHashing.h>
#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CChecksum.h>
#include <maths/common/COrderings.h>

#include <boost/math/constants/constants.hpp>

#include <cmath>
#include <iomanip>

namespace ml {
namespace maths {
namespace time_series {
namespace {
const std::string TOTAL_COUNT_TAG("a");
const std::string ROWS_TAG("b");
const std::string COLUMNS_TAG("c");
const std::string CATEGORY_COUNTS_TAG("d");
const std::string SKETCH_TAG("e");

// Nested tags.
const std::string HASHES_TAG("a");
const std::string COUNTS_TAG("b");

const char DELIMITER(':');
const char PAIR_DELIMITER(';');
}

CCountMinSketch::CCountMinSketch(std::size_t rows, std::size_t columns)
    : m_Rows(rows), m_Columns(columns), m_TotalCount(0.0),
      m_Sketch(TUInt32FloatPrVec()) {
}

CCountMinSketch::CCountMinSketch(core::CStateRestoreTraverser& traverser)
    : m_Rows(0), m_Columns(0), m_TotalCount(0.0), m_Sketch() {
    if (traverser.traverseSubLevel([this](auto& traverser_) {
            return this->acceptRestoreTraverser(traverser_);
        }) == false) {
        traverser.setBadState();
    }
}

void CCountMinSketch::swap(CCountMinSketch& other) noexcept {
    if (this == &other) {
        return;
    }

    std::swap(m_Rows, other.m_Rows);
    std::swap(m_Columns, other.m_Columns);
    std::swap(m_TotalCount, other.m_TotalCount);

    try {
        TUInt32FloatPrVec* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
        if (counts != nullptr) {
            TUInt32FloatPrVec* otherCounts =
                std::get_if<TUInt32FloatPrVec>(&other.m_Sketch);
            if (otherCounts != nullptr) {
                counts->swap(*otherCounts);
            } else {
                auto& otherSketch = std::get<SSketch>(other.m_Sketch);
                TUInt32FloatPrVec tmp;
                tmp.swap(*counts);
                m_Sketch = SSketch();
                std::get<SSketch>(m_Sketch).s_Hashes.swap(otherSketch.s_Hashes);
                std::get<SSketch>(m_Sketch).s_Counts.swap(otherSketch.s_Counts);
                other.m_Sketch = TUInt32FloatPrVec();
                std::get<TUInt32FloatPrVec>(other.m_Sketch).swap(tmp);
            }
        } else {
            auto& sketch = std::get<SSketch>(m_Sketch);
            auto* otherSketch = std::get_if<SSketch>(&other.m_Sketch);
            if (otherSketch != nullptr) {
                sketch.s_Hashes.swap(otherSketch->s_Hashes);
                sketch.s_Counts.swap(otherSketch->s_Counts);
            } else {
                auto& otherCounts = std::get<TUInt32FloatPrVec>(other.m_Sketch);
                TUInt32FloatPrVec tmp;
                tmp.swap(otherCounts);
                other.m_Sketch = SSketch();
                sketch.s_Hashes.swap(std::get<SSketch>(other.m_Sketch).s_Hashes);
                sketch.s_Counts.swap(std::get<SSketch>(other.m_Sketch).s_Counts);
                m_Sketch = TUInt32FloatPrVec();
                std::get<TUInt32FloatPrVec>(m_Sketch).swap(tmp);
            }
        }
    } catch (const std::exception& e) {
        LOG_ABORT(<< "Unexpected exception " << e.what());
    }
}

bool CCountMinSketch::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == ROWS_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_Rows) == false) {
                LOG_ERROR(<< "Invalid number rows in " << traverser.value());
                return false;
            }
        } else if (name == COLUMNS_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_Columns) == false) {
                LOG_ERROR(<< "Invalid number columns in " << traverser.value());
                return false;
            }
        } else if (name == TOTAL_COUNT_TAG) {
            if (m_TotalCount.fromString(traverser.value()) == false) {
                LOG_ERROR(<< "Invalid total count in " << traverser.value());
                return false;
            }
        } else if (name == CATEGORY_COUNTS_TAG) {
            m_Sketch = TUInt32FloatPrVec();
            auto& counts = std::get<TUInt32FloatPrVec>(m_Sketch);
            if (core::CPersistUtils::fromString(traverser.value(), counts, DELIMITER,
                                                PAIR_DELIMITER) == false) {
                LOG_ERROR(<< "Invalid category counts in " << traverser.value());
                return false;
            }
        } else if (name == SKETCH_TAG) {
            m_Sketch = SSketch();
            auto& sketch = std::get<SSketch>(m_Sketch);
            sketch.s_Hashes.reserve(m_Rows);
            sketch.s_Counts.reserve(m_Rows);
            if (traverser.traverseSubLevel([&](auto& traverser_) {
                    return sketch.acceptRestoreTraverser(traverser_, m_Rows, m_Columns);
                }) == false) {
                return false;
            }
        }
    } while (traverser.next());

    return true;
}
void CCountMinSketch::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(ROWS_TAG, m_Rows);
    inserter.insertValue(COLUMNS_TAG, m_Columns);
    inserter.insertValue(TOTAL_COUNT_TAG, m_TotalCount, core::CIEEE754::E_SinglePrecision);
    const TUInt32FloatPrVec* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        inserter.insertValue(CATEGORY_COUNTS_TAG,
                             core::CPersistUtils::toString(*counts, DELIMITER, PAIR_DELIMITER));
    } else {
        try {
            const auto& sketch = std::get<SSketch>(m_Sketch);
            inserter.insertLevel(SKETCH_TAG, [&](auto& inserter_) {
                sketch.acceptPersistInserter(inserter_);
            });
        } catch (const std::exception& e) {
            LOG_ABORT(<< "Unexpected exception " << e.what());
        }
    }
}

std::size_t CCountMinSketch::rows() const {
    return m_Rows;
}

std::size_t CCountMinSketch::columns() const {
    return m_Columns;
}

double CCountMinSketch::delta() const {
    const SSketch* sketch = std::get_if<SSketch>(&m_Sketch);
    if (sketch == nullptr) {
        return 0.0;
    }
    return std::exp(-static_cast<double>(m_Rows));
}

double CCountMinSketch::oneMinusDeltaError() const {
    const SSketch* sketch = std::get_if<SSketch>(&m_Sketch);
    if (sketch == nullptr) {
        return 0.0;
    }
    return std::min(boost::math::double_constants::e / static_cast<double>(m_Columns), 1.0) *
           m_TotalCount;
}

void CCountMinSketch::add(uint32_t category, double count) {
    LOG_TRACE(<< "Adding category = " << category << ", count = " << count);

    m_TotalCount += count;

    auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        auto itr = std::lower_bound(counts->begin(), counts->end(), category,
                                    common::COrderings::SFirstLess());

        if (itr == counts->end() || itr->first != category) {
            itr = counts->insert(itr, TUInt32FloatPr(category, 0.0));
        }

        itr->second += count;

        if (itr->second <= 0.0) {
            counts->erase(itr);
        } else {
            this->sketch();
        }
    } else {
        try {
            auto& sketch = std::get<SSketch>(m_Sketch);
            for (std::size_t i = 0; i < sketch.s_Hashes.size(); ++i) {
                uint32_t hash = (sketch.s_Hashes[i])(category);
                std::size_t j = static_cast<std::size_t>(hash) % m_Columns;
                sketch.s_Counts[i][j] += count;
                LOG_TRACE(<< "count (i,j) = (" << i << "," << j << ")"
                          << " -> " << sketch.s_Counts[i][j]);
            }
        } catch (const std::exception& e) {
            LOG_ABORT(<< "Unexpected exception " << e.what());
        }
    }
}

void CCountMinSketch::removeFromMap(uint32_t category) {
    auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        auto itr = std::lower_bound(counts->begin(), counts->end(), category,
                                    common::COrderings::SFirstLess());
        if (itr != counts->end() && itr->first == category) {
            counts->erase(itr);
        }
    }
}

void CCountMinSketch::age(double alpha) {
    auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        for (auto& count : *counts) {
            count.second *= alpha;
        }
    } else {
        try {
            auto& sketch = std::get<SSketch>(m_Sketch);
            for (std::size_t i = 0; i < sketch.s_Counts.size(); ++i) {
                for (std::size_t j = 0; j < sketch.s_Counts[i].size(); ++j) {
                    sketch.s_Counts[i][j] *= alpha;
                }
            }
        } catch (const std::exception& e) {
            LOG_ABORT(<< "Unexpected exception " << e.what());
        }
    }
}

double CCountMinSketch::totalCount() const {
    return m_TotalCount;
}

double CCountMinSketch::count(uint32_t category) const {
    using TMinAccumulator = common::CBasicStatistics::COrderStatisticsStack<double, 1>;

    const auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        auto itr = std::lower_bound(counts->begin(), counts->end(), category,
                                    common::COrderings::SFirstLess());

        return itr == counts->end() || itr->first != category
                   ? 0.0
                   : static_cast<double>(itr->second);
    }

    TMinAccumulator result;
    try {
        const auto& sketch = std::get<SSketch>(m_Sketch);
        for (std::size_t i = 0; i < sketch.s_Hashes.size(); ++i) {
            uint32_t hash = (sketch.s_Hashes[i])(category);
            std::size_t j = static_cast<std::size_t>(hash) % m_Columns;
            LOG_TRACE(<< "count (i,j) = (" << i << "," << j << ")"
                      << " <- " << sketch.s_Counts[i][j]);
            result.add(sketch.s_Counts[i][j]);
        }
    } catch (const std::exception& e) {
        LOG_ABORT(<< "Unexpected exception " << e.what());
    }
    return result.count() > 0 ? result[0] : 0.0;
}

double CCountMinSketch::fraction(uint32_t category) const {
    return this->count(category) / m_TotalCount;
}

bool CCountMinSketch::sketched() const {
    return std::get_if<SSketch>(&m_Sketch) != nullptr;
}

uint64_t CCountMinSketch::checksum(uint64_t seed) const {
    seed = common::CChecksum::calculate(seed, m_Rows);
    seed = common::CChecksum::calculate(seed, m_Columns);
    seed = common::CChecksum::calculate(seed, m_TotalCount);

    const auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts == nullptr) {
        try {
            const auto& sketch = std::get<SSketch>(m_Sketch);
            seed = common::CChecksum::calculate(seed, sketch.s_Hashes);
            return common::CChecksum::calculate(seed, sketch.s_Counts);
        } catch (const std::exception& e) {
            LOG_ABORT(<< "Unexpected exception " << e.what());
        }
    }
    return common::CChecksum::calculate(seed, *counts);
}

void CCountMinSketch::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CCountMinSketch");
    const auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        core::CMemoryDebug::dynamicSize("m_Counts", *counts, mem);
    } else {
        try {
            const auto& sketch = std::get<SSketch>(m_Sketch);
            mem->addItem("SSketch", sizeof(SSketch));
            core::CMemoryDebug::dynamicSize("sketch", sketch, mem);
            core::CMemoryDebug::dynamicSize("s_Hashes", sketch.s_Hashes, mem);
            core::CMemoryDebug::dynamicSize("s_Counts", sketch.s_Counts, mem);
        } catch (const std::exception& e) {
            LOG_ABORT(<< "Unexpected exception " << e.what());
        }
    }
}

std::size_t CCountMinSketch::memoryUsage() const {
    std::size_t mem = 0;
    const auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        mem += core::CMemory::dynamicSize(*counts);
    } else {
        try {
            const auto& sketch = std::get<SSketch>(m_Sketch);
            mem += sizeof(SSketch);
            mem += core::CMemory::dynamicSize(sketch.s_Hashes);
            mem += core::CMemory::dynamicSize(sketch.s_Counts);
        } catch (const std::exception& e) {
            LOG_ABORT(<< "Unexpected exception " << e.what());
        }
    }
    return mem;
}

void CCountMinSketch::sketch() {
    static const std::size_t FLOAT_SIZE = sizeof(common::CFloatStorage);
    static const std::size_t HASH_SIZE =
        sizeof(core::CHashing::CUniversalHash::CUInt32UnrestrictedHash);
    static const std::size_t PAIR_SIZE = sizeof(TUInt32FloatPr);
    static const std::size_t VEC_SIZE = sizeof(TUInt32FloatPrVec);
    static const std::size_t SKETCH_SIZE = sizeof(SSketch);

    auto* counts = std::get_if<TUInt32FloatPrVec>(&m_Sketch);
    if (counts != nullptr) {
        std::size_t countsSize = VEC_SIZE + PAIR_SIZE * counts->capacity();
        std::size_t sketchSize = SKETCH_SIZE + m_Rows * (m_Columns * FLOAT_SIZE + HASH_SIZE);

        if (countsSize > sketchSize) {
            if (counts->capacity() > counts->size() &&
                counts->size() < (sketchSize - VEC_SIZE) / PAIR_SIZE) {
                TUInt32FloatPrVec shrunk;
                shrunk.reserve((sketchSize - VEC_SIZE) / PAIR_SIZE);
                shrunk.assign(counts->begin(), counts->end());
                counts->swap(shrunk);
                return;
            }

            LOG_TRACE(<< "Sketching " << counts->size() << " counts");

            TUInt32FloatPrVec counts_;
            counts_.swap(*counts);
            m_TotalCount = 0.0;
            m_Sketch = SSketch(m_Rows, m_Columns);
            for (std::size_t i = 0; i < counts_.size(); ++i) {
                this->add(counts_[i].first, counts_[i].second);
            }
        }
    }
}

CCountMinSketch::SSketch::SSketch(std::size_t rows, std::size_t columns)
    : s_Counts(rows, TFloatVec(columns, 0.0)) {
    core::CHashing::CUniversalHash::generateHashes(rows, s_Hashes);
}

bool CCountMinSketch::SSketch::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser,
                                                      std::size_t rows,
                                                      std::size_t columns) {
    do {
        const std::string& name = traverser.name();
        if (name == HASHES_TAG) {
            core::CHashing::CUniversalHash::CFromString hashFromString(PAIR_DELIMITER);
            if (core::CPersistUtils::fromString(traverser.value(), hashFromString,
                                                s_Hashes, DELIMITER) == false ||
                s_Hashes.size() != rows) {
                LOG_ERROR(<< "Invalid hashes in " << traverser.value());
                return false;
            }
        } else if (name == COUNTS_TAG) {
            s_Counts.push_back(TFloatVec());
            if (core::CPersistUtils::fromString(traverser.value(), s_Counts.back(),
                                                DELIMITER) == false ||
                s_Counts.back().size() != columns) {
                LOG_ERROR(<< "Invalid counts in " << traverser.value());
                return false;
            }
        }
    } while (traverser.next());

    if (s_Counts.size() != rows) {
        LOG_ERROR(<< "Unexpected number of counts " << s_Counts.size()
                  << ", number of rows " << rows);
        return false;
    }
    return true;
}

void CCountMinSketch::SSketch::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CHashing::CUniversalHash::CToString hashToString(PAIR_DELIMITER);
    inserter.insertValue(
        HASHES_TAG, core::CPersistUtils::toString(s_Hashes, hashToString, DELIMITER));
    for (const auto& count : s_Counts) {
        inserter.insertValue(COUNTS_TAG, core::CPersistUtils::toString(count, DELIMITER));
    }
}
}
}
}
