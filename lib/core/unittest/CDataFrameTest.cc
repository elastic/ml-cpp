/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameTest.h"

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CDataFrameRowSlice.h>

#include <test/CRandomNumbers.h>

#include <boost/filesystem/path.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <vector>

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TFloatVec = std::vector<core::CFloatStorage>;
using TFloatVecItr = TFloatVec::iterator;
using TFloatVecCItr = TFloatVec::const_iterator;
using TSizeFloatVecUMap = boost::unordered_map<std::size_t, TFloatVec>;
using TRowCItr = core::CDataFrame::TRowCItr;
using TReadFunc = std::function<void(TRowCItr, TRowCItr)>;

TFloatVec testData(std::size_t numberRows, std::size_t numberColumns) {
    test::CRandomNumbers rng;
    TDoubleVec uniform;
    rng.generateUniformSamples(0.0, 10.0, numberRows * numberColumns, uniform);
    TFloatVec components(uniform.begin(), uniform.end());
    uniform.clear();
    uniform.shrink_to_fit();
    return components;
}

std::function<void(std::size_t&, TRowCItr, TRowCItr)>
makeReader(TFloatVec& components, std::size_t cols, bool& passed) {
    return [&components, cols, &passed](std::size_t& i, TRowCItr begin, TRowCItr end) mutable {
        TFloatVec expectedRow(cols);
        TFloatVec row(cols);
        for (auto j = begin; j != end; ++j) {
            std::copy(components.begin() + i, components.begin() + i + cols,
                      expectedRow.begin());
            j->copyTo(row.begin());
            if (passed && expectedRow != row) {
                LOG_DEBUG(<< "mismatch for row " << i / cols)
                LOG_DEBUG(<< "expected = " << core::CContainerPrinter::print(expectedRow));
                LOG_DEBUG(<< "actual   = " << core::CContainerPrinter::print(row));
                passed = false;
            } else if (passed && i / cols != j->index()) {
                LOG_DEBUG(<< "mismatch for row index " << i / cols << " vs " << j->index());
                passed = false;
            }
            i += cols;
        }
    };
}

class CThreadReader {
public:
    void operator()(TRowCItr begin, TRowCItr end) {
        TSizeFloatVecUMap::iterator entry;
        for (auto i = begin; i != end; ++i) {
            bool added;
            std::tie(entry, added) = m_Rows.emplace(i->index(), TFloatVec{});
            m_Duplicates |= (added == false);
            entry->second.resize(i->numberColumns());
            i->copyTo(entry->second.begin());
        }
    }

    bool duplicates() const {
        return m_Duplicates;
    }

    const TSizeFloatVecUMap& rowsRead() const {
        return m_Rows;
    }

private:
    bool m_Duplicates = false;
    TSizeFloatVecUMap m_Rows;
};
}

void CDataFrameTest::testInMainMemoryBasicReadWrite() {

    // Check we get the rows we write to the data frame in the order we write them.

    std::size_t rows{5000};
    std::size_t cols{10};
    TFloatVec components{testData(rows, cols)};

    bool passed{true};
    auto reader = makeReader(components, cols, passed);

    std::string type[]{"raw", "compressed"};
    std::string sync[]{"sync", "async"};
    std::size_t t{0};
    for (const auto& factory : {core::makeMainStorageDataFrame}) {
        LOG_DEBUG(<< "Test read/write " << type[t++]);

        for (auto readWriteToStoreAsync :
             {core::CDataFrame::EReadWriteToStorage::E_Sync,
              core::CDataFrame::EReadWriteToStorage::E_Async}) {
            LOG_DEBUG(<< "Read write to store "
                      << sync[static_cast<int>(readWriteToStoreAsync)]);

            for (auto end : {500 * cols, 2000 * cols, components.size()}) {
                core::CDataFrame frame{factory(cols, 1000, readWriteToStoreAsync)};

                for (std::size_t i = 0; i < end; i += cols) {
                    auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                        for (std::size_t end_ = i + cols; i < end_; ++i, ++output) {
                            *output = components[i];
                        }
                    };
                    frame.writeRow(writer);
                }
                frame.finishWritingRows();

                bool successful;
                std::size_t i{0};
                std::tie(std::ignore, successful) = frame.readRows(
                    1, std::bind(reader, std::ref(i), std::placeholders::_1,
                                 std::placeholders::_2));
                CPPUNIT_ASSERT(successful);
                CPPUNIT_ASSERT(passed);
            }
        }
    }
}

void CDataFrameTest::testInMainMemoryParallelRead() {

    // Check we get the rows we write to the data frame and that we get balanced
    // reads per thread.

    std::size_t cols{10};

    for (std::size_t rows : {4000, 5000, 6000}) {
        LOG_DEBUG(<< "Testing " << rows << " rows");

        TFloatVec components{testData(rows, cols)};

        core::CDataFrame frame{core::makeMainStorageDataFrame(cols, 1000)};
        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame.writeRow(writer);
        }
        frame.finishWritingRows();

        std::vector<TReadFunc> readers;
        bool successful;
        std::tie(readers, successful) = frame.readRows(3, CThreadReader{});
        CPPUNIT_ASSERT(successful);
        CPPUNIT_ASSERT_EQUAL(std::size_t{(rows + 1999) / 2000}, readers.size());

        TBoolVec rowRead(rows, false);
        for (const auto& reader_ : readers) {
            const auto& reader = *reader_.target<CThreadReader>();
            CPPUNIT_ASSERT_EQUAL(false, reader.duplicates());
            CPPUNIT_ASSERT(reader.rowsRead().size() <= 2000);
            for (const auto& row : reader.rowsRead()) {
                CPPUNIT_ASSERT(std::equal(components.begin() + row.first * cols,
                                          components.begin() + (row.first + 1) * cols,
                                          row.second.begin()));
                rowRead[row.first] = true;
            }
        }

        std::size_t rowsRead(std::count(rowRead.begin(), rowRead.end(), true));
        CPPUNIT_ASSERT_EQUAL(rows, rowsRead);
    }
}

void CDataFrameTest::testOnDiskBasicReadWrite() {

    // Check we get the rows we write to the data frame in the order we write them.

    std::size_t rows{5500};
    std::size_t cols{10};
    TFloatVec components{testData(rows, cols)};

    bool passed{true};
    auto reader = makeReader(components, cols, passed);

    core::CDataFrame frame{core::makeDiskStorageDataFrame(
        boost::filesystem::current_path().string(), cols, rows, 1000)};

    for (std::size_t i = 0; i < components.size(); i += cols) {
        auto writer = [&components, cols, i](TFloatVecItr output) mutable {
            for (std::size_t end = i + cols; i < end; ++i, ++output) {
                *output = components[i];
            }
        };
        frame.writeRow(writer);
    }
    frame.finishWritingRows();

    bool result;
    std::size_t i{0};
    std::tie(std::ignore, result) = frame.readRows(
        1, std::bind(reader, std::ref(i), std::placeholders::_1, std::placeholders::_2));
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(passed);
}

void CDataFrameTest::testOnDiskParallelRead() {

    std::size_t rows{5000};
    std::size_t cols{10};

    TFloatVec components{testData(rows, cols)};

    core::CDataFrame frame{core::makeDiskStorageDataFrame(
        boost::filesystem::current_path().string(), cols, rows, 1000)};

    for (std::size_t i = 0; i < components.size(); i += cols) {
        auto writer = [&components, cols, i](TFloatVecItr output) mutable {
            for (std::size_t end = i + cols; i < end; ++i, ++output) {
                *output = components[i];
            }
        };
        frame.writeRow(writer);
    }
    frame.finishWritingRows();

    std::vector<TReadFunc> readers;
    bool successful;
    std::tie(readers, successful) = frame.readRows(3, CThreadReader{});
    CPPUNIT_ASSERT(successful);
    CPPUNIT_ASSERT_EQUAL(std::size_t{(rows + 1999) / 2000}, readers.size());

    TBoolVec rowRead(rows, false);
    for (const auto& reader_ : readers) {
        const auto& reader = *reader_.target<CThreadReader>();
        CPPUNIT_ASSERT_EQUAL(false, reader.duplicates());
        CPPUNIT_ASSERT(reader.rowsRead().size() <= 2000);
        for (const auto& row : reader.rowsRead()) {
            CPPUNIT_ASSERT(std::equal(components.begin() + row.first * cols,
                                      components.begin() + (row.first + 1) * cols,
                                      row.second.begin()));
            rowRead[row.first] = true;
        }
    }

    std::size_t rowsRead(std::count(rowRead.begin(), rowRead.end(), true));
    CPPUNIT_ASSERT_EQUAL(rows, rowsRead);
}

void CDataFrameTest::testMemoryUsage() {

    // This asserts on the memory used by the different types of data frames. This
    // is meant to catch large regressions in memory usage and as such the thresholds
    // shouldn't be treated as hard.

    std::size_t rows{10000};
    std::size_t cols{10};
    TFloatVec components{testData(rows, cols)};

    using TFactoryFunc = std::function<core::CDataFrame()>;

    TFactoryFunc makeOnDisk = std::bind(
        &core::makeDiskStorageDataFrame, boost::filesystem::current_path().string(),
        cols, 10000, 5000, core::CDataFrame::EReadWriteToStorage::E_Async);
    TFactoryFunc makeMainMemory = std::bind(&core::makeMainStorageDataFrame, cols, 5000,
                                            core::CDataFrame::EReadWriteToStorage::E_Sync);

    // 600 bytes and data size + 200 byte overhead.
    std::size_t maximumMemory[]{600, 10000 * cols * 4 + 200};

    std::size_t f{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        core::CDataFrame frame{factory()};

        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame.writeRow(writer);
        }
        frame.finishWritingRows();

        LOG_DEBUG(<< "Memory = " << frame.memoryUsage());
        CPPUNIT_ASSERT(frame.memoryUsage() < maximumMemory[f++]);
    }
}

CppUnit::Test* CDataFrameTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CDataFrameTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testInMainMemoryBasicReadWrite",
        &CDataFrameTest::testInMainMemoryBasicReadWrite));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testInMainMemoryParallelRead",
        &CDataFrameTest::testInMainMemoryParallelRead));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testOnDiskBasicReadWrite", &CDataFrameTest::testOnDiskBasicReadWrite));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testOnDiskParallelRead", &CDataFrameTest::testOnDiskParallelRead));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testMemoryUsage", &CDataFrameTest::testMemoryUsage));

    return suiteOfTests;
}
