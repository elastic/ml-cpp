/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CDataFrameTest.h"

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CDataFrameRowSlice.h>
#include <core/Concurrency.h>

#include <test/CRandomNumbers.h>

#include <boost/filesystem/path.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <mutex>
#include <vector>

using namespace ml;

namespace {
using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TFloatVec = std::vector<core::CFloatStorage>;
using TFloatVecItr = TFloatVec::iterator;
using TFloatVecCItr = TFloatVec::const_iterator;
using TSizeFloatVecUMap = boost::unordered_map<std::size_t, TFloatVec>;
using TRowItr = core::CDataFrame::TRowItr;
using TReadFunc = std::function<void(TRowItr, TRowItr)>;
using TFactoryFunc = std::function<std::unique_ptr<core::CDataFrame>()>;

TFloatVec testData(std::size_t numberRows, std::size_t numberColumns) {
    test::CRandomNumbers rng;
    TDoubleVec uniform;
    rng.generateUniformSamples(0.0, 10.0, numberRows * numberColumns, uniform);
    TFloatVec components(uniform.begin(), uniform.end());
    uniform.clear();
    uniform.shrink_to_fit();
    return components;
}

std::function<void(std::size_t&, TRowItr, TRowItr)>
makeReader(TFloatVec& components, std::size_t cols, bool& passed) {
    return [&components, cols, &passed](std::size_t& i, TRowItr beginRows,
                                        TRowItr endRows) mutable {
        TFloatVec expectedRow(cols);
        TFloatVec row(cols);
        for (auto j = beginRows; j != endRows; ++j) {
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
    void operator()(TRowItr beginRows, TRowItr endRows) {
        TSizeFloatVecUMap::iterator entry;
        for (auto row = beginRows; row != endRows; ++row) {
            bool added;
            std::tie(entry, added) = m_Rows.emplace(row->index(), TFloatVec{});
            m_Duplicates |= (added == false);
            entry->second.resize(row->numberColumns());
            row->copyTo(entry->second.begin());
        }
    }

    bool duplicates() const { return m_Duplicates; }

    const TSizeFloatVecUMap& rowsRead() const { return m_Rows; }

private:
    bool m_Duplicates = false;
    TSizeFloatVecUMap m_Rows;
};
}

void CDataFrameTest::setUp() {
    core::startDefaultAsyncExecutor();
}

void CDataFrameTest::tearDown() {
    core::stopDefaultAsyncExecutor();
}

void CDataFrameTest::testInMainMemoryBasicReadWrite() {

    // Check we get the rows we write to the data frame in the order we write them.

    std::size_t rows{5000};
    std::size_t cols{10};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    std::string sync[]{"sync", "async"};
    for (auto readWriteToStoreAsync : {core::CDataFrame::EReadWriteToStorage::E_Sync,
                                       core::CDataFrame::EReadWriteToStorage::E_Async}) {
        LOG_DEBUG(<< "Read write to store "
                  << sync[static_cast<int>(readWriteToStoreAsync)]);

        for (auto end : {500 * cols, 2000 * cols, components.size()}) {
            auto frame = core::makeMainStorageDataFrame(cols, capacity, readWriteToStoreAsync);

            for (std::size_t i = 0; i < end; i += cols) {
                auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                    for (std::size_t end_ = i + cols; i < end_; ++i, ++output) {
                        *output = components[i];
                    }
                };
                frame->writeRow(writer);
            }
            frame->finishWritingRows();

            bool successful;
            bool passed{true};
            std::size_t i{0};
            std::tie(std::ignore, successful) = frame->readRows(
                1, std::bind(makeReader(components, cols, passed), std::ref(i),
                             std::placeholders::_1, std::placeholders::_2));
            CPPUNIT_ASSERT(successful);
            CPPUNIT_ASSERT(passed);
        }
    }
}

void CDataFrameTest::testInMainMemoryParallelRead() {

    // Check we get the rows we write to the data frame and that we get balanced
    // reads per thread.

    std::size_t cols{10};
    std::size_t capacity{1000};

    for (std::size_t rows : {4000, 5000, 6000}) {
        LOG_DEBUG(<< "Testing " << rows << " rows");

        TFloatVec components{testData(rows, cols)};

        auto frame = core::makeMainStorageDataFrame(cols, capacity);
        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame->writeRow(writer);
        }
        frame->finishWritingRows();

        std::vector<CThreadReader> readers;
        bool successful;
        std::tie(readers, successful) = frame->readRows(3, CThreadReader{});
        CPPUNIT_ASSERT(successful);
        CPPUNIT_ASSERT_EQUAL(std::size_t{3}, readers.size());

        TBoolVec rowRead(rows, false);
        for (const auto& reader : readers) {
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
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    auto frame = core::makeDiskStorageDataFrame(
        boost::filesystem::current_path().string(), cols, rows, capacity);

    for (std::size_t i = 0; i < components.size(); i += cols) {
        auto writer = [&components, cols, i](TFloatVecItr output) mutable {
            for (std::size_t end = i + cols; i < end; ++i, ++output) {
                *output = components[i];
            }
        };
        frame->writeRow(writer);
    }
    frame->finishWritingRows();

    bool successful;
    bool passed{true};
    std::size_t i{0};
    std::tie(std::ignore, successful) = frame->readRows(
        1, std::bind(makeReader(components, cols, passed), std::ref(i),
                     std::placeholders::_1, std::placeholders::_2));
    CPPUNIT_ASSERT(successful);
    CPPUNIT_ASSERT(passed);
}

void CDataFrameTest::testOnDiskParallelRead() {

    std::size_t rows{5000};
    std::size_t cols{10};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    auto frame = core::makeDiskStorageDataFrame(
        boost::filesystem::current_path().string(), cols, rows, capacity);

    for (std::size_t i = 0; i < components.size(); i += cols) {
        auto writer = [&components, cols, i](TFloatVecItr output) mutable {
            for (std::size_t end = i + cols; i < end; ++i, ++output) {
                *output = components[i];
            }
        };
        frame->writeRow(writer);
    }
    frame->finishWritingRows();

    std::vector<CThreadReader> readers;
    bool successful;
    std::tie(readers, successful) = frame->readRows(3, CThreadReader{});
    CPPUNIT_ASSERT(successful);
    CPPUNIT_ASSERT_EQUAL(std::size_t{(rows + 1999) / 2000}, readers.size());

    TBoolVec rowRead(rows, false);
    for (const auto& reader : readers) {
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
    std::size_t capacity{5000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = std::bind(
        &core::makeDiskStorageDataFrame, boost::filesystem::current_path().string(),
        cols, rows, capacity, core::CDataFrame::EReadWriteToStorage::E_Async);
    TFactoryFunc makeMainMemory =
        std::bind(&core::makeMainStorageDataFrame, cols, capacity,
                  core::CDataFrame::EReadWriteToStorage::E_Sync);

    // 750 bytes and data size + 200 byte overhead.
    std::size_t maximumMemory[]{750, rows * cols * 4 + 200};

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test memory usage " << type[t]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame->writeRow(writer);
        }
        frame->finishWritingRows();

        LOG_DEBUG(<< "Memory = " << frame->memoryUsage());
        CPPUNIT_ASSERT(frame->memoryUsage() < maximumMemory[t++]);
    }
}

void CDataFrameTest::testReserve() {

    // Check that we preserve the visible rows after reserving.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = std::bind(
        &core::makeDiskStorageDataFrame, boost::filesystem::current_path().string(),
        cols, rows, capacity, core::CDataFrame::EReadWriteToStorage::E_Async);
    TFactoryFunc makeMainMemory =
        std::bind(&core::makeMainStorageDataFrame, cols, capacity,
                  core::CDataFrame::EReadWriteToStorage::E_Sync);

    LOG_DEBUG(<< "*** Test reserve before write ***");

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test reserve " << type[t++]);

        auto frame = factory();
        frame->reserve(1, 20);

        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame->writeRow(writer);
        }
        frame->finishWritingRows();

        bool successful;
        bool passed{true};
        std::size_t i{0};
        std::tie(std::ignore, successful) = frame->readRows(
            1, std::bind(makeReader(components, cols, passed), std::ref(i),
                         std::placeholders::_1, std::placeholders::_2));
        CPPUNIT_ASSERT(successful);
        CPPUNIT_ASSERT(passed);
    }

    LOG_DEBUG(<< "*** Test reserve after write ***");

    t = 0;
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test reserve " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame->writeRow(writer);
        }
        frame->finishWritingRows();

        frame->reserve(2, 20);

        bool successful;
        bool passed{true};
        std::size_t i{0};
        std::tie(std::ignore, successful) = frame->readRows(
            1, std::bind(makeReader(components, cols, passed), std::ref(i),
                         std::placeholders::_1, std::placeholders::_2));
        CPPUNIT_ASSERT(successful);
        CPPUNIT_ASSERT(passed);
    }
}

void CDataFrameTest::testResizeColumns() {

    // Test all rows are correctly resized and the extra elements are zero
    // initialized.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = std::bind(
        &core::makeDiskStorageDataFrame, boost::filesystem::current_path().string(),
        cols, rows, capacity, core::CDataFrame::EReadWriteToStorage::E_Async);
    TFactoryFunc makeMainMemory =
        std::bind(&core::makeMainStorageDataFrame, cols, capacity,
                  core::CDataFrame::EReadWriteToStorage::E_Sync);

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test resize " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame->writeRow(writer);
        }
        frame->finishWritingRows();

        frame->resizeColumns(2, 18);

        bool successful;
        bool passed{true};
        std::tie(std::ignore, successful) =
            frame->readRows(1, [&passed](TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    if (passed && row->numberColumns() != 18) {
                        LOG_DEBUG(<< "got " << row->numberColumns() << " columns");
                        passed = false;
                    } else if (passed) {
                        for (auto i = row->data() + 15; i != row->data() + 18; ++i) {
                            if (*i != 0.0) {
                                LOG_DEBUG(<< "expected zeros got " << *i);
                                passed = false;
                            }
                        }
                    }
                }
            });
        CPPUNIT_ASSERT(successful);
        CPPUNIT_ASSERT(passed);
    }
}

void CDataFrameTest::testWriteColumns() {

    // Test writing of extra column values.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t extraCols{3};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols + extraCols)};

    TFactoryFunc makeOnDisk = std::bind(
        &core::makeDiskStorageDataFrame, boost::filesystem::current_path().string(),
        cols, rows, capacity, core::CDataFrame::EReadWriteToStorage::E_Async);
    TFactoryFunc makeMainMemory =
        std::bind(&core::makeMainStorageDataFrame, cols, capacity,
                  core::CDataFrame::EReadWriteToStorage::E_Sync);

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test write columns " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols + extraCols) {
            auto writer = [&components, cols, i](TFloatVecItr output) mutable {
                for (std::size_t end = i + cols; i < end; ++i, ++output) {
                    *output = components[i];
                }
            };
            frame->writeRow(writer);
        }
        frame->finishWritingRows();

        frame->resizeColumns(2, 18);
        frame->writeColumns(2, [&](TRowItr beginRows, TRowItr endRows) mutable {
            for (auto row = beginRows; row != endRows; ++row) {
                for (std::size_t j = 15; j < 18; ++j) {
                    std::size_t index{row->index() * (cols + extraCols) + j};
                    row->writeColumn(j, components[index]);
                }
            }
        });

        bool successful;
        bool passed{true};
        std::size_t i{0};
        std::tie(std::ignore, successful) = frame->readRows(
            1, std::bind(makeReader(components, cols + extraCols, passed),
                         std::ref(i), std::placeholders::_1, std::placeholders::_2));
        CPPUNIT_ASSERT(successful);
        CPPUNIT_ASSERT(passed);
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
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testReserve", &CDataFrameTest::testReserve));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testResizeColumns", &CDataFrameTest::testResizeColumns));
    suiteOfTests->addTest(new CppUnit::TestCaller<CDataFrameTest>(
        "CDataFrameTest::testWriteColumns", &CDataFrameTest::testWriteColumns));

    return suiteOfTests;
}
