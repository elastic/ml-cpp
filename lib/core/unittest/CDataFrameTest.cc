/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CAlignment.h>
#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CDataFrameRowSlice.h>
#include <core/CPackedBitVector.h>
#include <core/Concurrency.h>

#include <test/CRandomNumbers.h>
#include <test/CTestTmpDir.h>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/unordered_map.hpp>

#include <functional>
#include <mutex>
#include <vector>

BOOST_AUTO_TEST_SUITE(CDataFrameTest)

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

std::function<void(TFloatVecItr, std::int32_t&)>
makeWriter(TFloatVec& components, std::size_t cols, std::size_t i) {
    return [&components, cols, i](TFloatVecItr col, std::int32_t& docHash) mutable {
        docHash = static_cast<std::int32_t>(i);
        for (std::size_t end_ = i + cols; i < end_; ++i, ++col) {
            *col = components[i];
        }
    };
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
                LOG_DEBUG(<< "mismatch for row " << i / cols);
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

class CTestFixture {
public:
    CTestFixture() { core::startDefaultAsyncExecutor(); }

    ~CTestFixture() { core::stopDefaultAsyncExecutor(); }
};

BOOST_FIXTURE_TEST_CASE(testInMainMemoryBasicReadWrite, CTestFixture) {

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
            auto frameAndDirectory = core::makeMainStorageDataFrame(
                cols, core::CAlignment::E_Unaligned, capacity, readWriteToStoreAsync);
            auto frame = std::move(frameAndDirectory.first);

            for (std::size_t i = 0; i < end; i += cols) {
                frame->writeRow(makeWriter(components, cols, i));
            }
            frame->finishWritingRows();

            bool successful;
            bool passed{true};
            std::size_t i{0};
            std::tie(std::ignore, successful) = frame->readRows(
                1, std::bind(makeReader(components, cols, passed), std::ref(i),
                             std::placeholders::_1, std::placeholders::_2));
            BOOST_TEST_REQUIRE(successful);
            BOOST_TEST_REQUIRE(passed);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testInMainMemoryParallelRead, CTestFixture) {

    // Check we get the rows we write to the data frame and that we get balanced
    // reads per thread.

    std::size_t cols{10};
    std::size_t capacity{1000};

    for (std::size_t rows : {4000, 5000, 6000}) {
        LOG_DEBUG(<< "Testing " << rows << " rows");

        TFloatVec components{testData(rows, cols)};

        auto frameAndDirectory = core::makeMainStorageDataFrame(
            cols, core::CAlignment::E_Unaligned, capacity);
        auto frame = std::move(frameAndDirectory.first);
        for (std::size_t i = 0; i < components.size(); i += cols) {
            frame->writeRow(makeWriter(components, cols, i));
        }
        frame->finishWritingRows();

        std::vector<CThreadReader> readers;
        bool successful;
        std::tie(readers, successful) = frame->readRows(3, CThreadReader{});
        BOOST_TEST_REQUIRE(successful);
        BOOST_REQUIRE_EQUAL(std::size_t{3}, readers.size());

        TBoolVec rowRead(rows, false);
        for (const auto& reader : readers) {
            BOOST_REQUIRE_EQUAL(false, reader.duplicates());
            BOOST_TEST_REQUIRE(reader.rowsRead().size() <= 2000);
            for (const auto& row : reader.rowsRead()) {
                BOOST_TEST_REQUIRE(std::equal(components.begin() + row.first * cols,
                                              components.begin() + (row.first + 1) * cols,
                                              row.second.begin()));
                rowRead[row.first] = true;
            }
        }

        std::size_t rowsRead(std::count(rowRead.begin(), rowRead.end(), true));
        BOOST_REQUIRE_EQUAL(rows, rowsRead);
    }
}

BOOST_FIXTURE_TEST_CASE(testOnDiskBasicReadWrite, CTestFixture) {

    // Check we get the rows we write to the data frame in the order we write them.

    std::size_t rows{5500};
    std::size_t cols{10};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    auto frameAndDirectory = core::makeDiskStorageDataFrame(
        test::CTestTmpDir::tmpDir(), cols, rows, core::CAlignment::E_Unaligned, capacity);
    auto frame = std::move(frameAndDirectory.first);

    for (std::size_t i = 0; i < components.size(); i += cols) {
        frame->writeRow(makeWriter(components, cols, i));
    }
    frame->finishWritingRows();

    bool successful;
    bool passed{true};
    std::size_t i{0};
    std::tie(std::ignore, successful) = frame->readRows(
        1, std::bind(makeReader(components, cols, passed), std::ref(i),
                     std::placeholders::_1, std::placeholders::_2));
    BOOST_TEST_REQUIRE(successful);
    BOOST_TEST_REQUIRE(passed);
}

BOOST_FIXTURE_TEST_CASE(testOnDiskParallelRead, CTestFixture) {

    // Check we get the rows we write to the data frame and that we get balanced
    // reads per thread.

    std::size_t rows{5000};
    std::size_t cols{10};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    auto frameAndDirectory = core::makeDiskStorageDataFrame(
        test::CTestTmpDir::tmpDir(), cols, rows, core::CAlignment::E_Unaligned, capacity);
    auto frame = std::move(frameAndDirectory.first);

    for (std::size_t i = 0; i < components.size(); i += cols) {
        frame->writeRow(makeWriter(components, cols, i));
    }
    frame->finishWritingRows();

    std::vector<CThreadReader> readers;
    bool successful;
    std::tie(readers, successful) = frame->readRows(3, CThreadReader{});
    BOOST_TEST_REQUIRE(successful);
    BOOST_REQUIRE_EQUAL(std::size_t{(rows + 1999) / 2000}, readers.size());

    TBoolVec rowRead(rows, false);
    for (const auto& reader : readers) {
        BOOST_REQUIRE_EQUAL(false, reader.duplicates());
        BOOST_TEST_REQUIRE(reader.rowsRead().size() <= 2000);
        for (const auto& row : reader.rowsRead()) {
            BOOST_TEST_REQUIRE(std::equal(components.begin() + row.first * cols,
                                          components.begin() + (row.first + 1) * cols,
                                          row.second.begin()));
            rowRead[row.first] = true;
        }
    }

    std::size_t rowsRead(std::count(rowRead.begin(), rowRead.end(), true));
    BOOST_REQUIRE_EQUAL(rows, rowsRead);
}

BOOST_FIXTURE_TEST_CASE(testReadRange, CTestFixture) {

    // Check we get the only the rows rows we request.

    std::size_t rows{5000};
    std::size_t cols{10};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows,
                                              core::CAlignment::E_Unaligned, capacity)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test read range " << type[t++]);

        auto frame = factory();

        for (std::size_t threads : {1, 3}) {
            LOG_DEBUG(<< "# threads = " << threads);

            for (std::size_t i = 0; i < components.size(); i += cols) {
                frame->writeRow(makeWriter(components, cols, i));
            }
            frame->finishWritingRows();

            for (std::size_t beginRowsInRange : {0, 1000, 1500}) {
                for (std::size_t endRowsInRange : {500, 2000, 5000}) {
                    LOG_DEBUG(<< "Reading [" << beginRowsInRange << ","
                              << endRowsInRange << ")");
                    bool passed{true};
                    frame->readRows(
                        threads, beginRowsInRange, endRowsInRange,
                        [&](TRowItr beginRows, TRowItr endRows) {
                            for (auto row = beginRows; row != endRows; ++row) {
                                if (passed && (row->index() < beginRowsInRange ||
                                               row->index() >= endRowsInRange)) {
                                    LOG_ERROR(<< "row " << row->index()
                                              << " out of range [" << beginRowsInRange
                                              << "," << endRowsInRange << ")");
                                    passed = false;
                                }
                                if (passed) {
                                    auto column = components.begin() + row->index() * cols;
                                    for (std::size_t i = 0; i < cols; ++i, ++column) {
                                        if ((*row)[i] != *column) {
                                            LOG_ERROR(<< "Unexpected column value for "
                                                      << row->index());
                                            passed = false;
                                        }
                                    }
                                }
                            }
                        });
                    BOOST_TEST_REQUIRE(passed);
                }
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testWriteRange, CTestFixture) {

    // Check we get the only write the rows we specify.

    std::size_t rows{5000};
    std::size_t cols{10};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(test::CTestTmpDir::tmpDir(), cols, rows,
                                              core::CAlignment::E_Unaligned, capacity)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test write range " << type[t++]);

        for (std::size_t threads : {1, 3}) {
            LOG_DEBUG(<< "# threads = " << threads);

            for (std::size_t beginRowsInRange : {0, 1000, 1500}) {
                for (std::size_t endRowsInRange : {500, 2000, 5000}) {

                    auto inRange = [beginRowsInRange, endRowsInRange](std::size_t index) {
                        return index >= beginRowsInRange && index < endRowsInRange;
                    };

                    auto frame = factory();
                    for (std::size_t i = 0; i < components.size(); i += cols) {
                        frame->writeRow(makeWriter(components, cols, i));
                    }
                    frame->finishWritingRows();

                    LOG_DEBUG(<< "Writing [" << beginRowsInRange << ","
                              << endRowsInRange << ")");

                    frame->writeColumns(threads, beginRowsInRange, endRowsInRange,
                                        [&](TRowItr beginRows, TRowItr endRows) {
                                            for (auto row = beginRows;
                                                 row != endRows; ++row) {
                                                (*row)[0] += 2.0;
                                            }
                                        });

                    bool passed{true};
                    frame->readRows(threads, [&](TRowItr beginRows, TRowItr endRows) {
                        for (auto row = beginRows; row != endRows; ++row) {
                            if (passed) {
                                auto column = components.begin() + row->index() * cols;
                                passed = ((*row)[0] ==
                                          *column + (inRange(row->index()) ? 2.0 : 0.0));
                                for (std::size_t i = 1; i < cols; ++i, ++column) {
                                    passed = ((*row)[i] != *column);
                                }
                                if (passed == false) {
                                    LOG_ERROR(<< "Unexpected column value for "
                                              << row->index());
                                }
                            }
                        }
                    });
                    BOOST_TEST_REQUIRE(passed);
                }
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testMemoryUsage, CTestFixture) {

    // This asserts on the memory used by the different types of data frames. This
    // is meant to catch large regressions in memory usage and as such the thresholds
    // shouldn't be treated as hard.

    std::size_t rows{10000};
    std::size_t cols{10};
    std::size_t capacity{5000};
    TFloatVec components{testData(rows, cols)};

    const std::string& rootDirectory{test::CTestTmpDir::tmpDir()};
    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(
                   rootDirectory, cols, rows, core::CAlignment::E_Unaligned,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    // Memory usage should be less than:
    //   1) 1075 + 4 times the root directory length bytes for on disk, and
    //   2) data size + doc ids size + 900 byte overhead in main memory.
    std::size_t maximumMemory[]{1075 + 4 * rootDirectory.length(),
                                rows * (cols + 1) * 4 + 900};

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test memory usage " << type[t]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols) {
            frame->writeRow(makeWriter(components, cols, i));
        }
        frame->finishWritingRows();

        LOG_DEBUG(<< "Memory = " << frame->memoryUsage()
                  << ", limit = " << maximumMemory[t]);
        BOOST_TEST_REQUIRE(frame->memoryUsage() < maximumMemory[t++]);
    }
}

BOOST_FIXTURE_TEST_CASE(testReserve, CTestFixture) {

    // Check that we preserve the visible rows after reserving.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(
                   test::CTestTmpDir::tmpDir(), cols, rows, core::CAlignment::E_Unaligned,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    LOG_DEBUG(<< "*** Test reserve before write ***");

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test reserve " << type[t++]);

        auto frame = factory();
        frame->reserve(1, 20);

        for (std::size_t i = 0; i < components.size(); i += cols) {
            frame->writeRow(makeWriter(components, cols, i));
        }
        frame->finishWritingRows();

        bool successful;
        bool passed{true};
        std::size_t i{0};
        std::tie(std::ignore, successful) = frame->readRows(
            1, std::bind(makeReader(components, cols, passed), std::ref(i),
                         std::placeholders::_1, std::placeholders::_2));
        BOOST_TEST_REQUIRE(successful);
        BOOST_TEST_REQUIRE(passed);
    }

    LOG_DEBUG(<< "*** Test reserve after write ***");

    t = 0;
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test reserve " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols) {
            frame->writeRow(makeWriter(components, cols, i));
        }
        frame->finishWritingRows();

        frame->reserve(2, 20);

        bool successful;
        bool passed{true};
        std::size_t i{0};
        std::tie(std::ignore, successful) = frame->readRows(
            1, std::bind(makeReader(components, cols, passed), std::ref(i),
                         std::placeholders::_1, std::placeholders::_2));
        BOOST_TEST_REQUIRE(successful);
        BOOST_TEST_REQUIRE(passed);
    }
}

BOOST_FIXTURE_TEST_CASE(testResizeColumns, CTestFixture) {

    // Test all rows are correctly resized and the extra elements are zero
    // initialized.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(
                   test::CTestTmpDir::tmpDir(), cols, rows, core::CAlignment::E_Unaligned,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test resize " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols) {
            frame->writeRow(makeWriter(components, cols, i));
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
        BOOST_TEST_REQUIRE(successful);
        BOOST_TEST_REQUIRE(passed);
    }
}

BOOST_FIXTURE_TEST_CASE(testWriteColumns, CTestFixture) {

    // Test writing of extra column values.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t extraCols{3};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols + extraCols)};

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(
                   test::CTestTmpDir::tmpDir(), cols, rows, core::CAlignment::E_Unaligned,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test write columns " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols + extraCols) {
            frame->writeRow(makeWriter(components, cols, i));
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
        BOOST_TEST_REQUIRE(successful);
        BOOST_TEST_REQUIRE(passed);
    }
}

BOOST_FIXTURE_TEST_CASE(testDocHashes, CTestFixture) {

    // Test we preserve the document hashes we write originally.

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t extraCols{3};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols + extraCols)};

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(
                   test::CTestTmpDir::tmpDir(), cols, rows, core::CAlignment::E_Unaligned,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test write columns " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols + extraCols) {
            frame->writeRow(makeWriter(components, cols, i));
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
        std::tie(std::ignore, successful) = frame->readRows(1, [
            &passed, cols, extraCols, expectedDocHash = std::int32_t{0}
        ](TRowItr beginRows, TRowItr endRows) mutable {
            for (auto row = beginRows; row != endRows; ++row) {
                if (passed && row->docHash() != expectedDocHash) {
                    LOG_ERROR(<< "Got doc hash " << row->docHash() << " expected "
                              << expectedDocHash << " for row " << row->index());
                    passed = false;
                }
                expectedDocHash += static_cast<std::int32_t>(cols + extraCols);
            }
        });
        BOOST_TEST_REQUIRE(successful);
        BOOST_TEST_REQUIRE(passed);
    }
}

BOOST_FIXTURE_TEST_CASE(testRowMask, CTestFixture) {

    // Test we read only the rows in a mask.

    using TSizeVec = std::vector<std::size_t>;

    TSizeVec rowsRead;

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t extraCols{3};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols + extraCols)};

    test::CRandomNumbers rng;

    TFactoryFunc makeOnDisk = [=] {
        return core::makeDiskStorageDataFrame(
                   boost::filesystem::current_path().string(), cols, rows,
                   core::CAlignment::E_Unaligned, capacity,
                   core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TFactoryFunc makeMainMemory = [=] {
        return core::makeMainStorageDataFrame(cols, core::CAlignment::E_Unaligned, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        LOG_DEBUG(<< "Test read rows " << type[t++]);

        auto frame = factory();

        for (std::size_t i = 0; i < components.size(); i += cols + extraCols) {
            frame->writeRow(makeWriter(components, cols, i));
        }
        frame->finishWritingRows();

        for (auto numberThreads : {1, 3}) {
            LOG_DEBUG(<< "# threads = " << numberThreads);

            TSizeVec readRowsIndices;

            // Edge cases:
            //   1) Mask doesn't intercept row range
            //   2) Ends of range and slice

            std::size_t ranges[][2]{{101, 3998}, {95, 5000}};
            TSizeVec rangeRowMaskIndices[]{
                {}, {95, 96, 97, 98, 99, 100, 3999, 4000, 4998, 4999}};

            for (auto i : {0, 1}) {
                core::CPackedBitVector rowMask{true};
                rowMask.extend(true, 100);
                rowMask.extend(false, 3898);
                rowMask.extend(true, 2);
                rowMask.extend(false, 997);
                rowMask.extend(true, 2);

                auto results =
                    frame
                        ->readRows(
                            numberThreads, ranges[i][0], ranges[i][1],
                            core::bindRetrievableState(
                                [](TSizeVec& readerReadRowsIndices,
                                   TRowItr beginRows, TRowItr endRows) mutable {
                                    for (auto row = beginRows; row != endRows; ++row) {
                                        readerReadRowsIndices.push_back(row->index());
                                    }
                                },
                                TSizeVec{}),
                            &rowMask)
                        .first;

                readRowsIndices.clear();
                for (const auto& result : results) {
                    readRowsIndices.insert(readRowsIndices.end(),
                                           result.s_FunctionState.begin(),
                                           result.s_FunctionState.end());
                }
                std::sort(readRowsIndices.begin(), readRowsIndices.end());

                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(rangeRowMaskIndices[i]),
                                    core::CContainerPrinter::print(readRowsIndices));
            }

            TSizeVec strides;
            TSizeVec rowMaskIndices;
            for (std::size_t i = 0; i < 200; ++i) {
                rng.generateUniformSamples(0, 50, 150, strides);

                core::CPackedBitVector rowMask{strides[0] == 0};
                for (auto stride : strides) {
                    if (rowMask.size() + stride > rows) {
                        break;
                    }
                    if (stride > 0) {
                        rowMask.extend(false, stride);
                        rowMask.extend(true);
                    }
                }
                rowMask.extend(false, rows - rowMask.size());
                rowMaskIndices.assign(rowMask.beginOneBits(), rowMask.endOneBits());

                auto results =
                    frame
                        ->readRows(
                            numberThreads, 0, rows,
                            core::bindRetrievableState(
                                [](TSizeVec& readerReadRowsIndices,
                                   TRowItr beginRows, TRowItr endRows) mutable {
                                    for (auto row = beginRows; row != endRows; ++row) {
                                        readerReadRowsIndices.push_back(row->index());
                                    }
                                },
                                TSizeVec{}),
                            &rowMask)
                        .first;

                readRowsIndices.clear();
                for (const auto& result : results) {
                    readRowsIndices.insert(readRowsIndices.end(),
                                           result.s_FunctionState.begin(),
                                           result.s_FunctionState.end());
                }
                std::sort(readRowsIndices.begin(), readRowsIndices.end());

                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(rowMaskIndices),
                                    core::CContainerPrinter::print(readRowsIndices));
            }
        }
    }
}

BOOST_FIXTURE_TEST_CASE(testAlignment, CTestFixture) {

    // Test all the rows have the requested alignment.

    using TAlignedFactoryFunc =
        std::function<std::unique_ptr<core::CDataFrame>(core::CAlignment::EType)>;

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};

    test::CRandomNumbers rng;

    TAlignedFactoryFunc makeOnDisk = [=](core::CAlignment::EType alignment) {
        return core::makeDiskStorageDataFrame(
                   boost::filesystem::current_path().string(), cols, rows, alignment,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TAlignedFactoryFunc makeMainMemory = [=](core::CAlignment::EType alignment) {
        return core::makeMainStorageDataFrame(cols, alignment, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        for (auto alignment : {core::CAlignment::E_Aligned8, core::CAlignment::E_Aligned16,
                               core::CAlignment::E_Aligned32}) {
            LOG_DEBUG(<< "Test aligned " << alignment << " " << type[t]);

            auto frame = factory(alignment);

            for (std::size_t i = 0; i < components.size(); i += cols) {
                frame->writeRow(makeWriter(components, cols, i));
            }
            frame->finishWritingRows();

            frame->readRows(1, [alignment](TRowItr beginRows, TRowItr endRows) {
                for (auto row = beginRows; row != endRows; ++row) {
                    BOOST_TEST_REQUIRE(core::CAlignment::isAligned(row->data(), alignment));
                }
            });
        }
        ++t;
    }
}

BOOST_FIXTURE_TEST_CASE(testAlignedExtraColumns, CTestFixture) {

    // Test all the rows have the requested alignment.

    using TAlignedFactoryFunc =
        std::function<std::unique_ptr<core::CDataFrame>(core::CAlignment::EType)>;

    std::size_t rows{5000};
    std::size_t cols{15};
    std::size_t capacity{1000};
    TFloatVec components{testData(rows, cols)};
    core::CDataFrame::TSizeAlignmentPrVec extraColumns{
        {2, core::CAlignment::E_Unaligned},
        {3, core::CAlignment::E_Aligned16},
        {1, core::CAlignment::E_Unaligned}};
    test::CRandomNumbers rng;

    TAlignedFactoryFunc makeOnDisk = [=](core::CAlignment::EType alignment) {
        return core::makeDiskStorageDataFrame(
                   boost::filesystem::current_path().string(), cols, rows, alignment,
                   capacity, core::CDataFrame::EReadWriteToStorage::E_Async)
            .first;
    };
    TAlignedFactoryFunc makeMainMemory = [=](core::CAlignment::EType alignment) {
        return core::makeMainStorageDataFrame(cols, alignment, capacity,
                                              core::CDataFrame::EReadWriteToStorage::E_Sync)
            .first;
    };

    std::string type[]{"on disk", "main memory"};
    std::size_t t{0};
    for (const auto& factory : {makeOnDisk, makeMainMemory}) {
        for (auto alignment : {core::CAlignment::E_Aligned16, core::CAlignment::E_Aligned32}) {
            LOG_DEBUG(<< "Test aligned " << alignment << " " << type[t]);

            auto frame = factory(alignment);

            for (std::size_t i = 0; i < components.size(); i += cols) {
                frame->writeRow(makeWriter(components, cols, i));
            }
            frame->finishWritingRows();

            auto offsets = frame->resizeColumns(1, extraColumns);

            LOG_DEBUG(<< core::CContainerPrinter::print(offsets));

            //frame->readRows(1, [alignment](TRowItr beginRows, TRowItr endRows) {
            //    for (auto row = beginRows; row != endRows; ++row) {
            //        BOOST_TEST_REQUIRE(core::CAlignment::isAligned(row->data(), alignment));
            //    }
            //});
        }
        ++t;
    }
}

BOOST_AUTO_TEST_SUITE_END()
