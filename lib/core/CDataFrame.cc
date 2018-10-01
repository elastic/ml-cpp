/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CDataFrame.h>

#include <core/CConcurrentWrapper.h>
#include <core/CDataFrameRowSlice.h>
#include <core/CLogger.h>
#include <core/CMemory.h>

#include <boost/make_unique.hpp>

#include <algorithm>
#include <future>
#include <memory>

namespace ml {
namespace core {
namespace data_frame_detail {

CRowCRef::CRowCRef(std::size_t index, TFloatVecCItr beginColumns, TFloatVecCItr endColumns)
    : m_Index{index}, m_BeginColumns{beginColumns}, m_EndColumns{endColumns} {
}

CFloatStorage CRowCRef::operator[](std::size_t i) const {
    return *(m_BeginColumns + i);
}

std::size_t CRowCRef::index() const {
    return m_Index;
}

std::size_t CRowCRef::numberColumns() const {
    return std::distance(m_BeginColumns, m_EndColumns);
}

CRowConstIterator::CRowConstIterator(std::size_t numberColumns, std::size_t index, TFloatVecCItr base)
    : m_NumberColumns{numberColumns}, m_Index{index}, m_Base{base} {
}

bool CRowConstIterator::operator==(const CRowConstIterator& rhs) const {
    return m_Base == rhs.m_Base;
}

bool CRowConstIterator::operator!=(const CRowConstIterator& rhs) const {
    return m_Base != rhs.m_Base;
}

CRowCRef CRowConstIterator::operator*() const {
    return CRowCRef{m_Index, m_Base, m_Base + m_NumberColumns};
}

CRowCPtr CRowConstIterator::operator->() const {
    return CRowCPtr{m_Index, m_Base, m_Base + m_NumberColumns};
}

CRowConstIterator& CRowConstIterator::operator++() {
    ++m_Index;
    m_Base += m_NumberColumns;
    return *this;
}

CRowConstIterator CRowConstIterator::operator++(int) {
    CRowConstIterator result{*this};
    ++m_Index;
    m_Base += m_NumberColumns;
    return result;
}

TFloatVecCItr CRowConstIterator::base() const {
    return m_Base;
}
}
using namespace data_frame_detail;

namespace {
using TFloatVec = CDataFrame::TFloatVec;
using TSizeFloatVecPrQueue = CDataFrame::TSizeFloatVecPrQueue;
using TRowSlicePtr = CDataFrame::TRowSlicePtr;
using TRowSlicePtrVec = CDataFrame::TRowSlicePtrVec;
using TRowSlicePtrVecCItr = TRowSlicePtrVec::const_iterator;

//! \brief A worker function object used to asynchronously write slices
//! to storage.
class CAsyncDataFrameStorageWriter final {
public:
    using TWriteSliceToStoreFunc = CDataFrame::TWriteSliceToStoreFunc;

public:
    CAsyncDataFrameStorageWriter(std::size_t sliceCapacity,
                                 const TWriteSliceToStoreFunc& writeSliceToStore)
        : m_SliceCapacity{sliceCapacity}, m_WriteSliceToStore{writeSliceToStore} {}

    TRowSlicePtrVec operator()(TSizeFloatVecPrQueue& sliceQueue) {
        // Loop until we're signaled to exit by an empty slice.

        TRowSlicePtrVec slices;

        std::size_t firstRow;
        TFloatVec slice;
        for (;;) {
            std::tie(firstRow, slice) = sliceQueue.pop();
            std::size_t size{slice.size()};
            if (size > 0) {
                auto storedSlice = m_WriteSliceToStore(firstRow, std::move(slice));
                slices.push_back(std::move(storedSlice));
            } else {
                break;
            }
        }

        return slices;
    }

private:
    std::size_t m_SliceCapacity;
    TWriteSliceToStoreFunc m_WriteSliceToStore;
};

//! \brief Reads a collection of slices from a data frame.
class CDataFrameRowSliceReader final {
public:
    using TReadFunc = CDataFrame::TReadFunc;
    using TReadFuncRef = std::reference_wrapper<TReadFunc>;
    using TReadSliceFromStoreFunc = CDataFrame::TReadSliceFromStoreFunc;

public:
    CDataFrameRowSliceReader(TReadFunc& reader,
                             std::size_t numberColumns,
                             CDataFrame::EReadWriteToStorage asyncReadFromStore,
                             const TReadSliceFromStoreFunc& readSliceFromStore)
        : m_Reader{reader}, m_NumberColumns{numberColumns},
          m_AsyncReadFromStore{asyncReadFromStore}, m_ReadSliceFromStore{readSliceFromStore} {}

    //! Read all slices in [\p beginSlices, \p endSlices) passing to the
    //! callback supplied to the constructor.
    //!
    //! \return False if the rows couldn't all be read.
    bool operator()(TRowSlicePtrVecCItr beginSlices, TRowSlicePtrVecCItr endSlices) {
        std::size_t firstRow;
        CDataFrameRowSliceHandle sliceBeingRead;
        std::size_t cols{m_NumberColumns};

        switch (m_AsyncReadFromStore) {
        case CDataFrame::EReadWriteToStorage::E_Async: {
            // The slices get read from storage on the thread executing this
            // function each slice is then concurrently read by the callback
            // on a worker thread managed by the concurrent wrapper. We use
            // a queue of length one so the memory used by the queue is only
            // one slice.

            CConcurrentWrapper<TReadFunc, 1, 1> asyncReader{m_Reader};
            for (auto i = beginSlices; i != endSlices; ++i) {
                std::tie(firstRow, sliceBeingRead) = m_ReadSliceFromStore(*i);
                if (sliceBeingRead.bad()) {
                    return false;
                }
                LOG_TRACE(<< "reading slice starting at row " << firstRow);
                asyncReader(
                    [ firstRow, slice = std::move(sliceBeingRead), cols ](TReadFuncRef reader) {
                        std::size_t rows = slice.size() / cols;
                        reader(CRowConstIterator{cols, firstRow, slice.begin()},
                               CRowConstIterator{cols, firstRow + rows, slice.end()});
                    });
            }
            break;
        }
        case CDataFrame::EReadWriteToStorage::E_Sync:
            for (auto i = beginSlices; i != endSlices; ++i) {
                std::tie(firstRow, sliceBeingRead) = m_ReadSliceFromStore(*i);
                if (sliceBeingRead.bad()) {
                    return false;
                }
                LOG_TRACE(<< "reading slice starting at row " << firstRow);
                std::size_t rows = sliceBeingRead.size() / cols;
                m_Reader(CRowConstIterator{cols, firstRow, sliceBeingRead.begin()},
                         CRowConstIterator{cols, firstRow + rows, sliceBeingRead.end()});
            }
            break;
        }

        return true;
    }

private:
    TReadFuncRef m_Reader;
    std::size_t m_NumberColumns;
    CDataFrame::EReadWriteToStorage m_AsyncReadFromStore;
    TReadSliceFromStoreFunc m_ReadSliceFromStore;
};

//! Compute the default slice capacity in rows.
std::size_t computeSliceCapacity(std::size_t numberColumns) {
    return std::max(1000000 / sizeof(CFloatStorage) / numberColumns, std::size_t(100));
}
}

CDataFrame::CDataFrame(std::size_t numberColumns,
                       std::size_t sliceCapacity,
                       EReadWriteToStorage asyncReadAndWriteToStore,
                       const TWriteSliceToStoreFunc& writeSliceToStore,
                       const TReadSliceFromStoreFunc& readSliceFromStore)
    : m_NumberColumns{numberColumns}, m_SliceCapacity{sliceCapacity},
      m_AsyncReadAndWriteToStore{asyncReadAndWriteToStore},
      m_WriteSliceToStore{writeSliceToStore}, m_ReadSliceFromStore{readSliceFromStore} {
}

CDataFrame::CDataFrame(std::size_t numberColumns,
                       EReadWriteToStorage asyncReadAndWriteToStore,
                       const TWriteSliceToStoreFunc& writeSliceToStore,
                       const TReadSliceFromStoreFunc& readSliceFromStore)
    : CDataFrame{numberColumns, computeSliceCapacity(numberColumns),
                 asyncReadAndWriteToStore, writeSliceToStore, readSliceFromStore} {
}

bool CDataFrame::reserve(std::size_t numberThreads, std::size_t numberColumns) {
    if (m_NumberColumns < numberColumns) {
        std::size_t stride;
        std::tie(numberThreads, stride) = this->numberOfThreadsAndStride(numberThreads);

        // TODO
    }
    return true;
}

CDataFrame::TReadFuncVecBoolPr CDataFrame::readRows(std::size_t numberThreads,
                                                    TReadFunc reader) const {
    if (m_NumberRows == 0) {
        return {{reader}, true};
    }

    if (numberThreads == 1) {
        // This all happens on the main thread to avoid a context switch.

        CDataFrameRowSliceReader sliceReader{
            reader, m_NumberColumns, m_AsyncReadAndWriteToStore, m_ReadSliceFromStore};
        bool successful{sliceReader(m_Slices.begin(), m_Slices.end())};
        return {{std::move(reader)}, successful};
    }

    // We use a fixed schedule whereby each reader reads non-overlapping
    // slices. This means we can get no contention on reads from the slice
    // vector. This is naturally load balanced because we arrange for each
    // reader to read, as close as possible, the same number of rows.

    std::size_t stride;
    std::tie(numberThreads, stride) = this->numberOfThreadsAndStride(numberThreads);

    TReadFuncVec readers{numberThreads, reader};

    std::vector<std::future<bool>> reads;
    reads.reserve(numberThreads);
    std::size_t j{0};
    for (std::size_t i = 0; i + 1 < numberThreads; ++i, j += stride) {
        auto begin = m_Slices.begin() + j;
        auto end = m_Slices.begin() + j + stride;
        CDataFrameRowSliceReader sliceReader{readers[i], m_NumberColumns, m_AsyncReadAndWriteToStore,
                                             m_ReadSliceFromStore};
        reads.push_back(std::async(std::launch::async, sliceReader, begin, end));
    }
    auto begin = m_Slices.begin() + j;
    auto end = m_Slices.end();
    CDataFrameRowSliceReader sliceReader{readers.back(), m_NumberColumns,
                                         m_AsyncReadAndWriteToStore, m_ReadSliceFromStore};
    reads.push_back(std::async(std::launch::async, sliceReader, begin, end));

    bool successful{true};
    for (auto& read : reads) {
        successful &= read.get();
    }

    return {std::move(readers), successful};
}

void CDataFrame::writeRow(const TWriteFunc& writeRow) {
    if (m_Writer == nullptr) {
        m_Writer = boost::make_unique<CDataFrameRowSliceWriter>(
            m_NumberRows, m_NumberColumns, m_SliceCapacity,
            m_AsyncReadAndWriteToStore, m_WriteSliceToStore);
    }
    (*m_Writer)(writeRow);
}

void CDataFrame::finishWritingRows() {
    // Get any slices which have been written, append and clear the writer.

    if (m_Writer != nullptr) {
        TRowSlicePtrVec slices;
        std::tie(m_NumberRows, slices) = m_Writer->finishWritingRows();
        m_Writer.reset();

        m_Slices.reserve(m_Slices.size() + slices.size());
        for (auto& slice : slices) {
            m_Slices.push_back(std::move(slice));
        }
        LOG_TRACE(<< "# slices = " << m_Slices.size());
    }
}

std::size_t CDataFrame::memoryUsage() const {
    return CMemory::dynamicSize(m_Slices) + CMemory::dynamicSize(m_Writer);
}

CDataFrame::CDataFrameRowSliceWriter::CDataFrameRowSliceWriter(std::size_t numberRows,
                                                               std::size_t numberColumns,
                                                               std::size_t sliceCapacity,
                                                               EReadWriteToStorage asyncWriteToStore,
                                                               TWriteSliceToStoreFunc writeSliceToStore)
    : m_NumberRows{numberRows}, m_NumberColumns{numberColumns}, m_SliceCapacity{sliceCapacity},
      m_AsyncWriteToStore{asyncWriteToStore}, m_WriteSliceToStore{writeSliceToStore} {
    m_SliceBeingWritten.reserve(m_SliceCapacity * m_NumberColumns);
    if (m_AsyncWriteToStore == EReadWriteToStorage::E_Async) {
        m_AsyncWriteToStoreResult =
            std::async(std::launch::async,
                       CAsyncDataFrameStorageWriter{sliceCapacity, writeSliceToStore},
                       std::ref(m_SlicesToAsyncWriteToStore));
    }
}

CDataFrame::CDataFrameRowSliceWriter::~CDataFrameRowSliceWriter() {
    this->finishAsyncWriteToStore();
}

void CDataFrame::CDataFrameRowSliceWriter::operator()(const TWriteFunc& writeRow) {
    // Write the next row at the end of the current slice being written
    // and if the slice is full pass to the thread storing slices.

    std::size_t end{m_SliceBeingWritten.size()};

    m_SliceBeingWritten.resize(end + m_NumberColumns);
    writeRow(m_SliceBeingWritten.begin() + end);
    ++m_NumberRows;

    if (m_SliceBeingWritten.size() == m_SliceCapacity * m_NumberColumns) {
        std::size_t firstRow{m_NumberRows - m_SliceCapacity};
        LOG_TRACE(<< "Storing slice [" << firstRow << "," << m_NumberRows << ")");

        switch (m_AsyncWriteToStore) {
        case EReadWriteToStorage::E_Async: {
            TSizeFloatVecPr slice{firstRow, std::move(m_SliceBeingWritten)};
            m_SlicesToAsyncWriteToStore.push(std::move(slice));
            break;
        }
        case EReadWriteToStorage::E_Sync:
            m_SyncWrittenSlices.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_SliceBeingWritten)));
            break;
        }
        m_SliceBeingWritten.clear();
        m_SliceBeingWritten.reserve(m_SliceCapacity * m_NumberColumns);
    }
}

CDataFrame::TSizeDataFrameRowSlicePtrVecPr
CDataFrame::CDataFrameRowSliceWriter::finishWritingRows() {
    // Passing a partial slice signals to the thread writing slices to
    // storage that we're done.

    std::size_t firstRow{m_NumberRows - m_SliceBeingWritten.size() / m_NumberColumns};
    LOG_TRACE(<< "Last slice "
              << (firstRow == m_NumberRows ? "empty"
                                           : "[" + std::to_string(firstRow) + "," +
                                                 std::to_string(m_NumberRows) + ")"));

    switch (m_AsyncWriteToStore) {
    case EReadWriteToStorage::E_Async:
        if (m_SliceBeingWritten.size() > 0) {
            TSizeFloatVecPr slice{firstRow, std::move(m_SliceBeingWritten)};
            m_SlicesToAsyncWriteToStore.push(std::move(slice));
        }
        this->finishAsyncWriteToStore();
        return {m_NumberRows, m_AsyncWriteToStoreResult.get()};
    case EReadWriteToStorage::E_Sync:
        if (m_SliceBeingWritten.size() > 0) {
            m_SyncWrittenSlices.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_SliceBeingWritten)));
        }
        break;
    }
    return {m_NumberRows, std::move(m_SyncWrittenSlices)};
}

CDataFrame::TSizeSizePr CDataFrame::numberOfThreadsAndStride(std::size_t target) const {

    std::size_t numberSlices{m_Slices.size()};
    std::size_t numberThreads{std::min(numberSlices, target)};
    std::size_t strideLowerBound{numberSlices / numberThreads};
    std::size_t strideUpperBound{strideLowerBound + 1};

    if (numberSlices % numberThreads == 0) {
        return {numberThreads, strideLowerBound};
    }
    if (numberSlices % strideUpperBound == 0) {
        return {numberThreads - 1, strideUpperBound};
    }
    return {numberThreads, strideUpperBound};
}

void CDataFrame::CDataFrameRowSliceWriter::finishAsyncWriteToStore() {
    // Signal to the thread writing slices to storage that we're done.

    if (m_Writing) {
        if (m_AsyncWriteToStore == EReadWriteToStorage::E_Async) {
            m_SlicesToAsyncWriteToStore.push({0, TFloatVec{}});
        }
        m_Writing = false;
    }
}

CDataFrame makeMainStorageDataFrame(std::size_t numberColumns,
                                    boost::optional<std::size_t> sliceCapacity,
                                    CDataFrame::EReadWriteToStorage readWriteToStoreAsync) {

    // The return copy is elided so we never need to call the explicitly
    // deleted the data frame copy constructor.

    auto writer = [](std::size_t firstRow, TFloatVec slice) {
        return boost::make_unique<CMainMemoryDataFrameRowSlice>(firstRow, std::move(slice));
    };
    auto reader = [](const TRowSlicePtr& slice) { return slice->read(); };

    if (sliceCapacity != boost::none) {
        return {numberColumns, *sliceCapacity, readWriteToStoreAsync, writer, reader};
    }

    return {numberColumns, readWriteToStoreAsync, writer, reader};
}

CDataFrame makeDiskStorageDataFrame(const std::string& rootDirectory,
                                    std::size_t numberColumns,
                                    std::size_t numberRows,
                                    boost::optional<std::size_t> sliceCapacity,
                                    CDataFrame::EReadWriteToStorage readWriteToStoreAsync) {
    // The return copy is elided so we never need to call the explicitly
    // deleted the data frame copy constructor.

    std::size_t minimumSpace{2 * numberRows * numberColumns * sizeof(CFloatStorage)};

    COnDiskDataFrameRowSlice::TTemporaryDirectoryPtr directory{
        std::make_shared<COnDiskDataFrameRowSlice::CTemporaryDirectory>(
            rootDirectory, minimumSpace)};

    // Note the writer lambda holds a reference to the directory shared
    // pointer is copied to the data frame. So this isn't destroyed, and
    // the folder cleaned up, until the data frame itself is destroyed.

    auto writer = [directory](std::size_t firstRow, TFloatVec slice) {
        return boost::make_unique<COnDiskDataFrameRowSlice>(directory, firstRow,
                                                            std::move(slice));
    };
    auto reader = [](const TRowSlicePtr& slice) { return slice->read(); };

    if (sliceCapacity != boost::none) {
        return {numberColumns, *sliceCapacity, readWriteToStoreAsync, writer, reader};
    }
    return {numberColumns, readWriteToStoreAsync, writer, reader};
}
}
}
