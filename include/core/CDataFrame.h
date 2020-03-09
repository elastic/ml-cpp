/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CDataFrame_h
#define INCLUDED_ml_core_CDataFrame_h

#include <core/CFloatStorage.h>
#include <core/CPackedBitVector.h>
#include <core/CVectorRange.h>
#include <core/Concurrency.h>
#include <core/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

namespace ml {
namespace core {
class CDataFrameRowSlice;
class CDataFrameRowSliceHandle;
class CTemporaryDirectory;

namespace data_frame_detail {

using TFloatVec = std::vector<CFloatStorage>;
using TFloatVecItr = TFloatVec::iterator;
using TInt32Vec = std::vector<std::int32_t>;
using TInt32VecCItr = TInt32Vec::const_iterator;

//! \brief A callback used to iterate over only the masked rows.
class CORE_EXPORT CPopMaskedRow {
public:
    CPopMaskedRow(std::size_t endSliceRows,
                  CPackedBitVector::COneBitIndexConstIterator& maskedRow,
                  const CPackedBitVector::COneBitIndexConstIterator& endMaskedRows)
        : m_EndSliceRows{endSliceRows}, m_MaskedRow{&maskedRow}, m_EndMaskedRows{&endMaskedRows} {
    }

    std::size_t operator()() const {
        return ++(*m_MaskedRow) == *m_EndMaskedRows
                   ? m_EndSliceRows
                   : std::min(**m_MaskedRow, m_EndSliceRows);
    }

private:
    std::size_t m_EndSliceRows;
    CPackedBitVector::COneBitIndexConstIterator* m_MaskedRow;
    const CPackedBitVector::COneBitIndexConstIterator* m_EndMaskedRows;
};

using TOptionalPopMaskedRow = boost::optional<CPopMaskedRow>;

//! \brief A lightweight wrapper around a single row of the data frame.
//!
//! DESCRIPTION:\n
//! This is a helper class used to read rows of a CDataFrame object. It is
//! lightweight (24 bytes) and is expected to only be valid transiently
//! during a read. It should not be stored.
//!
//! If the row resides in main memory then its data can be referenced. If
//! it resides on disk then this is only valid whilst being read and it
//! should be copied if needed longer.
class CORE_EXPORT CRowRef {
public:
    //! \param[in] index The row index.
    //! \param[in] beginColumns The iterator for the columns of row \p index.
    //! \param[in] endColumns The iterator for the end of the columns of row
    //! \p index.
    //! \param[in] docHash The row's hash.
    CRowRef(std::size_t index, TFloatVecItr beginColumns, TFloatVecItr endColumns, std::int32_t docHash);

    //! Get column \p i value.
    CFloatStorage operator[](std::size_t i) const;

    //! Get the row's index.
    std::size_t index() const;

    //! Get the number of columns.
    std::size_t numberColumns() const;

    //! Write \p value to \p column of the row.
    void writeColumn(std::size_t index, double value) const;

    //! Get the data backing the row.
    CFloatStorage* data() const;

    //! Copy the range to \p output iterator.
    //!
    //! \warning The output iterator that must be able to receive number of
    //! columns values.
    template<typename ITR>
    void copyTo(ITR output) const {
        std::copy(m_BeginColumns, m_EndColumns, output);
    }

    //! Get the row's hash.
    std::int32_t docHash() const;

private:
    std::size_t m_Index;
    TFloatVecItr m_BeginColumns;
    TFloatVecItr m_EndColumns;
    std::int32_t m_DocHash;
};

//! \brief Decorates CRowCRef to give it pointer semantics.
class CORE_EXPORT CRowPtr final : public CRowRef {
public:
    template<typename... ARGS>
    CRowPtr(ARGS&&... args) : CRowRef{std::forward<ARGS>(args)...} {}

    CRowPtr* operator->() { return this; }
    const CRowPtr* operator->() const { return this; }
};

//! \brief A forward iterator over rows of the data frame.
//!
//! DESCRIPTION:\n
//! This is a helper class used to read rows of a CDataFrame object which
//! dereferences to CRowRef objects.
class CORE_EXPORT CRowIterator final
    : public std::iterator<std::forward_iterator_tag, CRowRef, std::ptrdiff_t, CRowPtr, CRowRef> {
public:
    CRowIterator() = default;

    //! \param[in] numberColumns The number of columns in the data frame.
    //! \param[in] rowCapacity The capacity of each row in the data frame.
    //! \param[in] index The row index.
    //! \param[in] rowItr The iterator for the columns of the rows starting
    //! at \p index.
    //! \param[in] docHashItr The iterator for the document hashes of rows
    //! starting at \p index.
    //! \param[in] popMaskedRow Gets the next row in the mask.
    CRowIterator(std::size_t numberColumns,
                 std::size_t rowCapacity,
                 std::size_t index,
                 TFloatVecItr rowItr,
                 TInt32VecCItr docHashItr,
                 const TOptionalPopMaskedRow& popMaskedRow);

    //! \name Forward Iterator Contract
    //@{
    bool operator==(const CRowIterator& rhs) const;
    bool operator!=(const CRowIterator& rhs) const;
    CRowRef operator*() const;
    CRowPtr operator->() const;
    CRowIterator& operator++();
    CRowIterator operator++(int);
    //@}

private:
    std::size_t m_NumberColumns = 0;
    std::size_t m_RowCapacity = 0;
    std::size_t m_Index = 0;
    TFloatVecItr m_RowItr;
    TInt32VecCItr m_DocHashItr;
    TOptionalPopMaskedRow m_PopMaskedRow;
};
}

//! \brief A data frame representation.
//!
//! DESECRIPTION:\n
//! A table data structure with "fixed" number of columns and a dynamic number
//! of rows.
//!
//! It can be read in row order and is append only. Reading rows can also be
//! parallelized in which case each reader reads a disjoint subset of the data
//! frame's rows.
//!
//! Space can be reserved at any point to hold one or more additional columns.
//! These are not visible until they are written.
//!
//! IMPLEMENTATION:\n
//! This is a fairly lightweight container which is essentially responsible
//! for managing the read and write process to some underlying store format.
//! The store format is determined by the user implementing functionality to
//! read and write state from the store. For example, these could copy to /
//! from main memory, "write to" / "read from" disk, etc. A factory function
//! must be provided to the constructor which effectively that determines the
//! type of storage used. It is assumed that copying this has no side effects.
//!
//! The data frame is divided into slices each of which represent a number of
//! contiguous rows. The idea is that they contain a reasonable amount of memory
//! so that, for example, they significantly reduce the number of "writes to" /
//! "reads from" disk (a whole slice being written or read in one go), mean we'll
//! get good locality of reference and mean there is minimal book keeping overhead
//! (such as state for vector sizes, pointers to starts of memory blocks, etc).
//! In addition, it is assumed that access to the individual slices is thread
//! safe. If they share state the implementation must ensure that access to this
//! is synchronized.
//!
//! Reads and writes of a single row are also done via call backs supplied to the
//! readRows and writeRow functions. This is to achieve maximum decoupling from
//! the calling code for how the underlying values are used or where they come
//! from. It also means certain operations can be done very efficiently. For example,
//! a stream can be attached to a row writer function to copy the values directly
//! into the data frame storage.
//!
//! Read and writes to storage can optionally happen in a separate thread to the
//! row reading and writing to deal with the case that these operations can by
//! time consuming.
class CORE_EXPORT CDataFrame final {
public:
    using TBoolVec = std::vector<bool>;
    using TStrVec = std::vector<std::string>;
    using TStrVecVec = std::vector<TStrVec>;
    using TStrCRng = CVectorRange<const TStrVec>;
    using TFloatVec = std::vector<CFloatStorage>;
    using TFloatVecItr = TFloatVec::iterator;
    using TInt32Vec = std::vector<std::int32_t>;
    using TRowRef = data_frame_detail::CRowRef;
    using TRowItr = data_frame_detail::CRowIterator;
    using TRowFunc = std::function<void(TRowItr, TRowItr)>;
    using TRowFuncVec = std::vector<TRowFunc>;
    using TRowFuncVecBoolPr = std::pair<TRowFuncVec, bool>;
    using TWriteFunc = std::function<void(TFloatVecItr, std::int32_t&)>;
    using TRowSlicePtr = std::shared_ptr<CDataFrameRowSlice>;
    using TRowSlicePtrVec = std::vector<TRowSlicePtr>;
    using TRowSlicePtrVecCItr = TRowSlicePtrVec::const_iterator;
    using TSizeRowSliceHandlePr = std::pair<std::size_t, CDataFrameRowSliceHandle>;
    using TWriteSliceToStoreFunc =
        std::function<TRowSlicePtr(std::size_t, TFloatVec, TInt32Vec)>;

    //! Controls whether to read and write to storage asynchronously.
    enum class EReadWriteToStorage { E_Async, E_Sync };

public:
    //! The maximum number of distinct categorical fields we can faithfully represent.
    static const std::size_t MAX_CATEGORICAL_CARDINALITY;

    //! The default value indicating that a value is missing.
    static const std::string DEFAULT_MISSING_STRING;

public:
    //! \param[in] inMainMemory True if the data frame is stored in main memory.
    //! \param[in] numberColumns The number of columns in the data frame.
    //! \param[in] sliceCapacityInRows The capacity of a slice of the data frame
    //! as a number of rows.
    //! \param[in] readAndWriteToStoreSyncStrategy Controls whether reads and
    //! writes from slice storage are synchronous or asynchronous.
    //! \param[in] writeSliceToStore The callback to write a slice to storage.
    //!
    //! \warning This requires that \p writeSliceToStore and \p readSliceFromStore
    //! can be copied and are thread safe. If they are not stateless then it is
    //! the implementers responsibility to ensure these conditions are satisfied.
    CDataFrame(bool inMainMemory,
               std::size_t numberColumns,
               std::size_t sliceCapacityInRows,
               EReadWriteToStorage readAndWriteToStoreSyncStrategy,
               const TWriteSliceToStoreFunc& writeSliceToStore);

    //! Overload which manages the setting of slice capacity to a sensible default.
    CDataFrame(bool inMainMemory,
               std::size_t numberColumns,
               EReadWriteToStorage readAndWriteToStoreSyncStrategy,
               const TWriteSliceToStoreFunc& writeSliceToStore);

    ~CDataFrame();

    CDataFrame(const CDataFrame&) = delete;
    CDataFrame& operator=(const CDataFrame&) = delete;
    CDataFrame(CDataFrame&&) = default;
    CDataFrame& operator=(CDataFrame&&) = default;

    //! Check if the data frame resides in main memory.
    bool inMainMemory() const;

    //! Get the number of rows in the data frame.
    std::size_t numberRows() const;

    //! Get the number of columns in the data frame.
    std::size_t numberColumns() const;

    //! Reserve space for up to \p numberColumns.
    //!
    //! This enables in-place updates of the data frame for analytics operations
    //! that append columns.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] rowCapacity The desired number of columns.
    void reserve(std::size_t numberThreads, std::size_t rowCapacity);

    //! Resize to contain \p numberColumns columns.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] numberColumns The desired number of columns.
    void resizeColumns(std::size_t numberThreads, std::size_t numberColumns);

    //! This reads rows using one or more readers.
    //!
    //! One reader is bound to one thread. Each thread reads a disjoint subset
    //! of slices of the data frame with the i'th reader reading slices
    //! \f$[ i \lfloor m / \min(m,n) \rfloor, (i+1) \lfloor m / \min(m,n) \rfloor)\f$
    //! for \f$m\f$ slices and \f$n\f$ readers to eliminate contention on the
    //! data frame state.
    //!
    //! \warning If there is more than one thread and the reader has shared
    //! state then the caller must ensure that access to this is thread safe.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] beginRows The row at which to start reading.
    //! \param[in] endRows The row (exclusive) at which to stop reading.
    //! \param[in] reader The callback to read rows.
    //! \param[in] rowMask If supplied only the rows corresponding to the one
    //! bits of this vector are read.
    //! \return The readers used. This is intended to allow the reader to
    //! accumulate state and pass it back. RVO means the copy on return will
    //! be elided. If the reader holds state by reference it must synchronize
    //! access to it.
    TRowFuncVecBoolPr readRows(std::size_t numberThreads,
                               std::size_t beginRows,
                               std::size_t endRows,
                               TRowFunc reader,
                               const CPackedBitVector* rowMask = nullptr) const;

    //! Convenience overload which reads all rows.
    TRowFuncVecBoolPr readRows(std::size_t numberThreads, TRowFunc reader) const {
        return this->readRows(numberThreads, 0, this->numberRows(), std::move(reader));
    }

    //! Convenience overload for typed readers.
    //!
    //! The reason for this is to wrap up the code to extract the typed readers
    //! from the return type. A common case is that these will cache some state
    //! for which the calling code will need access to the underlying type to
    //! retrieve.
    //!
    //! \note READER must implement the TRowFunc contract.
    template<typename READER>
    std::pair<std::vector<READER>, bool>
    readRows(std::size_t numberThreads,
             std::size_t beginRows,
             std::size_t endRows,
             READER reader,
             const CPackedBitVector* rowMask = nullptr) const {

        TRowFuncVecBoolPr result{this->readRows(numberThreads, beginRows, endRows,
                                                TRowFunc(std::move(reader)), rowMask)};

        std::vector<READER> readers;
        readers.reserve(result.first.size());
        for (auto& reader_ : result.first) {
            readers.push_back(std::move(*reader_.target<READER>()));
        }

        return {std::move(readers), result.second};
    }

    //! Convenience overload for typed reading of all rows.
    template<typename READER>
    std::pair<std::vector<READER>, bool>
    readRows(std::size_t numberThreads, READER reader) const {
        return this->readRows(numberThreads, 0, this->numberRows(), std::move(reader));
    }

    //! Overwrite a number of columns with \p writer.
    //!
    //! The caller must ensure that the columns overwritten are in range.
    //!
    //! \warning If there is more than one thread and the writer has shared
    //! state then the caller must ensure that access to this is thread safe.
    //!
    //! \param[in] numberThreads The target number of threads to use.
    //! \param[in] beginRows The row at which to start writing.
    //! \param[in] endRows The row (exclusive) at which to stop writing.
    //! \param[in] writer The callback to write the columns.
    //! \param[in] rowMask If supplied only the rows corresponding to the one
    //! bits of this vector are written.
    //! \return The writers used. This is intended to allow the writer to
    //! accumulate state and pass it back. RVO means the copy on return will
    //! be elided. If the writer holds state by reference it must synchronize
    //! access to it.
    TRowFuncVecBoolPr writeColumns(std::size_t numberThreads,
                                   std::size_t beginRows,
                                   std::size_t endRows,
                                   TRowFunc writer,
                                   const CPackedBitVector* rowMask = nullptr);

    //! Convenience overload which writes all rows.
    TRowFuncVecBoolPr writeColumns(std::size_t numberThreads, TRowFunc reader) {
        return this->writeColumns(numberThreads, 0, this->numberRows(), std::move(reader));
    }

    //! Convenience overload for typed writers.
    //!
    //! The reason for this is to wrap up the code to extract the typed writers
    //! from the return type.
    //!
    //! \note WRITER must implement the TRowFunc contract.
    template<typename WRITER>
    std::pair<std::vector<WRITER>, bool>
    writeColumns(std::size_t numberThreads,
                 std::size_t beginRows,
                 std::size_t endRows,
                 WRITER writer,
                 const CPackedBitVector* rowMask = nullptr) {

        TRowFuncVecBoolPr result{this->writeColumns(
            numberThreads, beginRows, endRows, TRowFunc(std::move(writer)), rowMask)};

        std::vector<WRITER> writers;
        writers.reserve(result.first.size());
        for (auto& writer_ : result.first) {
            writers.push_back(std::move(*writer_.target<WRITER>()));
        }

        return {std::move(writers), result.second};
    }

    //! Convenience overload for typed reading of all rows.
    template<typename WRITER>
    std::pair<std::vector<WRITER>, bool>
    writeColumns(std::size_t numberThreads, WRITER writer) {
        return this->writeColumns(numberThreads, 0, this->numberRows(), std::move(writer));
    }

    //! Parses the strings in \p columnValues and writes one row via writeRow.
    void parseAndWriteRow(const TStrCRng& columnValues, const std::string* hash = nullptr);

    //! This writes a single row of the data frame via a callback.
    //!
    //! If asynchronous read and write to store was selected in the constructor
    //! this creates a thread to store slices which is only joined when the
    //! finishWritingRows is called (or the data frame is destroyed).
    //!
    //! Until finishWritingRows is called the written row is not visible outside
    //! the data frame.
    //!
    //! \param[in] writeRow A function which writes a single row setting the
    //! values in supplied an output iterator to the column values of the row.
    //!
    //! \warning The caller MUST call finishWritingRows after they have finished
    //! writing rows.
    void writeRow(const TWriteFunc& writeRow);

    //! Write the column names.
    void columnNames(TStrVec columnNames);

    //! Write the string which indicates that a value is missing.
    void missingString(std::string missing);

    //! Write which columns contain categorical data.
    void categoricalColumns(TStrVec categoricalColumnNames);

    //! Write which columns contain categorical data.
    void categoricalColumns(TBoolVec columnIsCategorical);

    //! This retrieves the asynchronous work from writing the rows to the store
    //! and updates the stored rows.
    //!
    //! Until this is called the written rows are not visible outside the data
    //! frame.
    //!
    //! \warning This MUST be called after the last row is written to commit the
    //! work and to join the thread used to store the slices.
    void finishWritingRows();

    //! \return The column names if any.
    const TStrVec& columnNames() const;

    //! \return The string values of the categories for each column.
    const TStrVecVec& categoricalColumnValues() const;

    //! \return Indicator of columns containing categorical data.
    const TBoolVec& columnIsCategorical() const;

    //! Get the memory used by the data frame.
    std::size_t memoryUsage() const;

    //! Get a checksum of all the data stored in the data frame.
    std::uint64_t checksum() const;

    //! Get the estimated memory usage for a data frame with \p numberRows rows and
    //! \p numberColumns columns.
    static std::size_t estimateMemoryUsage(bool inMainMemory,
                                           std::size_t numberRows,
                                           std::size_t numberColumns);

    //! Get the value to use for a missing element in a data frame.
    static constexpr double valueOfMissing() {
        return std::numeric_limits<double>::quiet_NaN();
    }

private:
    using TStrSizeUMap = boost::unordered_map<std::string, std::size_t>;
    using TStrSizeUMapVec = std::vector<TStrSizeUMap>;
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeDataFrameRowSlicePtrVecPr = std::pair<std::size_t, TRowSlicePtrVec>;
    using TOptionalPopMaskedRow = data_frame_detail::TOptionalPopMaskedRow;

    //! \brief Writes rows to the data frame.
    class CDataFrameRowSliceWriter final {
    public:
        CDataFrameRowSliceWriter(std::size_t numberRows,
                                 std::size_t rowCapacity,
                                 std::size_t sliceCapacityInRows,
                                 EReadWriteToStorage writeToStoreSyncStrategy,
                                 TWriteSliceToStoreFunc writeSliceToStore);

        //! Write a single row using the callback \p writeRow.
        void operator()(const TWriteFunc& writeRow);

        //! Finish writing the rows and return the number of rows written and
        //! the slices.
        TSizeDataFrameRowSlicePtrVecPr finishWritingRows();

    private:
        std::size_t m_NumberRows;
        std::size_t m_RowCapacity;
        std::size_t m_SliceCapacityInRows;
        EReadWriteToStorage m_WriteToStoreSyncStrategy;
        TWriteSliceToStoreFunc m_WriteSliceToStore;
        TFloatVec m_RowsOfSliceBeingWritten;
        TInt32Vec m_DocHashesOfSliceBeingWritten;
        std::future<TRowSlicePtr> m_SliceWrittenAsyncToStore;
        TRowSlicePtrVec m_SlicesWrittenToStore;
    };
    using TRowSliceWriterPtr = std::unique_ptr<CDataFrameRowSliceWriter>;

private:
    TRowFuncVecBoolPr parallelApplyToAllRows(std::size_t numberThreads,
                                             std::size_t beginRows,
                                             std::size_t endRows,
                                             TRowFunc&& func,
                                             const CPackedBitVector* rowMask,
                                             bool commitResult) const;
    TRowFuncVecBoolPr sequentialApplyToAllRows(std::size_t beginRows,
                                               std::size_t endRows,
                                               TRowFunc& func,
                                               const CPackedBitVector* rowMask,
                                               bool commitResult) const;

    void applyToRowsOfOneSlice(TRowFunc& func,
                               std::size_t firstRowToRead,
                               std::size_t endRowsToRead,
                               const TOptionalPopMaskedRow& popMaskedRow,
                               const CDataFrameRowSliceHandle& slice) const;

    TRowSlicePtrVecCItr beginSlices(std::size_t beginRows) const;
    TRowSlicePtrVecCItr endSlices(std::size_t endRows) const;

    template<typename ITR>
    bool maskedRowsInSlice(ITR& maskedRow,
                           ITR endMaskedRows,
                           std::size_t beginSliceRows,
                           std::size_t endSliceRows) const;

private:
    //! True if the data frame resides in main memory.
    bool m_InMainMemory;
    //! The number of rows in the data frame.
    std::size_t m_NumberRows = 0;
    //! The number of columns in the data frame.
    std::size_t m_NumberColumns;
    //! The number of columns a row could contain. This is greater than or
    //! equal to m_NumberColumns.
    std::size_t m_RowCapacity;
    //! The capacity of a slice of the data frame as a number of rows.
    std::size_t m_SliceCapacityInRows;

    //! If true read and write asynchronously to storage.
    EReadWriteToStorage m_ReadAndWriteToStoreSyncStrategy;
    //! The callback to write a slice to storage.
    TWriteSliceToStoreFunc m_WriteSliceToStore;

    //! Optional column names.
    TStrVec m_ColumnNames;

    //! The string values of the categories.
    TStrVecVec m_CategoricalColumnValues;

    //! A lookup for the integer value of categories.
    TStrSizeUMapVec m_CategoricalColumnValueLookup;

    //! The string which indicates that a category is missing.
    std::string m_MissingString;

    //! Indicator vector of the columns which contain categorical values.
    TBoolVec m_ColumnIsCategorical;

    //! \name Parse Counters
    //@{
    std::uint64_t m_MissingValueCount = 0;
    std::uint64_t m_BadValueCount = 0;
    std::uint64_t m_BadDocHashCount = 0;
    //@}

    //! The stored slices.
    TRowSlicePtrVec m_Slices;

    //! The slice writer which is currently active.
    TRowSliceWriterPtr m_Writer;
};

//! Make a data frame which uses main memory storage for its slices.
//!
//! \param[in] numberColumns The number of columns in the data frame created.
//! \param[in] sliceCapacity If none null this overrides the default slice
//! capacity in rows.
//! \param[in] readWriteToStoreSyncStrategy Controls whether reads and writes
//! from slice storage are synchronous or asynchronous.
CORE_EXPORT
std::pair<std::unique_ptr<CDataFrame>, std::shared_ptr<CTemporaryDirectory>>
makeMainStorageDataFrame(std::size_t numberColumns,
                         boost::optional<std::size_t> sliceCapacity = boost::none,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy =
                             CDataFrame::EReadWriteToStorage::E_Sync);

//! Make a data frame which uses disk storage for its slices.
//!
//! \param[in] rootDirectory The name of the directory to which write the
//! data frame slices.
//! \param[in] numberColumns The number of columns in the data frame created.
//! \param[in] numberRows The number of rows that will be added.
//! \param[in] sliceCapacity If none null this overrides the default slice
//! capacity in rows.
//! \param[in] readWriteToStoreSyncStrategy Controls whether reads and writes
//! from slice storage are synchronous or asynchronous.
CORE_EXPORT
std::pair<std::unique_ptr<CDataFrame>, std::shared_ptr<CTemporaryDirectory>>
makeDiskStorageDataFrame(const std::string& rootDirectory,
                         std::size_t numberColumns,
                         std::size_t numberRows,
                         boost::optional<std::size_t> sliceCapacity = boost::none,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy =
                             CDataFrame::EReadWriteToStorage::E_Async);
}
}

#endif // INCLUDED_ml_core_CDataFrame_h
