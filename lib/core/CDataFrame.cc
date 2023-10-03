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

#include <core/CDataFrame.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrameRowSlice.h>
#include <core/CHashing.h>
#include <core/CLogger.h>
#include <core/CMemoryDef.h>
#include <core/CPackedBitVector.h>
#include <core/CStringUtils.h>
#include <core/CVectorRange.h>
#include <core/Concurrency.h>
#include <core/Constants.h>

#include <algorithm>
#include <future>
#include <limits>
#include <memory>

namespace ml {
namespace core {
namespace {
core::CFloatStorage truncateToFloatRange(double value) {
    double largest{static_cast<double>(std::numeric_limits<float>::max())};
    return std::min(std::max(value, -largest), largest);
}
}

namespace data_frame_detail {

CRowRef::CRowRef(std::size_t index, TFloatVecItr beginColumns, TFloatVecItr endColumns, std::int32_t docHash)
    : m_Index{index}, m_BeginColumns{beginColumns}, m_EndColumns{endColumns}, m_DocHash{docHash} {
}

std::size_t CRowRef::index() const {
    return m_Index;
}

std::size_t CRowRef::numberColumns() const {
    return std::distance(m_BeginColumns, m_EndColumns);
}

void CRowRef::writeColumn(std::size_t column, double value) const {
    m_BeginColumns[column] = value;
}

CFloatStorage* CRowRef::data() const {
    return &(*m_BeginColumns);
}

std::int32_t CRowRef::docHash() const {
    return m_DocHash;
}

CRowIterator::CRowIterator(std::size_t numberColumns,
                           std::size_t rowCapacity,
                           std::size_t index,
                           TFloatVecItr rowItr,
                           TInt32VecCItr docHashItr,
                           const TOptionalPopMaskedRow& popMaskedRow)
    : m_NumberColumns{numberColumns}, m_RowCapacity{rowCapacity}, m_Index{index},
      m_RowItr{rowItr}, m_DocHashItr{docHashItr}, m_PopMaskedRow{popMaskedRow} {
}

bool CRowIterator::operator==(const CRowIterator& rhs) const {
    return m_RowItr == rhs.m_RowItr;
}

bool CRowIterator::operator!=(const CRowIterator& rhs) const {
    return m_RowItr != rhs.m_RowItr;
}

CRowRef CRowIterator::operator*() const {
    return CRowRef{m_Index, m_RowItr, m_RowItr + m_NumberColumns, *m_DocHashItr};
}

CRowPtr CRowIterator::operator->() const {
    return CRowPtr{m_Index, m_RowItr, m_RowItr + m_NumberColumns, *m_DocHashItr};
}

CRowIterator& CRowIterator::operator++() {
    if (m_PopMaskedRow != std::nullopt) {
        std::size_t nextIndex{(*m_PopMaskedRow)()};
        m_RowItr += m_RowCapacity * (nextIndex - m_Index);
        m_DocHashItr += nextIndex - m_Index;
        m_Index = nextIndex;
    } else {
        ++m_Index;
        m_RowItr += m_RowCapacity;
        ++m_DocHashItr;
    }
    return *this;
}

CRowIterator CRowIterator::operator++(int) {
    CRowIterator result{*this};
    this->operator++();
    return result;
}
}

using namespace data_frame_detail;

CDataFrame::CDataFrame(bool inMainMemory,
                       std::size_t numberColumns,
                       CAlignment::EType rowAlignment,
                       std::size_t sliceCapacityInRows,
                       EReadWriteToStorage readAndWriteToStoreSyncStrategy,
                       const TWriteSliceToStoreFunc& writeSliceToStore)
    : m_InMainMemory{inMainMemory}, m_NumberColumns{numberColumns},
      m_RowCapacity{CAlignment::roundup<CFloatStorage>(rowAlignment, numberColumns)},
      m_SliceCapacityInRows{sliceCapacityInRows}, m_RowAlignment{rowAlignment},
      m_ReadAndWriteToStoreSyncStrategy{readAndWriteToStoreSyncStrategy},
      m_WriteSliceToStore{writeSliceToStore}, m_ColumnNames(numberColumns),
      m_CategoricalColumnValues(numberColumns), m_MissingString{DEFAULT_MISSING_STRING},
      m_ColumnIsCategorical(numberColumns, false) {
}

CDataFrame::~CDataFrame() = default;

bool CDataFrame::inMainMemory() const {
    return m_InMainMemory;
}

std::size_t CDataFrame::numberRows() const {
    return m_NumberRows;
}

std::size_t CDataFrame::numberColumns() const {
    return m_NumberColumns;
}

void CDataFrame::reserve(std::size_t numberThreads, std::size_t rowCapacity) {

    rowCapacity = CAlignment::roundup<CFloatStorage>(m_RowAlignment, rowCapacity);

    if (m_RowCapacity >= rowCapacity) {
        return;
    }

    std::size_t oldRowCapacity{m_RowCapacity};
    m_RowCapacity = rowCapacity;

    parallel_for_each(numberThreads, m_Slices.begin(), m_Slices.end(),
                      [oldRowCapacity, this](TRowSlicePtr& slice) {
                          slice->reserve(oldRowCapacity, m_RowCapacity - oldRowCapacity);
                      });
}

void CDataFrame::resizeColumns(std::size_t numberThreads, std::size_t numberColumns) {
    this->reserve(numberThreads, numberColumns);
    m_ColumnNames.resize(numberColumns);
    m_CategoricalColumnValues.resize(numberColumns);
    m_ColumnIsCategorical.resize(numberColumns, false);
    m_NumberColumns = numberColumns;
}

CDataFrame::TSizeVecSizePr CDataFrame::resizeColumns(std::size_t numberThreads,
                                                     const TSizeAlignmentPrVec& extraColumns) {
    TSizeVec result;
    result.reserve(extraColumns.size());
    std::size_t index{m_NumberColumns};
    for (const auto& columns : extraColumns) {
        std::size_t count;
        CAlignment::EType alignment;
        std::tie(count, alignment) = columns;
        if (CAlignment::less(m_RowAlignment, alignment)) {
            HANDLE_FATAL(<< "Unsupported column alignment " << CAlignment::print(alignment));
        }
        index = CAlignment::roundup<CFloatStorage>(alignment, index);
        result.push_back(index);
        index += count;
    }
    std::size_t numberExtraColumns{index - m_NumberColumns};
    this->resizeColumns(numberThreads, index);
    return {result, numberExtraColumns};
}

void CDataFrame::resizeRows(std::size_t numberRows) {
    if (numberRows == m_NumberRows) {
        return;
    }

    if (numberRows > m_NumberRows) {
        // Add new rows if the size is being increased.
        for (std::size_t i = this->numberRows(); i < numberRows; ++i) {
            this->writeRow([this](TFloatVecItr columns, std::int32_t&) {
                for (std::size_t j = 0; j < m_NumberColumns; ++j, ++columns) {
                    *columns = 0.0;
                }
            });
        }
        this->finishWritingRows();
        return;
    }

    m_NumberRows = numberRows;

    // Find the last slice given the new size.
    auto lastSlice = m_Slices.begin();
    for (/**/; (*lastSlice)->indexOfLastRow(m_RowCapacity) + 1 < numberRows; ++lastSlice) {
    }

    // Remove extra rows if the number of rows is being reduced.
    m_Slices.erase(lastSlice + 1, m_Slices.end());
    if ((*lastSlice)->indexOfLastRow(m_RowCapacity) + 1 > m_NumberRows) {
        auto handle = (*lastSlice)->read();
        auto rows = handle.rows();
        auto docHashes = handle.docHashes();
        rows.resize(m_RowCapacity * (m_NumberRows - (*lastSlice)->indexOfFirstRow()));
        docHashes.resize((m_NumberRows - (*lastSlice)->indexOfFirstRow()));
        (*lastSlice)->write(rows, docHashes);
    }
}

CDataFrame::TRowFuncVecBoolPr CDataFrame::readRows(std::size_t numberThreads,
                                                   std::size_t beginRows,
                                                   std::size_t endRows,
                                                   TRowFunc reader,
                                                   const CPackedBitVector* rowMask) const {

    beginRows = std::min(beginRows, m_NumberRows);
    endRows = std::min(endRows, m_NumberRows);

    if (beginRows >= endRows) {
        return {{std::move(reader)}, true};
    }

    TRowFuncVec readers(numberThreads, std::move(reader));
    bool successful{
        numberThreads > 1
            ? this->parallelApplyToAllRows(beginRows, endRows, readers, rowMask, false)
            : this->sequentialApplyToAllRows(beginRows, endRows, readers, rowMask, false)};

    return {std::move(readers), successful};
}

bool CDataFrame::readRows(std::size_t beginRows,
                          std::size_t endRows,
                          TRowFuncVec& readers,
                          const CPackedBitVector* rowMask) const {

    beginRows = std::min(beginRows, m_NumberRows);
    endRows = std::min(endRows, m_NumberRows);

    if (beginRows >= endRows) {
        return true;
    }

    return readers.size() > 1
               ? this->parallelApplyToAllRows(beginRows, endRows, readers, rowMask, false)
               : this->sequentialApplyToAllRows(beginRows, endRows, readers, rowMask, false);
}

CDataFrame::TRowFuncVecBoolPr CDataFrame::writeColumns(std::size_t numberThreads,
                                                       std::size_t beginRows,
                                                       std::size_t endRows,
                                                       TRowFunc writer,
                                                       const CPackedBitVector* rowMask) {

    beginRows = std::min(beginRows, m_NumberRows);
    endRows = std::min(endRows, m_NumberRows);

    if (beginRows >= endRows) {
        return {{std::move(writer)}, true};
    }

    TRowFuncVec writers(numberThreads, writer);
    bool successful{
        numberThreads > 1
            ? this->parallelApplyToAllRows(beginRows, endRows, writers, rowMask, true)
            : this->sequentialApplyToAllRows(beginRows, endRows, writers, rowMask, true)};
    return {std::move(writers), successful};
}

void CDataFrame::parseAndWriteRow(const TStrCRng& columnValues,
                                  const TPtrdiffVec* columnMap,
                                  const std::string* hash) {

    auto stringToValue = [this](bool isCategorical, TStrSizeUMap& categoryLookup,
                                TStrVec& categories, const std::string& columnValue) {
        if (columnValue == m_MissingString) {
            ++m_MissingValueCount;
            return core::CFloatStorage{valueOfMissing()};
        }

        if (isCategorical) {
            // This encodes in a format suitable for efficient storage. The
            // actual encoding approach is chosen when the analysis runs.
            std::size_t id;
            if (categories.size() == MAX_CATEGORICAL_CARDINALITY) {
                auto itr = categoryLookup.find(columnValue);
                id = itr != categoryLookup.end()
                         ? itr->second
                         : static_cast<std::int64_t>(MAX_CATEGORICAL_CARDINALITY);
            } else {
                // We can represent up to float mantissa bits - 1 distinct
                // categories so can faithfully store categorical fields with
                // up to around 17M distinct values. For higher cardinalities
                // one would need to use some form of dimension reduction such
                // as hashing anyway.
                std::size_t newId{categories.size()};
                id = categoryLookup.emplace(columnValue, newId).first->second;
                if (id == newId) {
                    categories.push_back(columnValue);
                }
            }
            return core::CFloatStorage{static_cast<double>(id)};
        }

        // Use NaN to indicate missing or bad values in the data frame. This
        // needs handling with care from an analysis perspective. If analyses
        // can deal with missing values they need to treat NaNs as missing
        // otherwise we must impute or exit with failure.

        double value;
        if (core::CStringUtils::stringToTypeSilent(columnValue, value) == false) {
            ++m_BadValueCount;
            return core::CFloatStorage{valueOfMissing()};
        }

        // Tuncation is very unlikely since the values will typically be
        // standardised.
        return truncateToFloatRange(value);
    };

    // This is only used when writing rows so is resized lazily.
    if (m_CategoricalColumnValueLookup.size() != m_NumberColumns) {
        this->fillCategoricalColumnValueLookup();
    }

    this->writeRow([&](TFloatVecItr columns, std::int32_t& docHash) {
        if (columnMap != nullptr) {
            for (std::size_t i = 0; i < columnMap->size(); ++i, ++columns) {
                std::ptrdiff_t j{(*columnMap)[i]};
                *columns = stringToValue(m_ColumnIsCategorical[i],
                                         m_CategoricalColumnValueLookup[i],
                                         m_CategoricalColumnValues[i],
                                         j >= 0 ? columnValues[j] : m_MissingString);
            }
        } else {
            for (std::size_t i = 0; i < columnValues.size(); ++i, ++columns) {
                *columns = stringToValue(
                    m_ColumnIsCategorical[i], m_CategoricalColumnValueLookup[i],
                    m_CategoricalColumnValues[i], columnValues[i]);
            }
        }
        docHash = 0;
        if (hash != nullptr &&
            core::CStringUtils::stringToTypeSilent(*hash, docHash) == false) {
            ++m_BadDocHashCount;
        }
    });
}

void CDataFrame::writeRow(const TWriteFunc& writeRow) {
    if (m_Writer == nullptr) {
        m_Writer = std::make_unique<CDataFrameRowSliceWriter>(
            m_NumberRows, m_RowCapacity, m_SliceCapacityInRows,
            m_ReadAndWriteToStoreSyncStrategy, m_WriteSliceToStore);
    }
    (*m_Writer)(writeRow);
}

bool CDataFrame::hasColumnNames() const {
    return std::any_of(m_ColumnNames.begin(), m_ColumnNames.end(),
                       [](const auto& name) { return name.empty() == false; });
}

void CDataFrame::columnNames(TStrVec columnNames) {
    if (columnNames.size() != m_NumberColumns) {
        HANDLE_FATAL(<< "Expected '" << m_NumberColumns << "' column names values but got "
                     << columnNames.size() << ". The values are "
                     << CContainerPrinter::print(columnNames) << ".");
    } else {
        m_ColumnNames = std::move(columnNames);
    }
}

const std::string& CDataFrame::missingString() const {
    return m_MissingString;
}

void CDataFrame::missingString(std::string missing) {
    m_MissingString = std::move(missing);
}

void CDataFrame::categoricalColumns(TStrVec categoricalColumnNames) {
    std::sort(categoricalColumnNames.begin(), categoricalColumnNames.end());
    for (std::size_t i = 0; i < m_ColumnNames.size(); ++i) {
        auto categorical = std::lower_bound(categoricalColumnNames.begin(),
                                            categoricalColumnNames.end(),
                                            m_ColumnNames[i]);
        m_ColumnIsCategorical[i] = categorical != categoricalColumnNames.end() &&
                                   *categorical == m_ColumnNames[i];
    }
}

void CDataFrame::categoricalColumns(TBoolVec columnIsCategorical) {
    if (columnIsCategorical.size() != m_NumberColumns) {
        HANDLE_FATAL(<< "Expected '" << m_NumberColumns << "' 'is categorical' column indicator values but got "
                     << columnIsCategorical.size() << ". The values are "
                     << CContainerPrinter::print(columnIsCategorical) << ".");
    } else {
        m_ColumnIsCategorical = std::move(columnIsCategorical);
    }
}

void CDataFrame::categoricalColumnValues(TStrVecVec categoricalColumnValues) {
    if (categoricalColumnValues.size() != m_NumberColumns) {
        HANDLE_FATAL(<< "Expected '" << m_NumberColumns << "' categorical column values but got "
                     << categoricalColumnValues.size() << ". The values are "
                     << CContainerPrinter::print(categoricalColumnValues) << ".");
    } else {
        m_CategoricalColumnValues = std::move(categoricalColumnValues);
        this->fillCategoricalColumnValueLookup();
    }
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

    // Recover memory from categorical field parsing.

    for (std::size_t i = 0; i < m_CategoricalColumnValues.size(); ++i) {
        if (m_CategoricalColumnValues[i].size() >= MAX_CATEGORICAL_CARDINALITY) {
            LOG_WARN(<< "Failed to represent all distinct values of " << m_ColumnNames[i]);
        }
        m_CategoricalColumnValues.shrink_to_fit();
    }
    m_CategoricalColumnValueLookup.clear();
    m_CategoricalColumnValueLookup.shrink_to_fit();
}

const CDataFrame::TStrVec& CDataFrame::columnNames() const {
    return m_ColumnNames;
}

const CDataFrame::TStrVecVec& CDataFrame::categoricalColumnValues() const {
    return m_CategoricalColumnValues;
}

const CDataFrame::TBoolVec& CDataFrame::columnIsCategorical() const {
    return m_ColumnIsCategorical;
}

std::size_t CDataFrame::memoryUsage() const {
    std::size_t memory{memory::dynamicSize(m_ColumnNames)};
    memory += memory::dynamicSize(m_CategoricalColumnValues);
    memory += memory::dynamicSize(m_CategoricalColumnValueLookup);
    memory += memory::dynamicSize(m_MissingString);
    memory += memory::dynamicSize(m_ColumnIsCategorical);
    memory += memory::dynamicSize(m_Slices);
    memory += memory::dynamicSize(m_Writer);
    return memory;
}

std::uint64_t CDataFrame::checksum() const {
    std::vector<std::uint64_t> checksums(m_Slices.size(), 0);
    parallel_for_each(0, m_Slices.size(), [&](std::size_t index) {
        checksums[index] = m_Slices[index]->checksum();
    });

    std::uint64_t result{0};
    for (auto checksum : checksums) {
        result = CHashing::hashCombine(result, checksum);
    }
    return result;
}

std::size_t CDataFrame::estimateMemoryUsage(bool inMainMemory,
                                            std::size_t numberRows,
                                            std::size_t numberColumns,
                                            CAlignment::EType alignment) {
    return sizeof(CDataFrame) + core::memory::dynamicSize(TStrVec(numberColumns)) +
           core::memory::dynamicSize(TStrVecVec(numberColumns)) +
           core::memory::dynamicSize(TStrSizeUMapVec(numberColumns)) +
           core::memory::dynamicSize(TBoolVec(numberColumns)) +
           (inMainMemory ? numberRows * CAlignment::roundupSizeof<CFloatStorage>(alignment, numberColumns)
                         : 0);
    ;
}

void CDataFrame::fillCategoricalColumnValueLookup() {
    m_CategoricalColumnValueLookup.clear();
    m_CategoricalColumnValueLookup.resize(m_NumberColumns);
    for (std::size_t i = 0; i < m_CategoricalColumnValues.size(); ++i) {
        for (std::size_t j = 0; j < m_CategoricalColumnValues[i].size(); ++j) {
            m_CategoricalColumnValueLookup[i].emplace(m_CategoricalColumnValues[i][j], j);
        }
    }
}

bool CDataFrame::parallelApplyToAllRows(std::size_t beginRows,
                                        std::size_t endRows,
                                        TRowFuncVec& funcs,
                                        const CPackedBitVector* rowMask,
                                        bool commitResult) const {

    // If we're reading in parallel then we don't want to interleave reads
    // from storage and applying the function because we're already fully
    // balancing our work across the slices.

    using TSliceFuncVec = std::vector<std::function<void(const TRowSlicePtr&)>>;

    CPackedBitVector::COneBitIndexConstIterator maskedRow;
    CPackedBitVector::COneBitIndexConstIterator endMaskedRows;
    if (rowMask != nullptr) {
        maskedRow = rowMask->beginOneBits();
        endMaskedRows = rowMask->endOneBits();
    }

    std::atomic_bool successful{true};
    CDataFrameRowSliceHandle readSlice;

    TSliceFuncVec sliceFuncs;
    sliceFuncs.reserve(funcs.size());

    for (auto& func : funcs) {
        sliceFuncs.push_back([=, &func, &successful](const TRowSlicePtr& slice) mutable {
            if (successful.load() == false) {
                return;
            }

            std::size_t beginSliceRows{std::max(slice->indexOfFirstRow(), beginRows)};
            std::size_t endSliceRows{
                std::min(slice->indexOfLastRow(m_RowCapacity) + 1, endRows)};

            if (rowMask != nullptr &&
                this->maskedRowsInSlice(maskedRow, endMaskedRows,
                                        beginSliceRows, endSliceRows) == false) {
                return;
            }

            readSlice = slice->read();
            if (readSlice.bad()) {
                successful.store(false);
                return;
            }

            TOptionalPopMaskedRow popMaskedRow;
            if (rowMask != nullptr) {
                beginSliceRows = *maskedRow;
                popMaskedRow = CPopMaskedRow{endSliceRows, maskedRow, endMaskedRows};
            }

            this->applyToRowsOfOneSlice(func, beginSliceRows, endSliceRows,
                                        popMaskedRow, readSlice);
            if (commitResult) {
                slice->write(readSlice.rows(), readSlice.docHashes());
            }
        });
    }

    parallel_for_each(this->beginSlices(beginRows), this->endSlices(endRows), sliceFuncs);

    return successful.load();
}

bool CDataFrame::sequentialApplyToAllRows(std::size_t beginRows,
                                          std::size_t endRows,
                                          TRowFuncVec& func,
                                          const CPackedBitVector* rowMask,
                                          bool commitResult) const {

    CPackedBitVector::COneBitIndexConstIterator maskedRow;
    CPackedBitVector::COneBitIndexConstIterator endMaskedRows;
    if (rowMask != nullptr) {
        maskedRow = rowMask->beginOneBits();
        endMaskedRows = rowMask->endOneBits();
    }

    CDataFrameRowSliceHandle readSlice;

    switch (m_ReadAndWriteToStoreSyncStrategy) {
    case CDataFrame::EReadWriteToStorage::E_Async: {
        // The slices get read from storage on the thread executing this
        // function each slice is then concurrently read by the callback
        // on a worker thread.

        std::future<void> backgroundApply;

        // We need to wait and this isn't guaranteed by the future destructor.
        CWaitIfValidWhenExitingScope<void> waitFor(backgroundApply);

        for (auto slice = this->beginSlices(beginRows), endSlices = this->endSlices(endRows);
             slice != endSlices; ++slice) {

            std::size_t beginSliceRows{std::max((*slice)->indexOfFirstRow(), beginRows)};
            std::size_t endSliceRows{
                std::min((*slice)->indexOfLastRow(m_RowCapacity) + 1, endRows)};

            if (rowMask != nullptr &&
                this->maskedRowsInSlice(maskedRow, endMaskedRows,
                                        beginSliceRows, endSliceRows) == false) {
                continue;
            }

            readSlice = (*slice)->read();
            if (readSlice.bad()) {
                return false;
            }

            // We wait here so at most one slice is copied into memory.
            wait_for_valid(backgroundApply);

            backgroundApply = async(
                defaultAsyncExecutor(),
                [ =, &func, readSlice_ = std::move(readSlice) ]() mutable {

                    TOptionalPopMaskedRow popMaskedRow;
                    if (rowMask != nullptr) {
                        beginSliceRows = *maskedRow;
                        popMaskedRow = CPopMaskedRow{endSliceRows, maskedRow, endMaskedRows};
                    }

                    this->applyToRowsOfOneSlice(func[0], beginSliceRows, endSliceRows,
                                                popMaskedRow, readSlice_);

                    if (commitResult) {
                        (*slice)->write(readSlice_.rows(), readSlice_.docHashes());
                    }
                });
        }
        break;
    }
    case CDataFrame::EReadWriteToStorage::E_Sync:
        for (auto slice = this->beginSlices(beginRows), endSlices = this->endSlices(endRows);
             slice != endSlices; ++slice) {

            std::size_t beginSliceRows{std::max((*slice)->indexOfFirstRow(), beginRows)};
            std::size_t endSliceRows{
                std::min((*slice)->indexOfLastRow(m_RowCapacity) + 1, endRows)};

            if (rowMask != nullptr &&
                this->maskedRowsInSlice(maskedRow, endMaskedRows,
                                        beginSliceRows, endSliceRows) == false) {
                continue;
            }

            readSlice = (*slice)->read();
            if (readSlice.bad()) {
                return false;
            }

            TOptionalPopMaskedRow popMaskedRow;
            if (rowMask != nullptr) {
                beginSliceRows = *maskedRow;
                popMaskedRow = CPopMaskedRow{endSliceRows, maskedRow, endMaskedRows};
            }

            this->applyToRowsOfOneSlice(func[0], beginSliceRows, endSliceRows,
                                        popMaskedRow, readSlice);

            if (commitResult) {
                (*slice)->write(readSlice.rows(), readSlice.docHashes());
            }
        }
        break;
    }

    return true;
}

void CDataFrame::applyToRowsOfOneSlice(TRowFunc& func,
                                       std::size_t firstRowToRead,
                                       std::size_t endRowsToRead,
                                       const TOptionalPopMaskedRow& popMaskedRow,
                                       const CDataFrameRowSliceHandle& slice) const {

    LOG_TRACE(<< "Applying function to rows [" << firstRowToRead << ","
              << endRowsToRead << ")");

    std::size_t offsetOfFirstRowToRead{firstRowToRead - slice.indexOfFirstRow()};
    std::size_t offsetOfEndRowsToRead{endRowsToRead - slice.indexOfFirstRow()};

    std::size_t beginRowData{offsetOfFirstRowToRead * m_RowCapacity};
    std::size_t endRowData{offsetOfEndRowsToRead * m_RowCapacity};

    func(CRowIterator{m_NumberColumns, m_RowCapacity, firstRowToRead,
                      slice.beginRows() + beginRowData,
                      slice.beginDocHashes() + offsetOfFirstRowToRead, popMaskedRow},
         CRowIterator{m_NumberColumns, m_RowCapacity, endRowsToRead,
                      slice.beginRows() + endRowData,
                      slice.beginDocHashes() + offsetOfEndRowsToRead, popMaskedRow});
}

CDataFrame::TRowSlicePtrVecCItr CDataFrame::beginSlices(std::size_t beginRows) const {
    return std::upper_bound(m_Slices.begin(), m_Slices.end(), beginRows,
                            [](std::size_t row, const TRowSlicePtr& slice) {
                                return row < slice->indexOfFirstRow();
                            }) -
           1;
}

CDataFrame::TRowSlicePtrVecCItr CDataFrame::endSlices(std::size_t endRows) const {
    return std::upper_bound(m_Slices.begin(), m_Slices.end(), endRows,
                            [](std::size_t row, const TRowSlicePtr& slice) {
                                return row < slice->indexOfFirstRow();
                            });
}

template<typename ITR>
bool CDataFrame::maskedRowsInSlice(ITR& maskedRow,
                                   ITR endMaskedRows,
                                   std::size_t beginSliceRows,
                                   std::size_t endSliceRows) const {
    while (maskedRow != endMaskedRows && *maskedRow < beginSliceRows) {
        ++maskedRow;
    }
    return maskedRow != endMaskedRows && *maskedRow < endSliceRows;
}

const std::size_t CDataFrame::MAX_CATEGORICAL_CARDINALITY{
    1 << (std::numeric_limits<float>::digits)};
const std::string CDataFrame::DEFAULT_MISSING_STRING{"\0", 1};

CDataFrame::CDataFrameRowSliceWriter::CDataFrameRowSliceWriter(
    std::size_t numberRows,
    std::size_t rowCapacity,
    std::size_t sliceCapacityInRows,
    EReadWriteToStorage writeToStoreSyncStrategy,
    TWriteSliceToStoreFunc writeSliceToStore)
    : m_NumberRows{numberRows}, m_RowCapacity{rowCapacity}, m_SliceCapacityInRows{sliceCapacityInRows},
      m_WriteToStoreSyncStrategy{writeToStoreSyncStrategy}, m_WriteSliceToStore{writeSliceToStore} {
    m_RowsOfSliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
    m_DocHashesOfSliceBeingWritten.reserve(m_SliceCapacityInRows);
}

void CDataFrame::CDataFrameRowSliceWriter::operator()(const TWriteFunc& writeRow) {
    // Write the next row at the end of the current slice being written
    // and if the slice is full pass to the thread storing slices.

    std::size_t start{m_RowsOfSliceBeingWritten.size()};
    m_RowsOfSliceBeingWritten.resize(start + m_RowCapacity);
    m_DocHashesOfSliceBeingWritten.emplace_back();
    writeRow(m_RowsOfSliceBeingWritten.begin() + start,
             m_DocHashesOfSliceBeingWritten.back());
    ++m_NumberRows;

    if (m_DocHashesOfSliceBeingWritten.size() == m_SliceCapacityInRows) {
        std::size_t firstRow{m_NumberRows - m_SliceCapacityInRows};
        LOG_TRACE(<< "Storing slice [" << firstRow << "," << m_NumberRows << ")");

        switch (m_WriteToStoreSyncStrategy) {
        case EReadWriteToStorage::E_Async: {
            if (m_SliceWrittenAsyncToStore.valid()) {
                m_SlicesWrittenToStore.push_back(m_SliceWrittenAsyncToStore.get());
            }
            m_SliceWrittenAsyncToStore =
                async(defaultAsyncExecutor(), m_WriteSliceToStore, firstRow,
                      std::move(m_RowsOfSliceBeingWritten),
                      std::move(m_DocHashesOfSliceBeingWritten));
            break;
        }
        case EReadWriteToStorage::E_Sync:
            m_SlicesWrittenToStore.push_back(
                m_WriteSliceToStore(firstRow, std::move(m_RowsOfSliceBeingWritten),
                                    std::move(m_DocHashesOfSliceBeingWritten)));
            break;
        }
        m_RowsOfSliceBeingWritten.clear();
        m_DocHashesOfSliceBeingWritten.clear();
        m_RowsOfSliceBeingWritten.reserve(m_SliceCapacityInRows * m_RowCapacity);
        m_DocHashesOfSliceBeingWritten.reserve(m_SliceCapacityInRows);
    }
}

CDataFrame::TSizeDataFrameRowSlicePtrVecPr
CDataFrame::CDataFrameRowSliceWriter::finishWritingRows() {
    if (m_SliceWrittenAsyncToStore.valid()) {
        m_SlicesWrittenToStore.push_back(m_SliceWrittenAsyncToStore.get());
    }

    if (m_DocHashesOfSliceBeingWritten.empty() == false) {
        std::size_t firstRow{m_NumberRows - m_RowsOfSliceBeingWritten.size() / m_RowCapacity};
        LOG_TRACE(<< "Last slice [" << std::to_string(firstRow) << ","
                  << std::to_string(m_NumberRows) + ")");
        m_SlicesWrittenToStore.push_back(
            m_WriteSliceToStore(firstRow, std::move(m_RowsOfSliceBeingWritten),
                                std::move(m_DocHashesOfSliceBeingWritten)));
    }

    return {m_NumberRows, std::move(m_SlicesWrittenToStore)};
}

std::size_t dataFrameDefaultSliceCapacity(std::size_t numberColumns) {
    // There is some overhead traversing the data frame for each chunk we
    // use. We also on average get better locality of reference by using
    // larger chunks. However, if we set the chunk size too large it won't
    // fit in cache and it also makes masked access of disk backed frames
    // more expensive. This is at the upper end of L2 and lower end of L3
    // cache size and performance testing shows it provides a reasonable
    // tradeoff without the trouble of trying to portably determine cache
    // sizes at runtime.
    std::size_t eightMbChunkSize{8 * constants::BYTES_IN_MEGABYTES /
                                 sizeof(CFloatStorage) / numberColumns};
    return std::max(eightMbChunkSize, std::size_t{128});
}

std::pair<std::unique_ptr<CDataFrame>, std::shared_ptr<CTemporaryDirectory>>
makeMainStorageDataFrame(std::size_t numberColumns,
                         std::optional<std::size_t> sliceCapacity,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy,
                         CAlignment::EType alignment) {
    auto writer = [](std::size_t firstRow, TFloatVec rows, TInt32Vec docHashes) {
        return std::make_unique<CMainMemoryDataFrameRowSlice>(
            firstRow, std::move(rows), std::move(docHashes));
    };

    if (sliceCapacity == std::nullopt) {
        sliceCapacity = dataFrameDefaultSliceCapacity(numberColumns);
    }

    return {std::make_unique<CDataFrame>(true, numberColumns, alignment, *sliceCapacity,
                                         readWriteToStoreSyncStrategy, writer),
            nullptr};
}

std::pair<std::unique_ptr<CDataFrame>, std::shared_ptr<CTemporaryDirectory>>
makeDiskStorageDataFrame(const std::string& rootDirectory,
                         std::size_t numberColumns,
                         std::size_t numberRows,
                         std::optional<std::size_t> sliceCapacity,
                         CDataFrame::EReadWriteToStorage readWriteToStoreSyncStrategy,
                         CAlignment::EType alignment) {
    std::size_t minimumSpace{2 * numberRows * numberColumns * sizeof(CFloatStorage)};

    auto directory = std::make_shared<CTemporaryDirectory>(rootDirectory, minimumSpace);

    // Note the writer lambda holding a reference to the directory shared
    // pointer is copied to the data frame. So this isn't destroyed, and
    // the folder cleaned up, until the data frame itself is destroyed.

    auto writer = [directory](std::size_t firstRow, TFloatVec rows, TInt32Vec docHashes) {
        return std::make_unique<COnDiskDataFrameRowSlice>(
            directory, firstRow, std::move(rows), std::move(docHashes));
    };

    if (sliceCapacity == std::nullopt) {
        sliceCapacity = dataFrameDefaultSliceCapacity(numberColumns);
    }

    return {std::make_unique<CDataFrame>(false, numberColumns, alignment, *sliceCapacity,
                                         readWriteToStoreSyncStrategy, writer),
            directory};
}
}
}
