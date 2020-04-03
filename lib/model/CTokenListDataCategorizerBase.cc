/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CTokenListDataCategorizerBase.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>

#include <maths/COrderings.h>

#include <model/CTokenListReverseSearchCreator.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <set>

namespace ml {
namespace model {

// Initialise statics
const std::string CTokenListDataCategorizerBase::PRETOKENISED_TOKEN_FIELD{"..."};

// We use short field names to reduce the state size
namespace {
const std::string TOKEN_TAG{"a"};
const std::string TOKEN_CATEGORY_COUNT_TAG{"b"};
const std::string CATEGORY_TAG{"c"};
}

CTokenListDataCategorizerBase::CTokenListDataCategorizerBase(CLimits& limits,
                                                             const TTokenListReverseSearchCreatorCPtr& reverseSearchCreator,
                                                             double threshold,
                                                             const std::string& fieldName)
    : CDataCategorizer{limits, fieldName}, m_ReverseSearchCreator{reverseSearchCreator},
      m_LowerThreshold{std::min(0.99, std::max(0.01, threshold))},
      // Upper threshold is half way between the lower threshold and 1
      m_UpperThreshold{(1.0 + m_LowerThreshold) / 2.0}, m_HasChanged{false} {
}

void CTokenListDataCategorizerBase::dumpStats() const {
    // ML category number is vector index plus one
    int categoryId{1};
    for (const auto& category : m_Categories) {
        LOG_DEBUG(<< "ML category=" << categoryId << '-'
                  << category.numMatches() << ' ' << category.baseString());
        ++categoryId;
    }
}

int CTokenListDataCategorizerBase::computeCategory(bool isDryRun,
                                                   const TStrStrUMap& fields,
                                                   const std::string& str,
                                                   std::size_t rawStringLen) {
    // First tokenise string
    std::size_t workWeight{0};
    auto preTokenisedIter = fields.find(PRETOKENISED_TOKEN_FIELD);
    if (preTokenisedIter != fields.end()) {
        if (this->addPretokenisedTokens(preTokenisedIter->second, m_WorkTokenIds,
                                        m_WorkTokenUniqueIds, workWeight) == false) {
            return -1;
        }
    } else {
        this->tokeniseString(fields, str, m_WorkTokenIds, m_WorkTokenUniqueIds, workWeight);
    }

    // Determine the minimum and maximum token weight that could possibly
    // match the weight we've got
    std::size_t minWeight{CTokenListDataCategorizerBase::minMatchingWeight(
        workWeight, m_LowerThreshold)};
    std::size_t maxWeight{CTokenListDataCategorizerBase::maxMatchingWeight(
        workWeight, m_LowerThreshold)};

    // We search previous categories in descending order of the number of matches
    // we've seen for them
    auto bestSoFarIter = m_CategoriesByCount.end();
    double bestSoFarSimilarity(m_LowerThreshold);
    for (auto iter = m_CategoriesByCount.begin(); iter != m_CategoriesByCount.end(); ++iter) {
        const CTokenListCategory& compCategory = m_Categories[iter->second];
        const TSizeSizePrVec& baseTokenIds = compCategory.baseTokenIds();
        std::size_t baseWeight(compCategory.baseWeight());

        // Check whether the current record matches the search for the existing
        // category - if it does then we'll put it in the existing category without any
        // further checks.  The first condition here ensures that we never say
        // a string with tokens matches the reverse search of a string with no
        // tokens (which the other criteria alone might say matched).
        bool matchesSearch{
            (baseWeight == 0) == (workWeight == 0) &&
            compCategory.maxMatchingStringLen() >= rawStringLen &&
            compCategory.isMissingCommonTokenWeightZero(m_WorkTokenUniqueIds) &&
            compCategory.containsCommonInOrderTokensInOrder(m_WorkTokenIds)};
        if (!matchesSearch) {
            // Quickly rule out wildly different token weights prior to doing
            // the expensive similarity calculations
            if (baseWeight < minWeight || baseWeight > maxWeight) {
                continue;
            }

            // Rule out categories where adding the current string would unacceptably
            // reduce the number of unique common tokens
            std::size_t origUniqueTokenWeight(compCategory.origUniqueTokenWeight());
            std::size_t commonUniqueTokenWeight(compCategory.commonUniqueTokenWeight());
            std::size_t missingCommonTokenWeight(
                compCategory.missingCommonTokenWeight(m_WorkTokenUniqueIds));
            double proportionOfOrig(double(commonUniqueTokenWeight - missingCommonTokenWeight) /
                                    double(origUniqueTokenWeight));
            if (proportionOfOrig < m_LowerThreshold) {
                continue;
            }
        }

        double similarity{this->similarity(m_WorkTokenIds, workWeight, baseTokenIds, baseWeight)};

        LOG_TRACE(<< similarity << '-' << compCategory.baseString() << '|' << str);

        if (matchesSearch || similarity > m_UpperThreshold) {
            if (similarity <= m_LowerThreshold) {
                // Not an ideal situation, but log at trace level to avoid
                // excessive log file spam
                LOG_TRACE(<< "Reverse search match below threshold : " << similarity
                          << '-' << compCategory.baseString() << '|' << str);
            }

            // This is a strong match, so accept it immediately and stop
            // looking for better matches - use vector index plus one as category
            int categoryId(1 + int(iter->second));
            this->addCategoryMatch(isDryRun, str, rawStringLen, m_WorkTokenIds,
                                   m_WorkTokenUniqueIds, iter);
            return categoryId;
        }

        if (similarity > bestSoFarSimilarity) {
            // This is a weak match, but remember it because it's the best we've
            // seen
            bestSoFarIter = iter;
            bestSoFarSimilarity = similarity;

            // Recalculate the minimum and maximum token counts that might
            // produce a better match
            minWeight = CTokenListDataCategorizerBase::minMatchingWeight(workWeight, similarity);
            maxWeight = CTokenListDataCategorizerBase::maxMatchingWeight(workWeight, similarity);
        }
    }

    if (bestSoFarIter != m_CategoriesByCount.end()) {
        // Return the best match - use vector index plus one as ML category
        int categoryId{1 + static_cast<int>(bestSoFarIter->second)};
        this->addCategoryMatch(isDryRun, str, rawStringLen, m_WorkTokenIds,
                               m_WorkTokenUniqueIds, bestSoFarIter);
        return categoryId;
    }

    // If we get here we haven't matched, so create a new category
    m_CategoriesByCount.emplace_back(1, m_Categories.size());
    m_Categories.emplace_back(isDryRun, str, rawStringLen, m_WorkTokenIds,
                              workWeight, m_WorkTokenUniqueIds);
    m_HasChanged = true;

    // Increment the counts of categories that use a given token
    for (const auto& workTokenId : m_WorkTokenIds) {
        // We get away with casting away constness ONLY because the category count
        // is not used in any of the multi-index keys
        const_cast<CTokenInfoItem&>(m_TokenIdLookup[workTokenId.first]).incCategoryCount();
    }

    // ML category is vector index plus one
    return static_cast<int>(m_Categories.size());
}

bool CTokenListDataCategorizerBase::createReverseSearch(int categoryId,
                                                        std::string& part1,
                                                        std::string& part2,
                                                        std::size_t& maxMatchingLength,
                                                        bool& wasCached) {
    if (m_ReverseSearchCreator == nullptr) {
        LOG_ERROR(<< "Cannot create reverse search - no reverse search creator");

        part1.clear();
        part2.clear();

        return false;
    }

    // Find the correct category object - ML category is vector index plus one
    if (categoryId < 1 || static_cast<std::size_t>(categoryId) > m_Categories.size()) {

        part1.clear();
        part2.clear();

        // -1 is supposed to be the only special value used for the category ID.
        if (categoryId != -1) {
            LOG_ERROR(<< "Programmatic error - invalid ML category: " << categoryId);
            return false;
        }

        return true;
    }

    CTokenListCategory& category{m_Categories[categoryId - 1]};
    maxMatchingLength = category.maxMatchingStringLen();

    // If we can retrieve cached reverse search terms we'll save a lot of time
    if (category.cachedReverseSearch(part1, part2) == true) {
        wasCached = true;
        return true;
    }

    const TSizeSizePrVec& baseTokenIds{category.baseTokenIds()};
    const TSizeSizePrVec& commonUniqueTokenIds{category.commonUniqueTokenIds()};
    if (commonUniqueTokenIds.empty()) {
        // There's quite a high chance this call will return false
        if (m_ReverseSearchCreator->createNoUniqueTokenSearch(
                categoryId, category.baseString(),
                category.maxMatchingStringLen(), part1, part2) == false) {
            // More detail should have been logged by the failed call
            LOG_ERROR(<< "Could not create reverse search");

            part1.clear();
            part2.clear();

            return false;
        }

        category.cacheReverseSearch(part1, part2);

        return true;
    }

    std::size_t availableCost{m_ReverseSearchCreator->availableCost()};

    // Determine the rarest tokens that we can afford within the available
    // length
    using TSizeSizeSizePrMMap = std::multimap<std::size_t, TSizeSizePr>;
    TSizeSizeSizePrMMap rareIdsWithCost;
    std::size_t lowestCost{std::numeric_limits<std::size_t>::max()};
    for (const auto& commonUniqueTokenId : commonUniqueTokenIds) {
        std::size_t tokenId{commonUniqueTokenId.first};
        std::size_t occurrences{static_cast<std::size_t>(std::count_if(
            baseTokenIds.begin(), baseTokenIds.end(), CSizePairFirstElementEquals(tokenId)))};
        const CTokenInfoItem& info{m_TokenIdLookup[tokenId]};
        std::size_t cost{m_ReverseSearchCreator->costOfToken(info.str(), occurrences)};
        rareIdsWithCost.insert(TSizeSizeSizePrMMap::value_type(
            info.categoryCount(), TSizeSizePr(tokenId, cost)));
        lowestCost = std::min(cost, lowestCost);
    }

    using TSizeSet = std::set<std::size_t>;
    TSizeSet costedCommonUniqueTokenIds;
    std::size_t cheapestCost{std::numeric_limits<std::size_t>::max()};
    auto cheapestIter = rareIdsWithCost.end();
    for (auto iter = rareIdsWithCost.begin();
         iter != rareIdsWithCost.end() && availableCost > lowestCost; ++iter) {
        if (iter->second.second < cheapestCost) {
            cheapestCost = iter->second.second;
            cheapestIter = iter;
        }

        if (availableCost < iter->second.second) {
            // We can't afford this token
            continue;
        }

        // By this point we don't care about the weights or costs
        costedCommonUniqueTokenIds.insert(iter->second.first);
        availableCost -= iter->second.second;
    }

    if (costedCommonUniqueTokenIds.empty()) {
        if (cheapestIter == rareIdsWithCost.end()) {
            LOG_ERROR(<< "Inconsistency - rareIdsWithCost is empty but "
                         "commonUniqueTokenIds wasn't for "
                      << categoryId);
        } else {
            LOG_ERROR(<< "No token was short enough to include in reverse search "
                         "for "
                      << categoryId << " - cheapest token was "
                      << cheapestIter->second.first << " with cost " << cheapestCost);
        }

        part1.clear();
        part2.clear();

        return false;
    }

    // If we get here we're going to create a search in the standard way - there
    // shouldn't be any more errors after this point

    m_ReverseSearchCreator->initStandardSearch(categoryId, category.baseString(),
                                               category.maxMatchingStringLen(),
                                               part1, part2);

    TSizeSizePr orderedCommonTokenBounds{category.orderedCommonTokenBounds()};
    for (std::size_t index = 0; index < baseTokenIds.size(); ++index) {
        std::size_t tokenId(baseTokenIds[index].first);
        if (costedCommonUniqueTokenIds.find(tokenId) !=
            costedCommonUniqueTokenIds.end()) {
            if (index >= orderedCommonTokenBounds.first &&
                index < orderedCommonTokenBounds.second) {
                m_ReverseSearchCreator->addInOrderCommonToken(
                    m_TokenIdLookup[tokenId].str(), part1, part2);
            } else {
                m_ReverseSearchCreator->addOutOfOrderCommonToken(
                    m_TokenIdLookup[tokenId].str(), part1, part2);
            }
        }
    }

    m_ReverseSearchCreator->closeStandardSearch(part1, part2);

    category.cacheReverseSearch(part1, part2);

    return true;
}

bool CTokenListDataCategorizerBase::hasChanged() const {
    return m_HasChanged;
}

bool CTokenListDataCategorizerBase::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_Categories.clear();
    m_CategoriesByCount.clear();
    m_TokenIdLookup.clear();
    m_WorkTokenIds.clear();
    m_WorkTokenUniqueIds.clear();
    m_HasChanged = false;

    do {
        const std::string& name{traverser.name()};
        if (name == TOKEN_TAG) {
            std::size_t nextIndex(m_TokenIdLookup.size());
            m_TokenIdLookup.push_back(CTokenInfoItem(traverser.value(), nextIndex));
        } else if (name == TOKEN_CATEGORY_COUNT_TAG) {
            if (m_TokenIdLookup.empty()) {
                LOG_ERROR(<< "Token category count precedes token string in "
                          << traverser.value());
                return false;
            }

            std::size_t categoryCount{0};
            if (core::CStringUtils::stringToType(traverser.value(), categoryCount) == false) {
                LOG_ERROR(<< "Invalid token category count in " << traverser.value());
                return false;
            }

            // We get away with casting away constness ONLY because the category
            // count is not used in any of the multi-index keys
            const_cast<CTokenInfoItem&>(m_TokenIdLookup.back()).categoryCount(categoryCount);
        } else if (name == CATEGORY_TAG) {
            CTokenListCategory category{traverser};
            m_CategoriesByCount.emplace_back(category.numMatches(), m_Categories.size());
            m_Categories.push_back(category);
        }
    } while (traverser.next());

    // Categories are persisted in order of creation, but this list needs to be
    // sorted by count instead
    std::sort(m_CategoriesByCount.begin(), m_CategoriesByCount.end(),
              maths::COrderings::SFirstGreater());

    return true;
}

void CTokenListDataCategorizerBase::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    CTokenListDataCategorizerBase::acceptPersistInserter(m_TokenIdLookup,
                                                         m_Categories, inserter);
}

void CTokenListDataCategorizerBase::acceptPersistInserter(const TTokenMIndex& tokenIdLookup,
                                                          const TTokenListCategoryVec& categories,
                                                          core::CStatePersistInserter& inserter) {
    for (const CTokenInfoItem& item : tokenIdLookup) {
        inserter.insertValue(TOKEN_TAG, item.str());
        inserter.insertValue(TOKEN_CATEGORY_COUNT_TAG, item.categoryCount());
    }

    for (const CTokenListCategory& category : categories) {
        inserter.insertLevel(CATEGORY_TAG,
                             std::bind(&CTokenListCategory::acceptPersistInserter,
                                       &category, std::placeholders::_1));
    }
}

CDataCategorizer::TPersistFunc CTokenListDataCategorizerBase::makeForegroundPersistFunc() const {
    return std::bind(
        static_cast<void (*)(const TTokenMIndex&, const TTokenListCategoryVec&, core::CStatePersistInserter&)>(
            &CTokenListDataCategorizerBase::acceptPersistInserter),
        std::cref(m_TokenIdLookup), std::cref(m_Categories), std::placeholders::_1);
}

CDataCategorizer::TPersistFunc CTokenListDataCategorizerBase::makeBackgroundPersistFunc() const {
    return std::bind(
        static_cast<void (*)(const TTokenMIndex&, const TTokenListCategoryVec&, core::CStatePersistInserter&)>(
            &CTokenListDataCategorizerBase::acceptPersistInserter),
        // Do NOT add std::ref wrappers around these arguments - they MUST be
        // copied for thread safety
        m_TokenIdLookup, m_Categories, std::placeholders::_1);
}

void CTokenListDataCategorizerBase::addCategoryMatch(bool isDryRun,
                                                     const std::string& str,
                                                     std::size_t rawStringLen,
                                                     const TSizeSizePrVec& tokenIds,
                                                     const TSizeSizeMap& tokenUniqueIds,
                                                     TSizeSizePrVecItr& iter) {
    if (m_Categories[iter->second].addString(isDryRun, str, rawStringLen,
                                             tokenIds, tokenUniqueIds) == true) {
        m_HasChanged = true;
    }

    std::size_t& count{iter->first};
    ++count;

    // Search backwards for the point where the incremented count belongs
    auto swapIter = m_CategoriesByCount.end();
    auto checkIter = iter;
    while (checkIter != m_CategoriesByCount.begin()) {
        --checkIter;
        if (count <= checkIter->first) {
            break;
        }
        swapIter = checkIter;
    }

    // Move the iterator we've matched nearer the front of the list if it
    // deserves this
    if (swapIter != m_CategoriesByCount.end()) {
        std::iter_swap(swapIter, iter);
    }
}

std::size_t CTokenListDataCategorizerBase::minMatchingWeight(std::size_t weight,
                                                             double threshold) {
    if (weight == 0) {
        return 0;
    }

    // When we build with aggressive optimisation, the result of the floating
    // point multiplication can be slightly out, so add a small amount of
    // tolerance
    static const double EPSILON{0.00000000001};

    // This assumes threshold is not negative - other code in this file must
    // enforce this.  Using floor + 1 due to threshold check being exclusive.
    // If threshold check is changed to inclusive, change formula to ceil
    // (without the + 1).
    return static_cast<std::size_t>(std::floor(double(weight) * threshold + EPSILON)) + 1;
}

std::size_t CTokenListDataCategorizerBase::maxMatchingWeight(std::size_t weight,
                                                             double threshold) {
    if (weight == 0) {
        return 0;
    }

    // When we build with aggressive optimisation, the result of the floating
    // point division can be slightly out, so subtract a small amount of
    // tolerance
    static const double EPSILON{0.00000000001};

    // This assumes threshold is not negative - other code in this file must
    // enforce this.  Using ceil - 1 due to threshold check being exclusive.
    // If threshold check is changed to inclusive, change formula to floor
    // (without the - 1).
    return static_cast<std::size_t>(std::ceil(double(weight) / threshold - EPSILON)) - 1;
}

std::size_t CTokenListDataCategorizerBase::idForToken(const std::string& token) {
    auto iter = boost::multi_index::get<SToken>(m_TokenIdLookup).find(token);
    if (iter != boost::multi_index::get<SToken>(m_TokenIdLookup).end()) {
        return iter->index();
    }

    std::size_t nextIndex{m_TokenIdLookup.size()};
    m_TokenIdLookup.push_back(CTokenInfoItem(token, nextIndex));
    return nextIndex;
}

bool CTokenListDataCategorizerBase::addPretokenisedTokens(const std::string& tokensCsv,
                                                          TSizeSizePrVec& tokenIds,
                                                          TSizeSizeMap& tokenUniqueIds,
                                                          std::size_t& totalWeight) {
    tokenIds.clear();
    tokenUniqueIds.clear();
    totalWeight = 0;

    m_CsvLineParser.reset(tokensCsv);
    std::string token;
    while (!m_CsvLineParser.atEnd()) {
        if (m_CsvLineParser.parseNext(token) == false) {
            return false;
        }

        this->tokenToIdAndWeight(token, tokenIds, tokenUniqueIds, totalWeight);
    }

    return true;
}

void CTokenListDataCategorizerBase::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTokenListDataCategorizerBase");
    this->CDataCategorizer::debugMemoryUsage(mem->addChild());
    core::CMemoryDebug::dynamicSize("m_ReverseSearchCreator", m_ReverseSearchCreator, mem);
    core::CMemoryDebug::dynamicSize("m_Categories", m_Categories, mem);
    core::CMemoryDebug::dynamicSize("m_CategoriesByCount", m_CategoriesByCount, mem);
    core::CMemoryDebug::dynamicSize("m_TokenIdLookup", m_TokenIdLookup, mem);
    core::CMemoryDebug::dynamicSize("m_WorkTokenIds", m_WorkTokenIds, mem);
    core::CMemoryDebug::dynamicSize("m_WorkTokenUniqueIds", m_WorkTokenUniqueIds, mem);
    core::CMemoryDebug::dynamicSize("m_CsvLineParser", m_CsvLineParser, mem);
}

std::size_t CTokenListDataCategorizerBase::memoryUsage() const {
    std::size_t mem = this->CDataCategorizer::memoryUsage();
    mem += core::CMemory::dynamicSize(m_ReverseSearchCreator);
    mem += core::CMemory::dynamicSize(m_Categories);
    mem += core::CMemory::dynamicSize(m_CategoriesByCount);
    mem += core::CMemory::dynamicSize(m_TokenIdLookup);
    mem += core::CMemory::dynamicSize(m_WorkTokenIds);
    mem += core::CMemory::dynamicSize(m_WorkTokenUniqueIds);
    mem += core::CMemory::dynamicSize(m_CsvLineParser);
    return mem;
}

void CTokenListDataCategorizerBase::updateModelSizeStats(CResourceMonitor::SModelSizeStats& modelSizeStats) const {

    modelSizeStats.s_TotalCategories = m_Categories.size();

    std::size_t categorizedMessagesThisCategorizer{0};
    for (auto categoryByCount : m_CategoriesByCount) {
        categorizedMessagesThisCategorizer += categoryByCount.first;
    }
    modelSizeStats.s_CategorizedMessages += categorizedMessagesThisCategorizer;

    for (std::size_t i = 0; i < m_CategoriesByCount.size(); ++i) {
        const CTokenListCategory& category{m_Categories[m_CategoriesByCount[i].second]};
        // Definitions for frequent/rare categories are:
        // - rare = single match
        // - frequent = matches more than 1% of messages
        if (category.numMatches() == 1) {
            ++modelSizeStats.s_RareCategories;
        } else if (category.numMatches() * 100 > categorizedMessagesThisCategorizer) {
            ++modelSizeStats.s_FrequentCategories;
        }
        for (std::size_t j = 0; j < i; ++j) {
            const CTokenListCategory& moreFrequentCategory{
                m_Categories[m_CategoriesByCount[j].second]};
            bool matchesSearch{moreFrequentCategory.maxMatchingStringLen() >=
                                   category.maxMatchingStringLen() &&
                               moreFrequentCategory.isMissingCommonTokenWeightZero(
                                   category.commonUniqueTokenIds()) &&
                               moreFrequentCategory.containsCommonInOrderTokensInOrder(
                                   category.baseTokenIds())};
            if (matchesSearch) {
                ++modelSizeStats.s_DeadCategories;
                LOG_DEBUG(<< "Category " << (m_CategoriesByCount[i].second + 1)
                          << " (" << category.baseString() << ") is killed by category "
                          << (m_CategoriesByCount[j].second + 1) << " ("
                          << moreFrequentCategory.baseString() << ")");
                break;
            }
        }
    }

    modelSizeStats.s_CategorizationStatus = CTokenListDataCategorizerBase::calculateCategorizationStatus(
        modelSizeStats.s_CategorizedMessages, modelSizeStats.s_TotalCategories,
        modelSizeStats.s_FrequentCategories, modelSizeStats.s_RareCategories,
        modelSizeStats.s_DeadCategories);
}

model_t::ECategorizationStatus
CTokenListDataCategorizerBase::calculateCategorizationStatus(std::size_t categorizedMessages,
                                                             std::size_t totalCategories,
                                                             std::size_t frequentCategories,
                                                             std::size_t rareCategories,
                                                             std::size_t deadCategories) {

    // Categorization status is "warn" if:

    // - At least 100 messages have been categorized
    if (categorizedMessages <= 100) {
        return model_t::E_CategorizationStatusOk;
    }

    // and one of the following holds:

    // - There is only 1 category
    if (totalCategories == 1) {
        return model_t::E_CategorizationStatusWarn;
    }

    // - More than 90% of categories are rare
    if (10 * rareCategories > 9 * totalCategories) {
        return model_t::E_CategorizationStatusWarn;
    }

    // - The number of categories is greater than 50% of the number of categorized messages
    if (2 * totalCategories > categorizedMessages) {
        return model_t::E_CategorizationStatusWarn;
    }

    // - There are no frequent match categories
    if (frequentCategories == 0) {
        return model_t::E_CategorizationStatusWarn;
    }

    // - More than 50% of categories are dead
    if (2 * deadCategories > totalCategories) {
        return model_t::E_CategorizationStatusWarn;
    }

    return model_t::E_CategorizationStatusOk;
}

std::size_t CTokenListDataCategorizerBase::numMatches(int categoryId) {
    if (categoryId < 1 || static_cast<std::size_t>(categoryId) > m_Categories.size()) {
        LOG_ERROR(<< "Programmatic error - invalid ML category: " << categoryId);
        return 0;
    }
    return m_Categories[categoryId - 1].numMatches();
}

CDataCategorizer::TIntVec CTokenListDataCategorizerBase::usurpedCategories(int categoryId) {
    CDataCategorizer::TIntVec usurped;
    if (categoryId < 1 || static_cast<std::size_t>(categoryId) > m_Categories.size()) {
        LOG_ERROR(<< "Programmatic error - invalid ML category: " << categoryId);
        return usurped;
    }
    auto iter = std::find_if(m_CategoriesByCount.begin(), m_CategoriesByCount.end(),
                             [categoryId](const TSizeSizePr& pr) {
                                 return pr.second == static_cast<std::size_t>(categoryId);
                             });
    if (iter == m_CategoriesByCount.end()) {
        LOG_WARN(<< "Could not find category definition for category: " << categoryId);
        return usurped;
    }
    ++iter;
    const CTokenListCategory& category{m_Categories[categoryId - 1]};
    for (; iter != m_CategoriesByCount.end(); ++iter) {
        const CTokenListCategory& lessFrequentCategory{
            m_Categories[static_cast<int>(iter->second) - 1]};
        bool matchesSearch{category.maxMatchingStringLen() >=
                               lessFrequentCategory.maxMatchingStringLen() &&
                           category.isMissingCommonTokenWeightZero(
                               lessFrequentCategory.commonUniqueTokenIds()) &&
                           category.containsCommonInOrderTokensInOrder(
                               lessFrequentCategory.baseTokenIds())};
        if (matchesSearch) {
            usurped.emplace_back(static_cast<int>(iter->second));
        }
    }
    return usurped;
}

std::size_t CTokenListDataCategorizerBase::numCategories() const {
    return m_Categories.size();
}

bool CTokenListDataCategorizerBase::categoryChangedAndReset(int categoryId) {
    if (categoryId < 1 || static_cast<std::size_t>(categoryId) > m_Categories.size()) {
        LOG_ERROR(<< "Programmatic error - invalid ML category: " << categoryId);
        return false;
    }
    return m_Categories[categoryId - 1].isChangedAndReset();
}

CTokenListDataCategorizerBase::CTokenInfoItem::CTokenInfoItem(const std::string& str,
                                                              std::size_t index)
    : m_Str{str}, m_Index{index}, m_CategoryCount{0} {
}

const std::string& CTokenListDataCategorizerBase::CTokenInfoItem::str() const {
    return m_Str;
}

void CTokenListDataCategorizerBase::CTokenInfoItem::debugMemoryUsage(
    const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CTokenInfoItem");
    core::CMemoryDebug::dynamicSize("m_Str", m_Str, mem);
}

std::size_t CTokenListDataCategorizerBase::CTokenInfoItem::memoryUsage() const {
    return core::CMemory::dynamicSize(m_Str);
}

std::size_t CTokenListDataCategorizerBase::CTokenInfoItem::index() const {
    return m_Index;
}

std::size_t CTokenListDataCategorizerBase::CTokenInfoItem::categoryCount() const {
    return m_CategoryCount;
}

void CTokenListDataCategorizerBase::CTokenInfoItem::categoryCount(std::size_t categoryCount) {
    m_CategoryCount = categoryCount;
}

void CTokenListDataCategorizerBase::CTokenInfoItem::incCategoryCount() {
    ++m_CategoryCount;
}

CTokenListDataCategorizerBase::CSizePairFirstElementEquals::CSizePairFirstElementEquals(std::size_t value)
    : m_Value(value) {
}

CTokenListDataCategorizerBase::SIdTranslater::SIdTranslater(const CTokenListDataCategorizerBase& categorizer,
                                                            const TSizeSizePrVec& tokenIds,
                                                            char separator)
    : s_Categorizer{categorizer}, s_TokenIds{tokenIds}, s_Separator{separator} {
}

std::ostream& operator<<(std::ostream& strm,
                         const CTokenListDataCategorizerBase::SIdTranslater& translator) {
    for (auto iter = translator.s_TokenIds.begin();
         iter != translator.s_TokenIds.end(); ++iter) {
        if (iter != translator.s_TokenIds.begin()) {
            strm << translator.s_Separator;
        }

        if (iter->first < translator.s_Categorizer.m_TokenIdLookup.size()) {
            strm << translator.s_Categorizer.m_TokenIdLookup[iter->first].str();
        } else {
            strm << "Out of bounds!";
        }
    }

    return strm;
}
}
}
