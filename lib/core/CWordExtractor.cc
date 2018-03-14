/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include <core/CWordExtractor.h>

#include <core/CWordDictionary.h>

#include <ctype.h>

namespace ml {
namespace core {

const std::string CWordExtractor::PUNCT_CHARS("!\"'(),-./:;?[]`");

void CWordExtractor::extractWordsFromMessage(const std::string& message,
                                             std::string& messageWords) {
    CWordExtractor::extractWordsFromMessage(1, message, messageWords);
}

void CWordExtractor::extractWordsFromMessage(size_t minConsecutive,
                                             const std::string& message,
                                             std::string& messageWords) {
    // Words are taken to be sub-strings of 1 or more letters, all lower case
    // except possibly the first, preceded by a space, and followed by 0 or 1
    // punctuation characters and then a space (or the end of the string).
    // Additionally, words must be in the dictionary.  (This implies we'll need
    // to get foreign dictionaries if we get customers using the product in
    // languages other than English.)

    // The minConsecutive argument indicates the minimum number of words that
    // must occur consecutively in the string before any of them are recorded.

    messageWords.clear();

    size_t messageLen(message.length());
    size_t consecutive(0);
    size_t wordStartPos(0);
    size_t rollbackPos(0);
    size_t spaceCount(0);
    size_t punctCount(0);
    bool inWord(false);
    std::string curWord;
    const CWordDictionary& dict = CWordDictionary::instance();
    for (size_t messagePos = 0; messagePos < messageLen; ++messagePos) {
        char thisChar(message[messagePos]);
        bool rollback(false);

        if (::isspace(static_cast<unsigned char>(thisChar))) {
            if (inWord && punctCount <= 1) {
                if (dict.isInDictionary(curWord)) {
                    messageWords.append(message,
                                        wordStartPos,
                                        messagePos - spaceCount - punctCount - wordStartPos);
                    messageWords += ' ';

                    ++consecutive;
                    if (consecutive >= minConsecutive) {
                        rollbackPos = messageWords.length();
                    }
                } else {
                    rollback = true;
                }
            }

            ++spaceCount;
            punctCount = 0;
            inWord = false;
        }
        // Not using ::ispunct() here, as its definition of punctuation is too
        // permissive (basically anything that's not a letter, number or space)
        else if (PUNCT_CHARS.find(thisChar) != std::string::npos) {
            ++punctCount;
            if (punctCount > 1) {
                rollback = true;
            }
        } else if (::isalpha(static_cast<unsigned char>(thisChar))) {
            if (punctCount == 0) {
                if (inWord) {
                    if (::isupper(static_cast<unsigned char>(thisChar))) {
                        inWord = false;
                        rollback = true;
                    } else {
                        curWord += thisChar;
                    }
                } else {
                    if (spaceCount > 0) {
                        inWord = true;
                        wordStartPos = messagePos;
                        curWord = thisChar;
                    }
                }
            } else {
                inWord = false;
                rollback = true;
            }

            spaceCount = 0;
            punctCount = 0;
        } else {
            spaceCount = 0;
            punctCount = 0;
            inWord = false;
            rollback = true;
        }

        if (rollback) {
            messageWords.erase(rollbackPos);
            consecutive = 0;
        }
    }

    if (inWord && punctCount <= 1 && dict.isInDictionary(curWord)) {
        ++consecutive;
        if (consecutive >= minConsecutive) {
            messageWords.append(message,
                                wordStartPos,
                                message.length() - wordStartPos - punctCount);
            messageWords += ' ';

            rollbackPos = messageWords.length();
        }
    }

    if (rollbackPos == 0) {
        messageWords.clear();
    } else {
        // Subtract 1 to strip the last space (since the above code always
        // appends a trailing space after each word)
        messageWords.erase(rollbackPos - 1);
    }
}
}
}
