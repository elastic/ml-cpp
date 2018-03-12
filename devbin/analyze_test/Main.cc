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

#include <iostream>
#include <vector>

template <typename T>
struct STemplated {
    void printFirst(void) {
        std::cout << s_First << std::endl;
    }

    T              s_First;
    std::vector<T> s_Second;
};

struct SSimple {
    void printFirst(void) {
        std::cout << s_First << std::endl;
    }

    int              s_First;
    std::vector<int> s_Second;
};

int main(int, char **) {
    {
        SSimple obj;
        obj.printFirst();
    }

    {
        STemplated<int> obj;
        obj.printFirst();
    }

    return 0;
}

