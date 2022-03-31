#ifndef PRINTABLE_OBJECT
#define PRINTABLE_OBJECT

#include <string>

namespace LS 
{
    class PrintableObject
    {
        public:
            virtual std::string getPrintString() = 0;
    };
}

#endif