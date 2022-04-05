#include <vector>

#include "./PrintableObject.h"

/*** FOR TESTING THE FILEMANAGER ***/
namespace FM
{
    class PrintableInt : public PrintableObject
    {
        public:
            PrintableInt(int value) : _value(value) {}
            std::string getPrintString() { return std::to_string(_value); }
        private:
            int _value;
    };
}