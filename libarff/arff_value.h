#ifndef __INCLUDED_ARFF_VALUE_H__
#define __INCLUDED_ARFF_VALUE_H__
/**
 * @file arff_value.h
 * @brief Contains 'ArffValue' class
 */

#include <string>
#include "arff_utils.h"


/**
 * @enum ArffValueEnum
 * @brief Types of data
 */
enum ArffValueEnum {
    /** integer */
    INTEGER = 0,
    /** float */
    FLOAT,
    /** date */
    DATE,
    /** string */
    STRING,
    /** numeric (only used by ArffAttr) */
    NUMERIC,
    /** nominal (only used by ArffAttr) */
    NOMINAL,
    /** unknown */
    UNKNOWN_VAL,
};

/**
 * @brief Converts the enum value to string
 * @param e the enum
 * @return desired string
 */
std::string arff_value2str(ArffValueEnum e);


/**
 * @class ArffValue arff_value.h
 * @brief Class to store int/float/string data-types
 */
class ArffValue {
public:
    /**
     * @brief Constructor for int data
     * @param i int value [0]
     */
    ArffValue(int32 i=0);

    /**
     * @brief Constructor for float data
     * @param f float value
     */
    ArffValue(float f);

    /**
     * @brief Constructor for numeric data
     * @param str string value
     *
     * Note that if the type is 'STRING', then the function will try to
     * convert the given string to float. If the conversion fails then
     * only the string will kept as it is.
     */
    ArffValue(const std::string& str);

    /**
     * @brief Constructor for string/nominal data
     * @param str string value
     */
    ArffValue(const std::string& str, ArffValueEnum aType);

    /**
     * @brief Copy constructor
     * @param src source from which to copy
     */
    ArffValue(const ArffValue& src);

    /**
     * @brief Missing value constructor
     * @param type data-type
     */
    ArffValue(ArffValueEnum type);

    /**
     * @brief Destructor
     */
    ~ArffValue();

    /**
     * @brief Set the data to be stored
     * @param i integer value
     */
    void set(int32 i);

    /**
     * @brief Set the data to be stored
     * @param f float value
     */
    void set(float f);

    /**
     * @brief Set the data to be stored
     * @param str string data
     * @param e whether you want 'DATE' or 'STRING' type [STRING]
     */
    void set(const std::string& str, ArffValueEnum e=STRING);

    /**
     * @brief Whether the value is missing or not
     * @return true if missing, else false
     */
    bool missing() const;

    /**
     * @brief Data type stored
     * @return type
     */
    ArffValueEnum type() const;

    /**
     * @brief Get the integer value
     * @return int value
     */
    operator int32() const;

    /**
     * @brief Get the float value
     * @return float value
     */
    operator float() const;

    /**
     * @brief Get the string value
     * @return string value
     */
    operator std::string() const;

    /**
     * @brief Equality operator
     * @param right RHS
     */
    bool operator ==(const ArffValue& right) const;

    /**
     * @brief Equality operator
     * @param right RHS
     */
    bool operator ==(int32 right) const;

    /**
     * @brief Equality operator
     * @param right RHS
     */
    bool operator ==(float right) const;

    /**
     * @brief Equality operator
     * @param right RHS
     */
    bool operator ==(const std::string& right) const;

    /**
     * @brief LHS equality operator
     * @param left LHS
     * @param right RHS
     */
    friend bool operator ==(int32 left, const ArffValue& right);

    /**
     * @brief LHS equality operator
     * @param left LHS
     * @param right RHS
     */
    friend bool operator ==(float left, const ArffValue& right);

    /**
     * @brief LHS equality operator
     * @param left LHS
     * @param right RHS
     */
    friend bool operator ==(const std::string& left, const ArffValue& right);


private:
    /** integer value */
    int32 m_int;
    /** float value */
    float m_float;
    /** string value */
    std::string m_str;
    /** data-type */
    ArffValueEnum m_type;
    /** value missing or not */
    bool m_missing;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_VALUE_H__
