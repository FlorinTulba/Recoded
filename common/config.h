/*
Allows configuring several C++ 'Recoded' projects
based on a text configuration file specific to each such project.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_CONFIG
#define H_CONFIG

#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <set>
#include <map>

/// Default validator accepting any value for ConfigItem from below
template<typename T>
struct ValValidator {
	/// Overriding methods will throw invalid_argument if the value doesn't respect specific constraints
	virtual void check(const T &/*val*/) const {}

	virtual ~ValValidator() {}
};

/// Interface for each configuration item
struct IConfigItem abstract {
	virtual const std::string& name() const = 0;///< name of the property
	virtual std::string toString() const = 0;	///< presents the value of the property as a string

	/// Reads the value of the property from is
	/// If reading is successful, it returns true
	/// It could validate the read value with ValValidator::check(value)
	virtual bool read(std::istream &is) = 0;

	virtual ~IConfigItem() = 0 {}
};

/// Checks a value to be within a given range
template<typename T>
class InRange : public ValValidator<T> {
protected:
	T minVal;	///< first acceptable value
	T maxVal;	///< last acceptable value

public:
	InRange(const T &minVal_, const T &maxVal_) :
			minVal(minVal_), maxVal(maxVal_) {
		if(minVal > maxVal) {
			std::ostringstream oss;
			oss<<__FUNCTION__ " expects minVal_ <= maxVal_, but received minVal_="
				<<minVal_<<" and maxVal_="<<maxVal_;
			std::cerr<<oss.str()<<std::endl;
			throw std::invalid_argument(oss.str());
		}
	}

	/// Throws if the value is outside the accepted range
	void check(const T &val) const override {
		if(val >= minVal && val <= maxVal)
			return;

		std::ostringstream oss;
		oss<<__FUNCTION__ " expects a value between ["
			<<minVal<<", "<<maxVal<<"], but received "<<val;
		std::cerr<<oss.str()<<std::endl;
		throw std::invalid_argument(oss.str());
	}
};

/// Checks a value to be within a given set of values
template<typename T>
class AmongSet : public ValValidator<T> {
protected:
	std::set<T> possibleVals;	///< the set of accepted values

public:
	AmongSet(const std::set<T> &possibleVals_) :
			possibleVals(possibleVals_) {
		if(possibleVals.empty()) {
			static const std::string err(__FUNCTION__ " expects non-empty set of possibleVals_");
			std::cerr<<err<<std::endl;
			throw std::invalid_argument(err);
		}
	}

	/// Throws if the value is outside the set of values
	void check(const T &val) const override {
		if(cend(possibleVals) != possibleVals.find(val))
			return;

		std::ostringstream oss;
		oss<<__FUNCTION__ " expects a value among { ";
		std::copy(std::cbegin(possibleVals), std::cend(possibleVals), std::ostream_iterator<T>(oss, " "));
		oss<<"}, but received "<<val;
		std::cerr<<oss.str()<<std::endl;
		throw std::invalid_argument(oss.str());
	}
};

/// Realization of IConfigItem
template<typename T>
class ConfigItem : public IConfigItem {
	const std::string _name;	///< property name

	/// Validates val from below
	const std::unique_ptr<ValValidator<T>> validator = std::make_unique<ValValidator<T>>();

	T val;	///< property value

public:
	typedef T type;	///< property type

	/// Set only the name
	ConfigItem(const std::string &name) :
		_name(name) {}

	/// Set name and default value. Throws if provided initVal is not correct
	ConfigItem(const std::string &name, const T &initVal,
			   std::unique_ptr<ValValidator<T>> _validator = nullptr) :
			_name(name),
			validator((bool)_validator ? std::move(_validator) : std::make_unique<ValValidator<T>>()),
			val(initVal) {
		validator->check(val);
	}

	ConfigItem(const ConfigItem&) = delete;
	ConfigItem(ConfigItem&&) = delete;
	void operator=(const ConfigItem&) = delete;
	void operator=(ConfigItem&&) = delete;

	const std::string& name() const override { return _name; }

	/**
	It expects the value right at the beginning of is.
	Boolean inputs need to be exactly 'true' or 'false'.

	It checks the value using the validator.

	For reading fancier values (like XML for instance),
	derive this type or IConfigItem and override this method accordingly.
	*/
	bool read(std::istream &is) override {
		const bool result = (bool)(is>>std::boolalpha>>val);
		validator->check(val);
		return result;
	}

	std::string toString() const override {
		std::ostringstream oss;
		oss<<std::boolalpha<<val;
		return oss.str();
	}

	/**
	@return a const reference to the value.
	The reference is useful particularly for Config::valueOf, which returns reference
	*/
	inline const T& value() const { return val; }
};

/// Allows customizing which file to use for the configuration. Define this function in your project
extern const std::string& configFile();

/// Allows configuring the project based on the properties from `config.txt`
class Config {
protected:
	/**
	Managed properties, specific to each project using this header.

	Define it like this in the project:
	
		const map<string, const unique_ptr<IConfigItem>>& Config::props() {
			static map<string, const unique_ptr<IConfigItem>> pairs;

		#define addProp(propName, propType, propDefVal, validator) \
			pairs.emplace(#propName, make_unique<ConfigItem<propType>>(#propName, propDefVal, validator))

			// Add here all configuration items
			addProp(nameOfBoolProp, bool, false, nullptr);
			addProp(nameOfIntProp, int, 15, std::make_unique<InRange<int>>(-4, 50));
			addProp(nameOfStringProp, string, "abc",
				std::make_unique<AmongSet<string>>(std::set<string> { "abc", "xyz", "mnop" }));

		#undef addProp

			return pairs;
		}
	*/
	static const std::map<std::string, const std::unique_ptr<IConfigItem>>& props();

	/**
	Reads the settings from configFile() and reports them to the console.
	The properties missing from the configuration file keep their default values.
	Throws exception when it cannot read a certain property.
	When the file doesn't exist, just shows the default properties.

	This class expects a configuration file like:

			#comment about nameOfBoolProp
			nameOfBoolProp false

			nameOfIntProp 15
			#comment about nameOfBoolProp
			nameOfStringProp abc

	For fancier input files (like XML-s) use specialized libraries or derive from this class.
	*/
	Config();

	Config(const Config&) = delete;
	void operator=(const Config&) = delete;

public:
	/// @return the singleton
	static const Config &get() {
		static Config instance;
		return instance;
	}

	/**
	Looks for prop.name() among props(). If prop isn't found, it returns the fallback value

	Example: 
	const bool valYesNoProp = Config::get().valueOf(ConfigItem<bool>("YesNoProp"));
	
	@param prop a ConfigItem<T>, whose name should be the sought property
	@param fallbackVal the value to be returned when the property isn't found

	@return a reference to the stored value corresponding to the located prop.
		When prop isn't found, it returns fallbackVal.
	*/
	template<class Property>
	typename const Property::type& valueOf(const Property &prop,
										   typename const Property::type &fallbackVal) const {
		const auto itAssoc = Config::props().find(prop.name());
		if(itAssoc == std::cend(Config::props())) {
			std::cerr<<"Unrecognized requested property: "<<prop.name()<<std::endl;
			std::cerr<<"Using the fallback value: "<<fallbackVal<<std::endl;
			return fallbackVal;
		}

		// itAssoc->second.get() has IConfigItem* type, which doesn't provide the required value() method
		const Property *actualProp = dynamic_cast<Property*>(itAssoc->second.get());
		if(actualProp == nullptr)
			return fallbackVal;

		return actualProp->value();
	}
};

#endif // H_CONFIG
