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
#include <memory>
#include <map>

/// Interface for each configuration item
struct IConfigItem abstract {
	virtual const std::string& name() const = 0;///< name of the property
	virtual std::string toString() const = 0;	///< presents the value of the property as a string

	/// Reads the value of the property from is; if reading is successful, it returns true
	virtual bool read(std::istream &is) = 0;

	virtual ~IConfigItem() = 0 {}
};

/// Realization of IConfigItem
template<typename T>
class ConfigItem : public IConfigItem {
	const std::string _name;	///< property name
	T val;						///< property value

public:
	typedef T type;				///< property type

	/// Set only the name
	ConfigItem(const std::string &name) :
		_name(name) {}

	/// Set name and default value
	ConfigItem(const std::string &name, const T &initVal) :
		_name(name), val(initVal) {}

	const std::string& name() const override { return _name; }

	/*
	It expects the value right at the beginning of is.
	Boolean inputs need to be exactly 'true' or 'false'.

	For reading fancier values (like XML for instance),
	derive this type or IConfigItem and override this method accordingly.
	*/
	bool read(std::istream &is) override { return (bool)(is>>std::boolalpha>>val); }

	std::string toString() const override {
		std::ostringstream oss;
		oss<<std::boolalpha<<val;
		return oss.str();
	}

	/**
	@return a const reference to the value.
	The reference is useful particularly for Config::valueOf, which returns address (pointer)
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

		#define addProp(propName, propType, propDefVal) \
			pairs.emplace(#propName, make_unique<ConfigItem<propType>>(#propName, propDefVal))

			// Add here all configuration items
			addProp(nameOfBoolProp, bool, false);
			addProp(nameOfIntProp, int, 15);
			addProp(nameOfStringProp, string, "abc");

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
	Looks for prop.name() among props().

	Example: 
	const bool *valYesNoProp = Config::get().valueOf(ConfigItem<bool>("YesNoProp"));
	
	@param prop a ConfigItem<T>, whose name should be the sought property

	@return the address of the stored value corresponding to the located prop.
		When prop isn't found, it returns nullptr.
	*/
	template<class Property>
	typename const Property::type* valueOf(const Property &prop) const {
		const auto itAssoc = Config::props().find(prop.name());
		if(itAssoc == std::cend(Config::props()))
			return nullptr;

		// itAssoc->second.get() has IConfigItem* type, which doesn't provide the required value() method
		const Property *actualProp = dynamic_cast<Property*>(itAssoc->second.get());
		if(actualProp == nullptr)
			return nullptr;

		return &actualProp->value();
	}
};

#endif // H_CONFIG
