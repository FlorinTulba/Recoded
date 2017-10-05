/*
Allows configuring several C++ 'Recoded' projects
based on a text configuration file specific to each such project.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "config.h"
#include "util.h"

#include <fstream>
#include <set>
#include <cassert>

using namespace std;

Config::Config() {
	set<string> defaultConfigItems;
	for(const auto &assoc : Config::props())
		defaultConfigItems.insert(assoc.first);

	ifstream ifs(configFile());
	if(!ifs)
		cout<<"Couldn't find `"<<configFile()<<"`!"<<endl;
	
	cout<<"Using configuration:"<<endl;

	if((bool)ifs) {
		string line, prop, err;
		while(nextRelevantLine(ifs, line)) {
			istringstream iss(line);
			if((iss>>prop).fail()) {
				err = "Couldn't read the name of the first / next property!";
				break;
			}

			cout<<prop<<" = ";

			const auto itAssoc = Config::props().find(prop);
			if(itAssoc == cend(Config::props())) {
				err = "Unrecognized property name!";
				break;
			}

			IConfigItem &propVal = *itAssoc->second;
			if(propVal.read(iss)) {
				cout<<propVal.toString()<<endl;

			} else {
				err = trim(line.substr(prop.size())) + " invalid value";
				break;
			}

			defaultConfigItems.erase(prop);
		}

		if(!err.empty()) {
			cerr<<err<<endl;
			throw runtime_error(err);
		}
	}

	for(const string &defaulted : defaultConfigItems) {
		cout<<defaulted<<" = ";
		const auto itAssoc = Config::props().find(defaulted);
		assert(itAssoc != cend(Config::props()));
		cout<<itAssoc->second->toString()<<endl;
	}
	cout<<endl;
}
