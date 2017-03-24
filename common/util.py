#! python3

#
# Various utilities shared by several 'Recoded' projects in Python
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#


def nextRelevantLine(f):
	'''
	Reads from a file the next non-empty and non-comment line. Comments start with "#".
	Returns None when reaching EOF.
	'''
	line = f.readline()
	while line:
		if not line[0] in "\r\n#":	# Ignore empty lines or comments (lines starting with '#')
			return line
		line = f.readline()

	return None
