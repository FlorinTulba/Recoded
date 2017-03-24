#
# Various utilities shared by several 'Recoded' projects in R
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#

if( ! exists('util_R')) {
    util_R <- T

    # Function 'nextRelevantLine' returns next non-empty and non-comment line
    # from a given input stream - 'conn' parameter (usually pointing a file)
    # The comments in the stream are lines starting with '#'.
    # When reaching EOF, the 'length()' of the result is 0. Otherwise, it will be 1.
    # So, 'length()' establishes the validity of the return value,
    # while 'nchar()' provides the number of characters of a valid line.
    nextRelevantLine <- function(conn) {
        while(length(line <- readLines(conn, n = 1L, warn = FALSE))) {
            if(nchar(line) > 0 && substring(line, 1, 1) != '#')
                break
        }
        line
    }
}
