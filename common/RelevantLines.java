/*
	Basic test files parser needed by several 'Recoded' projects

	@2017 Florin Tulba (florintulba@yahoo.com)
*/

import java.io.*;

/**
 * Parses a text file and provides non-empty lines that are neither comments lines. Comments start with '#'
 */
public class RelevantLines {
	protected BufferedReader br;		// reader for the file

	/*
	 * Prepares for parsing 'fileName'.
	 *
	 * @throw IOException when unable to open the provided file
	 */
	public RelevantLines(String fileName) throws IOException {
		br = new BufferedReader(new FileReader(fileName));
	}

	/*
	 * Returns next non-empty and non-comment line if any. It will return null otherwise.
	 *
	 * @throw IOException when unable to read from the BufferedReader
	 */
	public String nextLine() throws IOException {
    	String line = null;
        while((line = br.readLine()) != null) {
        	line = line.trim();
            if(line.length() > 0 && line.charAt(0) != '#')
            	break;
        }
        return line;
	}
}