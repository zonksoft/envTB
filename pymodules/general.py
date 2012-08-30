def read_file_as_table(filename):
    """
    Read in file as table, with lines as the first list index 
    and columns (separated by whitespaces) as the second.
    """
    filetoread = open(filename, 'r')
    data = [line.split() for line in filetoread.readlines()]
    filetoread.close()
    return data
