def read_file_as_table(filename):
    """
    Read in file as table, with lines as the first list index
    and columns (separated by whitespaces) as the second.
    """
    filetoread = open(filename, 'r')
    data = [line.split() for line in filetoread.readlines()]
    filetoread.close()
    return data


def split_by_empty_lines(data, ignorecomments=False):
    """
    Splits a list of lines by empty lines [].

    data: The list of lines, probably read from a file.
    ignorecomments: If True, all lines starting with a # will be removed.
    """

    blocks = []

    blank_line_found = True
    for line in data:
        if line != [] and blank_line_found is True:
            blank_line_found = False
            blocks.append([])
        if line == []:
            blank_line_found = True
        else:
            if line[0][0] != '#' or not ignorecomments:
                blocks[-1].append(line)

    return blocks
    
def string_to_number(s):
    """
    Convert string to number, if possible (float or integer).
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
