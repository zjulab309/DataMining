import os
from os import listdir
from os.path import isfile, join, split

# we output files as CSVs which is appropriate
import unicodecsv as csv

################################################################################
#
# Simple method to open a unicode CSV file.  If column name are provided the
# the method also writes them out as the first line of the CSV
#
################################################################################

def open_csv_file(name,column_names = None):
    output_file = open(name,"wb")
    output_csv_file = csv.writer(output_file,quoting = csv.QUOTE_MINIMAL)

    if column_names is not None:
        output_csv_file.writerow(column_names)

    return output_csv_file





################################################################################
#
# Read the format csv file ,and return the data in it
#
################################################################################
def read_csv(name):
    rows = []
    with open(name,'rb') as csvfile:
        reader = csv.reader(csvfile,delimiter = ',', quotechar = '"')
        for row in reader:
            row[1] = row[1].replace(u'\xa0',u' ')
            rows.extend([row])

    return rows
