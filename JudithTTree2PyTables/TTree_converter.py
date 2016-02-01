'This file implements the only function that needs to be called and example for the TTRee2pyTables converter'

import tables as tb
import os.path

from ROOT import TFile

from JudithTTree2PyTables.converter import read_from_root_tree


# import this function for use in scripts
def convert_tree(tree_file, plane_list, output_folder=None):
    '''
    Convert hit information from ROOT file to one pyTables file for each plane.

    Parameters
    ----------
    tree_file: string
        File path and name to ROOT file
    output_folder: string
        Folder where to store converted pyTables files
    plane_list: list of int
        Index of plane that shall be converted
    '''

    # set output file name
    if not output_folder:
        output_folder = os.path.dirname(tree_file)
    file_name = os.path.splitext(os.path.split(tree_file)[1])[0]
    file_root = os.path.join(output_folder, file_name)

    for plane_number in plane_list:

        # get data from ROOT file
        data = read_from_root_tree(tree_file, plane_number)

        # check if any data is returned (None when selected plane does not exist)
        if data is None:
            continue
        else:
            # keep standard format for hit table
            data['column'][:] += 1
            data['row'][:] += 1

            # create pyTables file
            with tb.open_file(file_root + '_plane' + str(plane_number) + '.h5', 'w') as out_file_h5:
                out_file_h5 = out_file_h5.createTable(out_file_h5.root, name='Hits', description=data.dtype, title='Converted data from ROOT file', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_file_h5.append(data)


# Standalone converter
if __name__ == "__main__":
    # select planes that will be converted and set input file and output folder
    plane_list = (0, 1, 3)
    tree_file = r'../example_files/example.root'
    output_folder = r'../example_files/'

    convert_tree(tree_file, plane_list, output_folder)
