import os.path

import tables as tb

from converter import read_from_root_tree


# import this function for use in scripts
def convert_tree(input_file, planes, output_folder=None):
    '''
    Convert hit information from ROOT file to one pyTables file for each plane.

    Parameters
    ----------
    input_file: string
        Fileanme of the input ROOT file.
    planes: list of strings
        Index of plane that shall be converted
    output_folder: string
        Folder where to store converted pyTables files.

    '''
    # set output file name
    if output_folder is None:
        output_folder = os.path.dirname(input_file)
    file_name = os.path.splitext(os.path.split(input_file)[1])[0]
    file_root = os.path.join(output_folder, file_name)

    for plane in planes:
        # get data from ROOT file
        data = read_from_root_tree(input_file, plane)

        # check if any data is returned (None when selected plane does not exist)
        if data is None:
            continue
        else:
            # keep standard format for hit table
            data['column'][:] += 1
            data['row'][:] += 1

            # create pyTables file
            with tb.open_file(filename=file_root + '_' + plane + '.h5', mode='w') as out_file_h5:
                out_file_h5 = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name='Hits',
                    description=data.dtype,
                    title='Converted data from Judith ROOT TTree',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_file_h5.append(data)
