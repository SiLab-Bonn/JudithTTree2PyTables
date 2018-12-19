import os.path

import tables as tb

from converter import read_from_root_tree


# import this function for use in scripts
def convert_tree(input_filename, plane, output_filename=None):
    '''
    Convert hit information in a ROOT TTree to PyTables table.

    Parameters
    ----------
    input_filename : string
        Fileanme of the input ROOT file.
    planes : string
        Name of the TTree, e.g. "Plane0".
    output_filename : string
        Filename of the output PyTables file.

    Returns
    -------
    output_filename : string
        Filename of the output PyTables file.
    '''
    # set output filename
    if output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + '_' + plane + '.h5'

    # get data from ROOT file
    data = read_from_root_tree(input_filename, plane)

    # check if any data is returned (None when selected plane does not exist)
    if data is not None:
        # keep standard format for hit table
        data['column'][:] += 1
        data['row'][:] += 1

        # create pyTables file
        with tb.open_file(filename=output_filename, mode='w') as out_file_h5:
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
    else:
        raise ValueError("invalid plane %s" % plane)

    return output_filename
