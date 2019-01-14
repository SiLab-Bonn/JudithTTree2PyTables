import logging
import os.path

import tables as tb

from converter import read_from_root_tree


# import this function for use in scripts
def convert_tree(input_filename, plane, output_filename=None, chunksize=1000000):
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
    chunk_size : int
        Chunk size of the data when reading from ROOT file.
        If chunksize is 0 or None, read all data.

    Returns
    -------
    output_filename : string
        Filename of the output PyTables file.
    '''
    # set output filename
    if output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + '_' + plane + '.h5'

    # create pyTables file
    with tb.open_file(filename=output_filename, mode='w') as out_file_h5:
        try:  # Check if table already exists, then append data
            out_file_h5.remove_node(out_file_h5.root, name='Hits')
            logging.info('Overwriting existing Hits')
        except tb.NodeError:  # Hits table does not exist, thus create new
            pass
        out_table = None

        start = 0
        while True:
            # get data from ROOT file
            data = read_from_root_tree(
                input_filename=input_filename,
                plane=plane,
                start=start,
                stop=start + chunksize)

            # check if any data is returned
            if data is None:
                break

            if out_table is None:
                out_table = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name='Hits',
                    description=data.dtype,
                    title='Converted data from Judith ROOT TTree',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

            # keep standard format for hit table
            data['column'][:] += 1
            data['row'][:] += 1
            out_table.append(data)
            if chunksize is None or chunksize == 0:
                break
            else:
                start += chunksize

    return output_filename
