from JudithTTree2pyBARPyTables.JudithTTree2pyBARPyTables import convert_tree

tree_file = r'example.root'  # Input file
output_folder = r'./'  # Set output folder for created files
plane_list = ("Plane0", "Plane1", "Plane2")  # Specify plane numbers to convert

# Converts single planes from ROOT file to DUT#-converted.h5
convert_tree(tree_file, plane_list, output_folder)
