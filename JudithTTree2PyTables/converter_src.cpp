// This file provides a function to read a Root TTree for a fast conversion to pyTables

#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <inttypes.h>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TDirectory.h>
#include <Rtypes.h>


std::string IntToStr(unsigned int const& pValue) {
	std::stringstream tStream;
	tStream << pValue;
	return tStream.str();
}

#pragma pack(push, 1)  // Make sure byte alignment corresponds to numpy byte alignment

// define struct that corresponds to a table row
struct data_row {
	int64_t event_number;
	uint8_t frame;
	uint16_t column;
	uint16_t row;
	uint16_t charge;
	// constructor
	data_row(int64_t event_number, uint8_t frame, uint16_t column, uint16_t row, uint16_t charge) :
			event_number(event_number), frame(frame), column(column), row(row), charge(charge) {}
};

#pragma pack(pop)  // pop needed to suppress VS C4103 compiler warning

void read_tree(const char *tree_file, const int& plane_number, std::vector<data_row>& data) {
	// open ROOT file
	TFile *root_file = TFile::Open(tree_file);
	std::string folder = std::string("Plane") + IntToStr(plane_number);
	std::string plane_name = folder + std::string("/Hits");

	if (root_file->GetDirectory(folder.c_str())) {  // make sure selected plane exists
		TTree *tree = (TTree*) root_file->Get(plane_name.c_str());  // open tree

		int nentries = tree->GetEntries();

		// prepare branches for reading
		int nhits = 0;
		int pixx[5000];
		int pixy[5000];
		int value[5000];

		tree->SetBranchAddress("NHits", &nhits);
		tree->SetBranchAddress("PixX", pixx);
		tree->SetBranchAddress("PixY", pixy);
		tree->SetBranchAddress("Value", value);

		for (int i = 0; i < nentries; ++i) {
			// get values of selected branches
			tree->GetEntry(i);
			if (nhits == 0)
				continue;

			// create new row for each hit in one event
			for (int iHit = 0; iHit < nhits; ++iHit) {
				data.push_back(data_row(i, 0, pixx[iHit], pixy[iHit], value[iHit]));
				// print actual 'event':
				// if (i > 1960000)
				// std::cout << "Event " << i << "\t" << 0 << "\t" << pixx[iHit] << "\t" << pixy[iHit] << "\t" << value[iHit] << "\n";
			}

		}
		std::cout << "Converted plane number " << plane_number << "\n";

	}
	else
		std::cout << "Chosen plane with number " << plane_number << " does not exist!" << "\n";

	delete root_file;
}
