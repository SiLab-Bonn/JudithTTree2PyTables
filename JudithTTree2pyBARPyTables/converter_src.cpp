// This file provides a function to read a Root TTree for a fast conversion to PyTables

#pragma once

#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TDirectory.h>
#include <Rtypes.h>


#pragma pack(push, 1)  // Make sure byte alignment corresponds to numpy byte alignment

// define struct that corresponds to a table row
struct data_row {
	int64_t event_number;
	uint8_t frame;
	uint16_t column;
	uint16_t row;
	uint16_t charge;
	uint16_t tdc_value;
	uint16_t tdc_time_stamp;
	uint8_t trigger_status;
	uint16_t event_status;
	int64_t pybar_event_number;
	// constructor
	data_row(int64_t event_number, uint8_t frame, uint16_t column, uint16_t row, uint16_t charge,
		     uint16_t tdc_value, uint16_t tdc_time_stamp, uint8_t trigger_status, uint16_t event_status, int64_t pybar_event_number) :
			 event_number(event_number), frame(frame), column(column), row(row), charge(charge),
			 tdc_value(tdc_value), tdc_time_stamp(tdc_time_stamp), trigger_status(trigger_status), event_status(event_status), pybar_event_number(pybar_event_number) {}
};

#pragma pack(pop)  // pop needed to suppress VS C4103 compiler warning

void read_tree(const char * input_file, const char * plane, std::vector<data_row>& data) {
	static const Int_t MAX_HITS = 4000;
	// open ROOT file
	TFile* root_file = (TFile*) new TFile(input_file, "READ");
	TTree* event = (TTree*) root_file->Get("Event");
	TDirectory* dir = (TDirectory*) root_file->GetDirectory(plane);
	if (dir != 0) { // make sure selected plane exists
		// open Hits TTree
		TTree* hits = (TTree*) root_file->Get((std::string(plane) + std::string("/Hits")).c_str());

		// prepare Events branches for reading
		ULong64_t time_stamp;
		ULong64_t frame_number; // pyBAR event number
		Int_t trigger_offset;
		Int_t trigger_info;
		Bool_t invalid;

		// set addresses of Event arrays
		event->SetBranchAddress("TimeStamp", &time_stamp);
		event->SetBranchAddress("FrameNumber", &frame_number);
		event->SetBranchAddress("TriggerOffset", &trigger_offset);
		event->SetBranchAddress("TriggerInfo", &trigger_info);
		event->SetBranchAddress("Invalid", &invalid);

		// prepare Hits branches for reading
		Int_t n_hits;
		Int_t pix_x[MAX_HITS];
		Int_t pix_y[MAX_HITS];
		Int_t value[MAX_HITS];
		Int_t timing[MAX_HITS];
		UShort_t tdc_value[MAX_HITS];
		UShort_t tdc_time_stamp[MAX_HITS];
		UChar_t trigger_status[MAX_HITS];
		UShort_t event_status[MAX_HITS];

		// set addresses of Hits arrays
		hits->SetBranchAddress("NHits", &n_hits);
		hits->SetBranchAddress("PixX", pix_x);
		hits->SetBranchAddress("PixY", pix_y);
		hits->SetBranchAddress("Value", value);
		hits->SetBranchAddress("Timing", timing);
		hits->SetBranchAddress("Tdc", tdc_value);
		hits->SetBranchAddress("TdcTs", tdc_time_stamp);
		hits->SetBranchAddress("TriggerStatus", trigger_status);
		hits->SetBranchAddress("EventStatus", event_status);

		Long64_t n_entries = hits->GetEntriesFast();
		for (Long64_t i = 0; i < n_entries; i++) {
			// get values of selected branches
			event->GetEntry(i);
			hits->GetEntry(i);
			if (n_hits == 0)
				continue;

			// create new row for each hit in one event
			for (Int_t iHit = 0; iHit < n_hits; iHit++) {
				data.push_back(data_row(i, timing[iHit], pix_x[iHit], pix_y[iHit], value[iHit],
					                    tdc_value[iHit], tdc_time_stamp[iHit], trigger_status[iHit], event_status[iHit], frame_number));
			}
		}
	}
	else
		throw std::invalid_argument("Invalid plane.");

	root_file->Close();
	delete root_file;
}
