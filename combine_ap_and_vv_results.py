import os
import json

def load_json(filepath):
	return json.load(open(filepath))

def save_json(data, filepath):
	json.dump(data, open(filepath, 'w'))

def main():

	print "loading input files"
	yes_no = load_json(vv_result_file)
	non_yes_no = load_json(ap_result_file)
	
	print "combining"
	combined = yes_no + non_yes_no

	print "saving combined file"
	save_json(combined, output_result_file)

if __name__ == "__main__":

	vv_result_file = 'predictions/vv_results.json'
	ap_result_file = 'predictions/ap_results.json'

	output_result_file = 'predictions/gvqa_results.json'

	main()