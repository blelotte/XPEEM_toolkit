args = getArgument();
argArray = split(args, ",");
//open a dialogue to select directory of images to open and assign it to a variable (eg input) 
input = argArray[0];
//open a dialogue to select location where images/results are to be stored 
output =  argArray[1];
// Print input paths (for debugging purpose)
// print("Arguments:");
// print(input);
// print(output);
//get list of files in the folder and assigns it to a variable array. 
list = getFileList(input); 
setBatchMode(true);
//indicate parameters to measure
//run("Set Measurements...", "area mean integrated display redirect=None decimal=2");

 	for (i = 0; i<list.length; i++) {
 	   	open(input+list[i]);

		run("Duplicate...", " ");
		run("Fire");
		//run("Brightness/Contrast...");
		//run("Enhance Contrast", "saturated=0.35");
		run("Set Scale...", "distance=20.560 known=1 unit=um");
		run("Scale Bar...", "width=5 height=2 thickness=10 font=32 location=[Lower Left] bold");
		run("RGB Color");
		saveAs(output+i+"-"+list[i]);
		run("Close All");

	}


run("Quit");
