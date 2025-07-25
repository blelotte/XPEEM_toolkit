// Version 8
// Objective avoid edges at the border.
// Changed, first median and Gabor edge filter, then normalisation.
// Added an additional transformation argument

// >> LOAD ARGUMENTS
args = getArgument();
argArray = split(args, ",");
// 0 Edge target image path
targetpath = argArray[0];
// 1 Source folder
input = argArray[1];
// 2 Target folder
output =  argArray[2];
// 3 Direct transform folder
drtrsf=  argArray[3];
// 4 Indirect transform folder
indrtrsf=  argArray[4];
// 5 Parameters of the transform calculation
params =  split(argArray[5],";");
// 6 (Optional), path to the raw target.
if(params[7]==0 || params[7]==2){
	// the 7th argument is the raw target image path
	rawtargetpath = argArray[6];
} else if(params[7]==1) { 
	rawtargetpath = targetpath;
}

// >> LOAD TARGET IMAGE
//get list of files in the folder and assigns it to a variable array. 
list = getFileList(input);
setBatchMode(false);
scale = params[6];
k = 0;
suffix =  ".tif";
while (k<list.length) {
	// >> LOAD SOURCE IMAGE
	sourceImage=list[k];
	if (endsWith(sourceImage, suffix)){
		// >> PROCESS SOURCE IMAGE
		sourcepath=input+sourceImage;
		open(sourcepath);
		// Get name of source image w/o extension
		dstdImNoExt=substring(sourceImage,0,lengthOf(sourceImage)-4);
		// Find edges on the target image
		run("Median...", "radius=3");	
		run("Edge Filter", "gaussian=4 edge=complex");	
		selectImage(sourceImage);
		close;
		selectImage("EdgeImag_"+sourceImage);
		close;
		selectImage("EdgeReal_"+sourceImage);
		rename(sourceImage);
		// Apply pre-transformation to the source image
		selectImage(sourceImage);
		// Get the source image
		image = getImageID();
		// Get the statistics of the image
		run("Set Measurements...", "mean redirect=None decimal=3");
		run("Measure");
		meanSource= getResult("Mean", 0);
		run("Multiply...", "value="+scale);
		updateDisplay();
				
		// >> PROCESS TARGET IMAGE
		// Open target image and find edges
		targetImage=File.getName(rawtargetpath);
		open(rawtargetpath);
		// If params[7] is 0 or 2, target is already edge-filtered target.
		if(params[7] == 1) {
			run("Median...", "radius=2");
			//run("Gaussian Blur...", "sigma=4");
			run("Edge Filter", "gaussian=3 edge=complex");	
			selectImage(targetImage);
			close;
			selectImage("EdgeImag_"+targetImage);
			close;
			selectImage("EdgeReal_"+targetImage);
			rename(targetImage);
		}
		selectImage(targetImage);
		// Get the current image
		image = getImageID();
		// Get the statistics of the image
		run("Set Measurements...", "mean redirect=None decimal=3");
		run("Measure");
		meanTarget = getResult("Mean", 0);
		ratio = meanSource/meanTarget;
		run("Multiply...", "value="+ratio);
		run("Multiply...", "value="+scale);
		updateDisplay();
		

		// >> CALCULATE TRANSFORM
		txtfle=dstdImNoExt+".txt";
		div=params[1];
		curl=params[2];
		im=params[3];
		def=params[4];
		stop=params[5];
		run("Tile");
		//run("Unwarp", "targetImage Null sourceImage Null Accurate "+def+" "+def+" "+div+" "+curl+" 0 im"+stop+" verbose save_transformations "+output+sourceImage);//+drtrsf+txtfle);
		run("bUnwarpJ", "source_image=" + sourceImage + " target_image=" + targetImage +" registration=Accurate image_subsample_factor=2 initial_deformation=["+def+"]  final_deformation=["+def+"] divergence_weight="+div+" curl_weight="+curl+" landmark_weight=10 image_weight=im consistency_weight=0 stop_threshold=stop save_transformations save_direct_transformation=["+drtrsf+txtfle+"] save_inverse_transformation=["+indrtrsf+txtfle+"]");
		run("Tile");
		run("Close All");

		// >> APPLY TRANSFORM
		// > Open source image
		open(sourcepath);
		// > If params[7] is 1 or 2, save the edge filtered transform.
		if(params[7]==1 || params[7]==2){
			selectImage(sourceImage);
			run("Median...", "radius=2");
			//run("Gaussian Blur...", "sigma=4");
			run("Edge Filter", "gaussian=3 edge=complex");	
			selectImage(sourceImage);
			close;
			selectImage("EdgeImag_"+sourceImage);
			close;
			selectImage("EdgeReal_"+sourceImage);
			rename(sourceImage);

		}
		// Open target image
		open(targetpath);

		call("bunwarpj.bUnwarpJ_.loadElasticTransform", drtrsf+txtfle, targetImage, sourceImage);
		

		// >> SAVE THE IMAGE AND CLOSE
		selectImage(sourceImage);
		saveAs(output+sourceImage);
		run("Close All");
	}
	k++;
}

run("Quit");
