// Version 8
// Objective avoid edges at the border.
// Changed, first median and edge filter, then normalisation.
// Added an additional transformation argument

setBatchMode(false);

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
// Location of the pre-transform folder (.txt file)
trf_path =  argArray[7];
print(trf_path);

// Print input paths (for debugging purpose)
// print("Arguments:");
// print(target);
// print(input);
// print(output);


// >> LOAD TARGET IMAGE
//get list of files in the folder and assigns it to a variable array. 
list = getFileList(input);

scale = params[6];
k = 0;
suffix =  ".tif";
while (k<list.length) {
	// >> Align the edge fitered using bUnwarpJ
	sourceImage=list[k];
	// Get name of source image w/o extension
	dstdImNoExt=substring(sourceImage,0,lengthOf(sourceImage)-4);
	
	if (endsWith(sourceImage, suffix) && (params[0]==0)){
		// >> (A) PROCESS SOURCE IMAGE
		sourcepath=input+sourceImage;
		open(sourcepath);
		targetImage=File.getName(targetpath);
		open(targetpath);
		rename(targetImage);

		selectImage(sourceImage);
		
		run("Concatenate...", "image1="+targetImage+" image2="+sourceImage);
		selectImage("Untitled");
		run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=300 maximum_image_size=514 feature_descriptor_size=16 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Affine interpolate");
		run("Stack to Images");
		rename(sourceImage);
		close("Aligned-0001");
		close("Untitled");

		// Find edges on the target image
		//run("Median...", "radius=7");	

		if(params[7] == 1 || params[7] == 2  || params[7] == 3  || params[7] == 4) {
			run("Median...", "radius=3");	
			run("Edge Filter", "gaussian=1 edge=complex");	
			selectImage(sourceImage);
			close;
			selectImage("EdgeImag_"+sourceImage);
			close;
			selectImage("EdgeReal_"+sourceImage);
		}
		rename(sourceImage);
		// Apply pre-transformation to the source image
		selectImage(sourceImage);
		// Get the source image
		image = getImageID();
		run("Duplicate...", " ");
		run("Square");
		run("Square Root");
		// Get the statistics of the image
		run("Set Measurements...", "mean redirect=None decimal=10");
		run("Measure");
		meanSource= getResult("Mean", 0);
		close;
		selectImage(sourceImage);
		run("Multiply...", "value="+scale);
		updateDisplay();
				
		// >> PROCESS TARGET IMAGE
		// Open target image and find edges
		open(targetpath);
		rename(targetImage);
		// If params[7] is 0 or 3, target is already edge-filtered target.
		if(params[7] == 1 || params[7] == 2) {
			run("Median...", "radius=3");
			run("Edge Filter", "gaussian=1 edge=complex");	
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
		run("Duplicate...", " ");
		run("Square");
		run("Square Root");
		// Get the statistics of the image
		run("Set Measurements...", "mean redirect=None decimal=10");
		run("Measure");
		meanTarget = getResult("Mean", 0);
		ratio = meanSource/meanTarget;
		selectImage(targetImage);
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
		if(params[7] == 5) {
			run("Extract SIFT Correspondences", "source_image=" + sourceImage + " target_image=" + targetImage +"  initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=400 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 filter maximal_alignment_error=5 minimal_inlier_ratio=0.05 minimal_number_of_inliers=7 expected_transformation=Similarity");
		}
		run("bUnwarpJ", "source_image=" + sourceImage + " target_image=" + targetImage +" registration=Accurate image_subsample_factor=2 initial_deformation=["+def+"]  final_deformation=["+def+"] divergence_weight="+div+" curl_weight="+curl+" landmark_weight=10000 image_weight=im consistency_weight=10 stop_threshold=stop save_transformations save_direct_transformation=["+drtrsf+txtfle+"] save_inverse_transformation=["+indrtrsf+txtfle+"]");
		run("Tile");
		run("Close All");

		// >> APPLY TRANSFORM
		// > Open source image
		open(sourcepath);
		rename(sourceImage);
		open(targetpath);
		rename(targetImage);

		run("Concatenate...", "image1="+targetImage+" image2="+sourceImage);
		selectImage("Untitled");
		run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=300 maximum_image_size=514 feature_descriptor_size=16 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Affine interpolate");
		run("Stack to Images");
		rename(sourceImage);
		close("Aligned-0001");

		// > If params[7] is 1 or 3, you want to save the edge filtered transform
		if(params[7]==1 || params[7]==3){
			selectImage(sourceImage);
			run("Median...", "radius=3");
			run("Find Edges");
		}
		
		// Open target image
		call("bunwarpj.bUnwarpJ_.loadElasticTransform", drtrsf+txtfle, sourceImage, sourceImage);
		call("bunwarpj.bUnwarpJ_.loadElasticTransform", trf_path, sourceImage, sourceImage);

		

		// >> SAVE THE IMAGE AND CLOSE
		selectImage(sourceImage);
		saveAs(output+sourceImage);
		run("Close All");
		run("Clear Results");
  		print("\\Clear");
	} else if (params[0]==1){
		// >> (B) Apply transform bUnwarpJ and adjust linearly SIFT
		// >> Align raw image using A calibrated transform, a reference image and a linear adjustement
		// > Open source image
		sourcepath=input+sourceImage;
		open(sourcepath);

		// Open target image
		targetImage=File.getName(targetpath);
		open(targetpath);
		call("bunwarpj.bUnwarpJ_.loadElasticTransform", trf_path, sourceImage, sourceImage);
		selectImage(sourceImage);
		makeRectangle(30,37,460,460);
		run("Crop");
		
		run("Concatenate...", "keep image1="+targetImage+" image2="+sourceImage);
		close(sourceImage);
		close(targetImage);
		selectImage("Untitled");
		run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=300 maximum_image_size=514 feature_descriptor_size=16 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Affine interpolate");
		run("Stack to Images");
		rename(sourceImage);
		// >> SAVE THE IMAGE AND CLOSE
		saveAs(output+dstdImNoExt+".tif");
		run("Close All");
	} else if (params[0]==2){
		// >> (C) MOPS + bUnwarpJ
		// >> Align raw image using a point of interest detector and a non-linear transform
		// > Open source image
		sourcepath=input+sourceImage;
		open(sourcepath);
		sourceInfo = getInfo("image.description");
		
		// Open target image
		targetImage=File.getName(targetpath);
		open(targetpath);
		call("bunwarpj.bUnwarpJ_.loadElasticTransform", trf_path, targetImage, sourceImage);
		txtfle=dstdImNoExt+".txt";
		selectImage(sourceImage);
		makeRectangle(30,37,460,460);
		run("Crop");
		run("Extract MOPS Correspondences", "source_image="+ sourceImage +" target_image="+targetImage+" initial_gaussian_blur=1 steps_per_scale_octave=1 minimum_image_size=420 maximum_image_size=550 feature_descriptor_size=8 closest/next_closest_ratio=0.92 maximal_alignment_error=10 inlier_ratio=0.05 expected_transformation=Affine");
		run("bUnwarpJ", "source_image=" + sourceImage + " target_image=" + targetImage +" registration=Accurate image_subsample_factor=2 initial_deformation=[Very Coarse]  final_deformation=[Very Coarse] divergence_weight=1000 curl_weight=1000 landmark_weight=1 image_weight=10 consistency_weight=10 stop_threshold=0.01 save_transformations save_direct_transformation=["+drtrsf+txtfle+"] save_inverse_transformation=["+indrtrsf+txtfle+"]");
		selectImage("Registered Source Image");
		run("Stack to Images");
		selectImage("Registered Source Image");
		rename(dstdImNoExt+".tif");
		setMetadata("Info", sourceInfo);

		// >> SAVE THE IMAGE AND CLOSE
		saveAs(output+dstdImNoExt+".tif");
		run("Close All");

	}
	k++;
}

run("Quit");
