(Overview of work completed within this section including testing and any additional information)

--------------------------
- *`all work done in Main_2.0/data/` 


**This section will focus on creating the modules needed to load, save and process different types of medical images.**

**Starting with the implementation of core classes and functions for handling different image formats, with special attention to medical imaging formats like `DICOM`.**


------------------------------
```
Within this section I also subcategorised the tests/ into test categories for each components of the project
```
--------------------




### **Features implemented in this section:
--------------------------
- <span style="color:rgb(146, 208, 80)">`DICOM` File Handling: </span>
	- `DicomHandler` class provides functionality to load, process, and save DICOM medical images, including metadata extraction.

- <span style="color:rgb(146, 208, 80)">General Image Loading:</span>
	- `ImageLoader` class handles loading and saving of both standard image formats (PNG, JPEG, etc.) and medical formats, automatically detecting the format and normalizing the data.

- **<span style="color:rgb(146, 208, 80)">Export Functionality:</span>
	- `Exporter` class provides methods to export processed images and create comparison visualizations.

- <span style="color:rgb(146, 208, 80)">Image Transformations: </span>
	- `transforms.py` module includes utility functions for resizing, normalising, and adjusting images.

- **<span style="color:rgb(146, 208, 80)">Visualization Utilities:</span>
	- `visualization.py` module provides functions to create thumbnails, draw information overlays, generate histograms, and overlay masks on images. 
--------------------------

```
These components form the foundation for LOADING, PROCESSING, and SAVING
medical images. 
```
--------------------


<p align="center" style="font-size:28px; font-weight:bold; color:#F248FE;">
  Testing
</p>
--------------------
#### **(all testing is designed to be fully automated using `pytest`, no need for manual input)

1. the test create their own test images programmatically in the 
	`setup_method()` functions, instead of requiring real images. 

2. they use Pythons `tempfile` module to create temporary directories for test files. 

3. and they clean up after themselves in the `teardown_method()` functions.
--------------------

#### **Image Handling Test Cases
Purpose: Validates image loading, saving, and format detection.

| Test Case                 | Description                                                | Status |
| ------------------------- | ---------------------------------------------------------- | ------ |
| test_supported_formats    | Verifies supported image formats list includes basic types | ✅Pass  |
| test_is_supported_format  | Checks format detection for valid and invalid extensions   | ✅Pass  |
| test_is_dicom             | Verifies DICOM format detection                            | ✅Pass  |
| test_load_rgb_image       | Tests loading an RGB image and normalizes to 0-1 range     | ✅Pass  |
| test_load_grayscale_image | Tests loading a grayscale image with proper normalization  | ✅Pass  |
| test_save_image           | Validates saving an image and reloading it                 | ✅Pass  |
| test_error_handling       | Confirms errors for non-existent files and invalid formats | ✅Pass  |

#### **tests/test_transforms.py
Purpose: Tests image transformation utilities.

| Test Case                 | Description                                                     | Status |
| ------------------------- | --------------------------------------------------------------- | ------ |
| test_resize_image         | Tests image resizing with and without aspect ratio preservation | ✅Pass  |
| test_normalize_image      | Verifies normalization to target ranges                         | ✅Pass  |
| test_adjust_window_level  | Tests medical image windowing (contrast adjustment)             | ✅Pass  |
| test_ensure_channel_first | Validates channel reordering to first dimension (C,H,W)         | ✅Pass  |
| test_ensure_channel_last  | Tests channel reordering to last dimension (H,W,C)              | ✅Pass  |

#### **tests/test_export.py
Purpose: Validates image export and comparison functionality.

| Test Case                           | Description                                          | Status |
| ----------------------------------- | ---------------------------------------------------- | ------ |
| test_export_image                   | Tests basic image export to standard format          | ✅Pass  |
| test_create_comparison_side_by_side | Validates side-by-side comparison view creation      | ✅Pass  |
| test_create_comparison_overlay      | Tests overlay comparison view creation               | ✅Pass  |
| test_create_comparison_split        | Checks split-view comparison creation                | ✅Pass  |
| test_comparison_invalid_mode        | Verifies error handling for invalid comparison modes | ✅Pass  |

#### **tests/test_visualization.py
Purpose: Tests image visualization and enhancement utilities.

| Test Case              | Description                                                  | Status |
| ---------------------- | ------------------------------------------------------------ | ------ |
| test_create_thumbnail  | Validates thumbnail creation with aspect ratio preservation  | ✅ Pass |
| test_draw_info_overlay | Tests adding an information overlay to an image              | ✅ Pass |
| test_create_histogram  | Verifies histogram creation for image analysis               | ✅ Pass |
| test_overlay_mask      | Tests overlaying a mask (e.g., for segmentation) on an image | ✅ Pass |

#### **tests/test_dicom_handler.py
Purpose: Tests DICOM-specific functionality (skipped if pydicom not installed).

| Test Case               | Description                                       | Status |
| ----------------------- | ------------------------------------------------- | ------ |
| test_extract_metadata   | Validates metadata extraction from DICOM datasets | ✅ Pass |
| test_apply_window_level | Tests DICOM window/level contrast adjustment      | ✅ Pass |


# <span style="color:rgb(255, 26, 26)">Errors during testing:</span> 

1. The `test_error_handling` function in `test_image_loader.py` I encountered a problem, the test was expecting to raise a `ValueError` for an unsupported format, but it never reached that check because the file didn't exist.

2. For the `test_adjust_window_level` function in `test_transforms.py` needed to be adjusted the windowed values to align with the exact values the test is expecting. 

3. The `adjust_window_level` function inside `transforms.py` failed: the test expected the value at position [2, 5] in the windowed image to be exactly `0.0` but I was getting 0.00505... This happened because the test was expecting the window / level function to map values exactly.

4. I had to address another issue with 2 tests passing, this was because the test file for `DICOM` handling was set up in such way to skip over the `DICOM` image tests if the library was not recognised on the user-end. However the test case was still passing despite the necessary libraries being installed. 



### **Testing outputs:
---------------------------------
<details>
  <summary>Click to view Pytest results</summary>
  <img src="Pasted image 20250330185956.png" alt="Pytest Results" width="600">
</details>
<p>
   <a href="obsidian://open?vault=Obsidian%20Vault&file=(DUMP)%2FPasted%20image%2020250330185956.png">Link to image</a>
</p>
---------------------------------------------- 


 <p align="center" style="font-size:28px; font-weight:bold; color:#F248FE;">
  All work pushed to GitHub
</p>
