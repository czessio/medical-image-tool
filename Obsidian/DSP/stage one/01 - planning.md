(stage one development with some features from stage two)
(core AI image cleaning / enhancing tool with structure for image generation) 




## **Project structure:**
------------------------------------------
```
main_1.0/
├── assets/                      # static resources 

├── ai/
│   ├── __init__.py
│   ├── cleaning/                # Section A: Cleaning tools
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── denoising.py            # image denoising
│   │   │   ├── super_resolution.py     # enhance resolution
│   │   │   └── artifact_removal.py     # remove scan artifacts
│   │   └── inference/
│   │       ├── __init__.py
│   │       └── cleaning_pipeline.py    # cleaning workflow
│   ├── generation/              # Section B: generation tools (for future stages)
│   │   ├── __init__.py
│   │   └── placeholder.md       # documentation for future implementation
│   └── weights/                 # pre-trained model weights
│       ├── cleaning/            # weights for cleaning models
│       └── generation/          # weights for future generative models

├── data/
│   ├── __init__.py
│   ├── io/                      # input/output operations
│   │   ├── __init__.py
│   │   ├── dicom_handler.py     # DICOM loading/saving
│   │   ├── image_loader.py      # standard image formats
│   │   └── export.py            # export functionality
│   └── processing/              # image processing utilities
│       ├── __init__.py
│       ├── transforms.py        # image transformations
│       └── visualization.py     # visualization helpers

├── gui/
│   ├── __init__.py
│   ├── main_window.py           # main application window
│   ├── viewers/
│   │   ├── __init__.py
│   │   ├── image_viewer.py      # single image viewing
│   │   └── comparison_view.py   # before/after comparison
│   ├── panels/
│   │   ├── __init__.py
│   │   └── cleaning_panel.py    # controls for cleaning tools
│   └── forms/                   # Qt Designer forms
│       ├── main_window.ui
│       └── preferences.ui

├── utils/
│   ├── __init__.py
│   ├── config.py                # configuration management
│   ├── logging_setup.py         # logging configuration
│   └── system_info.py           # system information (GPU, memory)

├── main.py                      # application entry point
├── requirements.txt             # dependencies
└── README.md                    # project documentation
```
------------------------------------------


## **Key classes and Methods:**
------------------------------------------
#### **AI Module:**
```python
 # ai/cleaning/models/denoising.py
 class DenoisingMode:
	 def __init__(self, model_path=None, device='cuda')
	 # start denoising model

	 def process(self, image):
	 # run denoising on input image


#ai/cleaning/interface/cleaning_pipeline.py
class CleaningPipeline:
	def __init__(self):
	# start cleaning models 

	def process_image(self, imgage, options):
	# apply selected cleaning options to image 
	# return image
```

***within `process_image` make sure to pass metadata alongside the image for DICOM-specific controller logic***

------------------------------------------
#### **Data Module:**
```python
# data/io/dicom_handler.py
class DicomHandler:
	@ staticmethod
	def load(path):
	# loads DICOM file

	@ staticmethod
	def save(image, metadata, path):
	# save as DICOM

# data/processing/visualization.py
def create_side_by_side(original, processed):
# create comparison visualization
```

------------------------------------------
#### **GUI Module:**
```python
# gui/main_window.py
class MainWindow(QMainWindow):
	def __init__(self):
	# UI components 
	# connect signals 

	def load_image(self):
	# user loads image

	def process_image(self):
	# send image to cleaning pipeline 
	# update display with result 


# gui/panels/cleaning_panel.py
class CleaningPanel(QWidget):
	def __init__(self, parent=None):
	# UI for image cleaning options

	def get_selected_options(self):
	# shows pipiline enabled image cleaning options 
```
------------------------------------------





## **Communication Flow:**
------------------------------------------
1. User loads image through `MainWindow`
2. User selects cleaning options in `CleaningPanel`
3. `MainWindow` collects options and sends image to `CleaningPipeline`
4. `CleaningPipeline` processes  image using appropriate models 
5. Processed image is returned and displayed in `ComparisonView`
------------------------------------------
## **Design points:**
------------------------------------------
- Clear separation between different stages 
	- cleaning (stage one)
	- generation (stage four)
- Modular architecture: each technique used in the cleaning process can be applied independently
- Easy to input any generative features in the next stages 
- UI / Logic separation 
- Threading: heavy processing runs in background threads to keep UI responsive 
------------------------------------------
## **Sections for stage one:**
------------------------------------------
- **Core Utilities and Configuration** 
- **Data Handling and Image I/O** (loading DICOM, PNG, etc.)
- **AI Model Base Classes and Infrastructure** (model loading, inference framework)
- **AI Cleaning Models Implementation** (denoising, super-resolution, artifact removal)
- **Basic GUI Framework** (main window, image display)
- **Comparison View and Export Functionality** (before/after view, export capabilities)
- **Integration and Application Entry Point** (tying everything together)
------------------------------------------
