
software application broken down into two sections:
(breakdown of all stages for development in 'development steps' note)

##### **<span style="color:rgb(146, 208, 80)">section A</span> cleaning tool used for:
- takes blurry, noise, low-quality, or incomplete medical scans and:
- uses AI to clean, sharpen, denoise or highlight important areas 
- helps doctors / doesn't generate anything new 

##### **<span style="color:rgb(0, 176, 240)">section B</span> generating tool used for: 
- generate missing parts of a scan 
- can synthesize medical scans 
- important features:
	- Missing region generation 
	- Synthetic data generation 
	- Cross-modality translation
	- Super-resolution via GAN's
	- Motion artifact correction 
	- Noise-to-Image translation 
	- AI-powered Pre-screening / Auto-Annotation




---------------------------------------------------------
### **Full tech stack:**
Python  3.10+ 
Anaconda 

---------------------------------------------------------
<span style="color:rgb(146, 208, 80)">AI/ML:</span>
PyTorch                                        --->  framework to build / train deep learning models
Torchvision / MONAI                   --->  prebuilt models and tools for medical scans
scikit-learn                                   --->  classification, clustering for machine learning
OpenCV / PIL                               --->  image processing
Grad-CAM / Captum / TorchXAI  ---> tools for explaining AI decisions

-------------------------------------------------------
<span style="color:rgb(146, 208, 80)">Data handling & medical image support:</span>
- pydicom                               ---> reads and works with DICOM medical scan files 
- NumPy / Pandas                  ---> handles arrays, tables and data manipulation 
- SimpleITK / nibabel             ---> work with 3D and medical imaging formats (like MRI)
---------------------------------------------------------
<span style="color:rgb(146, 208, 80)">GUI / desktop application framework:</span>
- PyQt6                                  ---> framework to build the app's graphical interface 
- Qt designer                         ---> drag-and-drop tool for UI
---------------------------------------------------------
<span style="color:rgb(146, 208, 80)">Database:</span>
- SQLite (for development)    ---> for development stage later switch to MySQL or PostgreSQL
---------------------------------------------------------
<span style="color:rgb(146, 208, 80)">Packaging:</span>
- PyInstaller                            ---> converts your python app into a `.exe` or desktop application
- NSIS or Inno (for setup)       ---> tools to create an installer wizard for app 
---------------------------------------------------------
<span style="color:rgb(146, 208, 80)">Security & privacy:</span>
- PyCryptodome / Fernet                                          ---> encrypts sensitive data
- anonymization tools (form pydicom / simpleITK)  ---> remove patient-identifiable info
---------------------------------------------------------
<span style="color:rgb(146, 208, 80)">Optional libraries / enhancements:</span>
- Matplotlib / Plotly               ---> data and image visualisation 
- TensorBoard / WandB         ---> track and monitor model training 
- OpenCL / CUDA                  ---> accelerate AI processing using GPU
---------------------------------------------------------








