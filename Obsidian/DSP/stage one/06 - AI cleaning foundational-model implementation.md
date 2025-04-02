(Overview of work completed within this section including testing and any additional information)

----------------------------

### **<span style="color:rgb(146, 208, 80)">Foundational cleaning models group, utilising 3 classic models:</span>

`DnCNN Denoiser`:
`ai/cleaning/models/foundational/dncnn_denoiser.py`
- Implements the "Beyond a Gaussian Denoiser" architecture
- Uses a deep CNN with residual learning to remove noise
- Excellent for medical image denoising where preserving fine details is critical

`EDSR Super-Resolution`:
`ai/cleaning/models/foundational/edsr_super_resolution.py`
- Implements the Enhanced Deep Super-Resolution Network
- Uses residual blocks and pixel shuffle for efficient upscaling
- Supports multiple upscaling factors (2x, 3x, 4x)

`U-Net Artifact Removal`:
`ai/cleaning/models/foundational/unet_artifact_removal.py`
- Implements the classic U-Net architecture for artifact removal
- Features encoder-decoder structure with skip connections
- Excellent for preserving structural details while removing artifacts

---------------------------------

### <span style="color:rgb(146, 208, 80)">Each model:</span>

1. inherits from `TorchModel`
2. implements required methods 
3. registered with `ModelRegistry`
4. handles both greyscale and RGB



------------------------------
<p align="center" style="font-size:28px; font-weight:bold; color:#F248FE;">
  Testing
</p>
------------------------

# Testing Foundational Models

---

#### **tests/ai/cleaning/foundational/test_dncnn_denoiser.py
Purpose: Tests the functionality and architecture of the DnCNN-based denoising model, including RGB handling and model components.

| Test Case                 | Description                                           | Status | Reason                                   |
| ------------------------- | ----------------------------------------------------- | ------ | ---------------------------------------- |
| test_model_registration   | Verifies DnCNN denoiser registers with model registry | ✅Pass  | Doesn't require model inference          |
| test_model_initialization | Tests model initializes without weights               | ✅Pass  | Only initializes architecture            |
| test_inference            | Tests full inference pipeline with denoising          | ✅Pass  | Using small model version (fewer layers) |
| test_rgb_input_handling   | Validates RGB to grayscale conversion                 | ✅Pass  | Using small model version                |
| test_model_architecture   | Checks model components are correctly structured      | ✅Pass  | Only checks model structure              |

---

#### **tests/ai/cleaning/foundational/test_edsr_super_resolution.py
Purpose: Tests EDSR-based super-resolution functionality, including 2x and 3x upscaling and model component structure.

| Test Case                 | Description                             | Status | Reason                                   |
| ------------------------- | --------------------------------------- | ------ | ---------------------------------------- |
| test_model_registration   | Verifies model registers properly       | ✅Pass  | Doesn't require model inference          |
| test_model_initialization | Tests model initializes without weights | ✅Pass  | Only initializes architecture            |
| test_inference_2x         | Tests 2x upscaling functionality        | ✅Pass  | Using small model version (fewer blocks) |
| test_inference_3x         | Tests 3x upscaling functionality        | ✅Pass  | Using small model version                |
| test_rgb_input_handling   | Validates RGB input handling            | ✅Pass  | Using small model version                |
| test_model_components     | Checks model components structure       | ✅Pass  | Only verifies component existence        |

---

#### **tests/ai/cleaning/foundational/test_unet_artifact_removal.py
Purpose: Tests U-Net-based artifact removal model, including up-sampling options and model architecture.

| Test Case                  | Description                             | Status | Reason                                     |
| -------------------------- | --------------------------------------- | ------ | ------------------------------------------ |
| test_model_registration    | Verifies model registers properly       | ✅Pass  | Doesn't require model inference            |
| test_model_initialization  | Tests model initializes without weights | ✅Pass  | Only initializes architecture              |
| test_inference_small       | Tests inference with minimal model      | ✅Pass  | Using small model version (fewer channels) |
| test_bilinear_vs_transpose | Tests both upsampling options           | ✅Pass  | Compares alternative architectures         |
| test_rgb_input_handling    | Validates RGB input handling            | ✅Pass  | Using small model version                  |
| test_model_structure       | Verifies U-Net component structure      | ✅Pass  | Only checks component existence            |

```
No errors encountered during testing the foundational models
```

### **Testing outputs:
---------------------------------
<details>
  <summary>Click to view Pytest results</summary>
  <img src="Pasted image 20250401153740.png" alt="Pytest Results" width="600">
</details>
<p>
   <a href="obsidian://open?vault=Obsidian%20Vault&file=(DUMP)%2FPasted%20image%2020250401153740.png">Link to image</a>
</p>
---------------------------------------------- 


 <p align="center" style="font-size:28px; font-weight:bold; color:#F248FE;">
  All work pushed to GitHub
</p>
