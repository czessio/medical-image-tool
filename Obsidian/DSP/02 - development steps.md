(the software development cycle broken down into stages: 1 - 6 
including a stage idea bank which holds optional features for later expansion)

---------------------------
```
Initial development will be done using pre-trained / already existing AI models. 

Within the project we will be using two groups of pre-trained models:

1. foundational models, widely used and reliable 
2. more recent specialised models, testing their potential 

then comparing the old models with the new.
```
---------------------------
```
Also if there is enough time after intial development I will be working on creating my own model specifically to be used in this application for best results, then test and compare against the other groups of models. 
```
---------------------------


#### **Stage 1: Core cleaning tool**

- Load and view medical images (DICOM, PNG, etc.)
    
- Apply AI-based cleaning:
    
    - Denoising
        
    - Super-resolution
        
    - Artifact removal
        
- Display before/after comparison
    
- Export cleaned image
    

---

#### **Stage 2: Basic UI + usability**

- Desktop GUI using PyQt6 or Kivy
    
- Drag-and-drop image loading
    
- Save/load sessions
    
- Basic user preferences/settings
    

---

#### **Stage 3: Enhancements**

- Add model for **motion artifact correction**
    
- Add **auto-segmentation and classification**
    
    - Highlight and label critical regions (e.g., tumours, lesions)
        
- Include pre-trained models and local inference
    
- Add zoom/pan tools and basic annotations
    

---

#### **Stage 4: Generative intelligence**

- **Missing region generation**
    
- **Noise-to-image translation**
    
- **Super-resolution via GANs**
    
- **Synthetic data generation** (for training or augmentation)
    
- (Optional) Cross-modality scan translation (e.g., CT → MRI)
    
- **Explainable AI Outputs** (e.g., Grad-CAM overlays, visual attention maps)
    

---

#### **Stage 5: Doctor assistance features**

- **AI-powered pre-screening** (auto-annotate suspicious regions)
    
- Generate simple summary reports from AI findings
    
- View full image processing history/logbook
    
- **Adaptive learning system** (model improves from usage and feedback)
    

---

#### **Stage 6: Final packaging & polish**

- Package application as `.exe` or app bundle (PyInstaller)
    
- Onboarding/help system
    
- Performance optimization
    
- Final cross-platform testing






#### **Idea bank (Optional features for later expansion)**
A collection of features to explore **after core development** is done:


- **Predictive Analytics for Disease Progression**  
    Forecast how a condition might evolve over time.
    
- **Integration with Electronic Health Records (EHR)**  
    Link AI insights with patient profiles.
    
- **Real-time Collaboration Tools**  
    Doctors collaborating remotely on the same scan.
    
- **Mobile Accessibility / Companion App**  
    Access, view, or sync image results on mobile.
    
- **Customizable Workflow Integration**  
    Fit the tool into different hospitals’ systems.
    
- **Quality Assurance Tools**  
    Assess scan quality and flag bad inputs.
    
- **Advanced Data Privacy/Anonymization**  
    For regulatory compliance and sharing.




hj