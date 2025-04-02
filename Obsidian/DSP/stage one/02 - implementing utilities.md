(Overview of work completed within this section including testing and any additional information)

---------------------------------------
- `all work done in Main_2.0/utils/` 

#### **structure:**
Structure was created by running a simple python script, and 
it was designed in the planning phase. 


#### **utilities:**
Started by writing in the files for the `utils` folder which broadly speaking 
establish the core utility modules that will support the rest of the application. 
- The 3 modules that I worked on here are:
	- `system_info.py`
	- `logging_setup.py`
	- `config.py`


`config.py`:
Manages application configuration, including model settings, paths, and GUI preferences. This will be used through the application to ensure consistency. 

`logging_setup.py`:
Sets up logging for both console and file, which is needed for debugging and tracking application behaviour. 

`system_info.py`:
Gathers information about the system's hardware capabilities, particularly focused on detecting GPU availability for AI model inference. This will help the application decide how to run the AI models efficiently. 

---------------------------------------

<p align="center" style="font-size:28px; font-weight:bold; color:#F248FE;">
  Testing
</p>



---------------------------------------
I have created a `tests` folder which will contain all scripts that I use for testing different features and modules along the way. All testing files are done using `pytest`.  

The 3 testing files that we used here are:
- `config.py`
- `logging.py`
- `test_system_info.py`
---------------------------------------

### **<span style="color:rgb(146, 208, 80)">All tests were passed successfully</span> 
---------------------------------------

#### **tests/config.py
Purpose: Tests configuration handling and default value behaviour.

| Test Case                      | Description                                                   | Status |
|-------------------------------|---------------------------------------------------------------|--------|
| test_config_creation          | Checks if default values are correctly set and config created | ✅ Pass |
| test_config_save_load         | Ensures config changes persist after save/load cycle          | ✅ Pass |
| test_config_get_default       | Returns fallback value when key is missing                    | ✅ Pass |

---

#### **tests/logging.py
Purpose: Validates logging setup and level filtering.

| Test Case                      | Description                                                       | Status |
|-------------------------------|-------------------------------------------------------------------|--------|
| test_logging_creates_file     | Confirms `.log` file is created after logging starts              | ✅ Pass |
| test_logging_levels           | Ensures lower-than-log-level messages are filtered out            | ✅ Pass |

---

#### **tests/test_system_info.py
Purpose: Verifies GPU/CPU detection and memory usage estimation.

| Test Case                                 | Description                                       | Status |
| ----------------------------------------- | ------------------------------------------------- | ------ |
| test_system_info_contains_required_fields | Checks OS/CPU/RAM/GPU info presence               | ✅Pass  |
| test_optimal_device                       | Device selection returns `cpu` or `cuda`          | ✅Pass  |
| test_memory_estimation                    | Validates memory estimation for float32/64 images | ✅Pass  |


### **Testing outputs:
---------------------------------
<details>
  <summary>Click to view Pytest results</summary>
  <img src="Pasted image 20250330152303.png" alt="Pytest Results" width="600">
</details>
<p>
   <a href="obsidian://open?vault=Obsidian%20Vault&file=(DUMP)%2FPasted%20image%2020250330152303.png">Link to image</a>
</p>
---------------------------------





<p align="center" style="font-size:28px; font-weight:bold; color:#F248FE;">
  All work pushed to GitHub
</p>
