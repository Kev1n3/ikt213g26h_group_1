# ikt213g26h_group_1
The core concept of this project is to achieve simple, real time surveillance. The system identifies one or more team members and highlights them with a colored box in a live video stream whenever they are in the frame of the camera.
# Precondition 
Ensure that you have the following installed on your system:
- **Python 3.10 or newer** (Python 3.14 was used for this project)
- **Git**
- **Installed a code editor (Recommended)**
## Quickstart / Installation 
1. **Clone the repository:**
```bash
git clone https://github.com/Kev1n3/ikt213g26h_group_1.git
cd ikt213g26h_group_1
```
2. **Pip - install:**

Linux:
```bash
python -m venv .venv 
source .venv/Scripts/activate
pip install -r requirements.txt
```
PowerShell:
```bash
python -m venv .venv 
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
CMD:
```bash
python -m venv .venv 
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```
3. **Add Reference Pictures & Run the Program**
   * Navigate to the **`dataset/`** directory in your project folder.
   * Rename the placeholder folder (e.g., `Test`) to the actual name of the person you want to recognize (e.g., `dataset/Dave/`).
   * Place clear reference photos of that person inside their named folder.
   * Run the training script to extract and save the facial embeddings:
     ```bash
     python src/train_classifier.py
     ```
   * Launch the real time recognition system:
     ```bash
     python main.py
     ```
**Troubleshooting & Common Problems**

1. **Different Python Version Installed** 

    If your operating system uses a different Python launcher or environment setup, you may need to 
    use the Windows launcher command (**py**) instead of the standar **python** command.

* Instead of:
    ```bash
    python main.py
    ```


* Use:  
    ```bash 
    py main.py 
    ```

2. **Easiest way if it does not work through command line**

    If the terminal commands for some reason fails or have difficulties to rename 
directory or input pictures to the reference directory. 
    * Open the project directly inside your editor's built in terminal (such as PyCharm or VS code)
    * Run your terminal commands from within the editor environment to ensure your virtual environment (.venv) is correctly linked and active.
    * Make sure you have renamed your reference folders and added your target recognition pictures to the dataset/ directory before executing the training and main scripts.
