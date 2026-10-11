# ikt213g26h_group_1
The core concept of this project is to achieve simple, real time surveillance. The system identifies one or more team members and highlights them with a colored box in a live video stream whenever they are in the frame of the camera.
# Precondition 
Ensure that you have the following installed on your system:
- **Python 3.10 or newer** (Python 3.14 was used for this project)
- **Git**
## Quickstart / Installation 
1. **Clone the repository:**
```bash
git clone https://github.com/Kev1n3/ikt213g26h_group_1.git
cd ikt213g26h_group_1
```
2. **Pip - install:**

linux

```bash
python -m venv .venv 
pip install -r requirements.txt
```
3. **Add Reference Pictures & Run the Program**
   * Navigate to the **`dataset/`** directory in your project folder.
   * Rename the placeholder folder (e.g., `Test`) to the actual name of the person you want to recognize (e.g., `dataset/Dave/`).
   * Place clear reference photos of that person inside their named folder.
   * Run the training script to extract and save the facial embeddings:
     ```bash
     python src/train_classifier.py
     ```
   * Launch the real-time recognition system:
     ```bash
     python main.py
     ```
