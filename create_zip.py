import os
import zipfile
from pathlib import Path

def zip_files(destination, source_dir, include_dirs, include_files, exclude_files=None):
    if exclude_files is None:
        exclude_files = []
    
    # Convert to absolute paths
    exclude_files = [os.path.abspath(f) for f in exclude_files]
    include_files = [os.path.abspath(f) for f in include_files]
    
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add directories
        for dir_path in include_dirs:
            dir_path = os.path.abspath(dir_path)
            for root, dirs, files in os.walk(dir_path):
                # Skip __pycache__ directories
                if '__pycache__' in dirs:
                    dirs.remove('__pycache__')
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path not in exclude_files:
                        arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                        zipf.write(file_path, arcname)
        
        # Add individual files
        for file_path in include_files:
            if os.path.exists(file_path) and file_path not in exclude_files:
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_zip = os.path.join(base_dir, "animal_human_detection_project.zip")
    
    # Define what to include
    include_dirs = [
        os.path.join(base_dir, "datasets"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "outputs"),
        os.path.join(base_dir, "test_videos"),
        os.path.join(base_dir, "wandb")
    ]
    
    include_files = [
        os.path.join(base_dir, "animal_human_detection.py"),
        os.path.join(base_dir, "test.py"),
        os.path.join(base_dir, "yolov8n.pt")
    ]
    
    # Exclude code.ipynb
    exclude_files = [os.path.join(base_dir, "code.ipynb")]
    
    # Create the zip file
    zip_files(output_zip, base_dir, include_dirs, include_files, exclude_files)
    print(f"Created zip file at: {output_zip}")
