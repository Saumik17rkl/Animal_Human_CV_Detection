# 🐾 Animal and Human Detection System

A real-time web application that detects and classifies animals and humans in videos using state-of-the-art deep learning models. The system combines YOLOv8 for object detection with EfficientNet for precise classification, providing accurate and efficient video analysis.

![Demo](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.0+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

## ✨ Features

- 🎥 **Video Processing**: Upload and process video files with progress tracking
- 🎯 **Accurate Detection**: Combines YOLOv8 and EfficientNet for high-accuracy detection
- 🚀 **Real-time Analysis**: Background processing with real-time progress updates
- 📱 **Responsive UI**: Clean, modern interface that works on all devices
- 📊 **Detailed Results**: View and download processed videos with bounding boxes and confidence scores
- ⚡ **Efficient**: Optimized for performance with background task processing

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (recommended for faster processing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/animal_human_detection.git
   cd animal_human_detection
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

3. **Use the application**:
   - Click to upload or drag & drop a video file
   - Monitor the processing progress in real-time
   - View and download the processed video with detections

## 🛠️ Project Structure

```
animal_human_detection/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore file
├── models/            # Pre-trained models directory
│   └── ...
├── template/          # Frontend templates
│   └── index.html     # Main web interface
├── uploads/           # Temporary storage for uploaded files
└── processed/         # Storage for processed videos
```

## 🧠 Models Used

- **YOLOv8**: For object detection and localization
- **EfficientNet**: For fine-grained classification of animals and humans

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue on GitHub.
- `--source`: Input source (path to image/video or camera index)
- `--output`: Directory to save output files (default: 'outputs/')
- `--conf`: Confidence threshold (default: 0.5)
- `--view`: Display the output in a window (default: False)

## Project Structure

```
animal_human_detection/
├── models/           # Pre-trained models
├── utils/            # Utility scripts
├── outputs/          # Output directory for results
├── detect.py         # Main detection script
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Dependencies

- Python 3.8+
- OpenCV
- PyTorch
- TorchVision
- NumPy
- Matplotlib (for visualization)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
