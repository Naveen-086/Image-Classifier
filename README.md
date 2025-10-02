# Image Classifier

A modern AI-powered image classification application that uses ResNet50 neural network to classify images into 10 CIFAR-10 categories with high accuracy.

## 🎯 Features

- **AI Classification**: Uses ResNet50 model trained on ImageNet with CIFAR-10 category mapping
- **10 Categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Real-time Processing**: Fast image classification with confidence scores
- **Modern UI**: Clean, responsive React frontend with TypeScript
- **Dynamic Results**: Non-hardcoded confidence scores with realistic predictions

## 🏗️ Architecture

### Backend
- **Framework**: Flask (Python)
- **AI Model**: ResNet50 → CIFAR-10 mapping
- **Image Processing**: PIL/Pillow
- **Deep Learning**: TensorFlow/Keras

### Frontend
- **Framework**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **Build Tool**: Vite
- **Icons**: Lucide React

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **npm**: 8 or higher

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Naveen-086/Image-Classifier.git
cd Image-Classifier
```

### 2. Backend Setup
```bash
# Navigate to project directory
cd project

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install Python dependencies
cd python_backend
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
# Navigate back to project root
cd ..

# Install Node.js dependencies
npm install
```

## ▶️ Running the Application

### Start Both Services

#### Start Backend Server
```bash
# Navigate to backend directory
cd python_backend

# Activate virtual environment (if not already active)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Start Flask server
python app_cifar_simple.py
```
The backend will start on: **http://localhost:5000**

#### Start Frontend Server
```bash
# In a new terminal, navigate to project root
cd project

# Start React development server
npm run dev
```
The frontend will start on: **http://localhost:5173**



## 📱 Using the Application

1. **Open Browser**: Navigate to http://localhost:5173
2. **Upload Image**: Click on the upload area or drag & drop an image
3. **View Results**: Get instant AI classification with confidence scores
4. **Try Different Images**: Test with various images to see the 10 CIFAR-10 categories

## 🎨 Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP

## 🔧 API Endpoints

### Backend Server (http://localhost:5000)

#### POST /api/classify
Upload and classify an image
```bash
curl -X POST -F "image=@your-image.jpg" http://localhost:5000/api/classify
```

#### GET /api/health
Check server health status
```bash
curl http://localhost:5000/api/health
```

#### GET /api/info
Get system information and capabilities
```bash
curl http://localhost:5000/api/info
```

## 📊 CIFAR-10 Categories

The AI classifies images into these 10 categories:

| Category | Icon | Examples |
|----------|------|----------|
| airplane | ✈️ | Commercial aircraft, jets, planes |
| automobile | 🚗 | Cars, sedans, passenger vehicles |
| bird | 🐦 | Various bird species and types |
| cat | 🐱 | Domestic cats and felines |
| deer | 🦌 | Deer and similar animals |
| dog | 🐕 | Dogs of all breeds and sizes |
| frog | 🐸 | Frogs, toads, amphibians |
| horse | 🐎 | Horses, ponies, equines |
| ship | 🚢 | Ships, boats, watercraft |
| truck | 🚚 | Trucks, lorries, large vehicles |

## 🛠️ Development

### Project Structure
```
project/
├── python_backend/
│   ├── app_cifar_simple.py    # Main Flask server
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── components/
│   │   ├── ClassificationResults.tsx
│   │   ├── Features.tsx
│   │   ├── Header.tsx
│   │   └── ImageUploader.tsx
│   ├── utils/
│   │   └── imageAnalysis.ts
│   ├── App.tsx               # Main React component
│   └── main.tsx              # React entry point
├── package.json              # Node.js dependencies
└── README.md                 # This file
```

### Adding New Features
1. **Backend**: Modify `app_cifar_simple.py` for new API endpoints
2. **Frontend**: Add new components in `src/components/`
3. **Styling**: Update Tailwind classes for UI changes

## 🔍 Troubleshooting

### Common Issues

#### Backend Won't Start
- Ensure Python 3.8+ is installed
- Activate virtual environment: `.venv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`
- Check if port 5000 is available

#### Frontend Won't Start  
- Ensure Node.js 16+ is installed
- Install dependencies: `npm install`
- Check if port 5173 is available

#### Model Loading Issues
- First run downloads ResNet50 model (~100MB)
- Ensure stable internet connection
- Check available disk space

#### Classification Accuracy
- Model works best with clear, well-lit images
- Images should primarily contain one of the 10 CIFAR-10 categories
- Avoid heavily processed or filtered images

## 🌐 Deployment

### Production Build
```bash
# Build frontend for production
npm run build

# Serve with a production server (e.g., nginx, apache)
# Backend can be deployed with gunicorn, uwsgi, etc.
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Naveen-086**
- GitHub: [@Naveen-086](https://github.com/Naveen-086)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **TensorFlow/Keras** for the deep learning framework
- **ResNet50** for the pre-trained model architecture
- **CIFAR-10** for the classification categories
- **React + Vite** for the modern frontend stack
- **Tailwind CSS** for the styling framework

---

**Ready to classify some images? Let's go! 🚀📸**