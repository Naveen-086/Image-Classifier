import React, { useCallback, useState } from 'react';
import { Upload, Camera, X, Loader2 } from 'lucide-react';

interface ImageUploaderProps {
  onImageUpload: (file: File, dataUrl: string) => void;
  uploadedImage: string | null;
  fileName: string;
  isProcessing: boolean;
  onReset: () => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageUpload,
  uploadedImage,
  fileName,
  isProcessing,
  onReset
}) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result) {
          onImageUpload(imageFile, e.target.result as string);
        }
      };
      reader.readAsDataURL(imageFile);
    }
  }, [onImageUpload]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result) {
          onImageUpload(file, e.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  if (uploadedImage) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Uploaded Image</h3>
            <button
              onClick={onReset}
              disabled={isProcessing}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          
          <div className="relative">
            <img
              src={uploadedImage}
              alt="Uploaded"
              className="w-full h-64 object-cover rounded-lg"
            />
            {isProcessing && (
              <div className="absolute inset-0 bg-black/50 rounded-lg flex items-center justify-center">
                <div className="bg-white rounded-lg p-4 flex items-center space-x-3">
                  <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                  <span className="text-gray-900 font-medium">Analyzing...</span>
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-4 text-sm text-gray-600">
            <span className="font-medium">File:</span> {fileName}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200">
      <div className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Image</h3>
        
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          className={`
            relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200
            ${isDragOver 
              ? 'border-blue-400 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
            }
          `}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className={`
                p-3 rounded-full transition-colors
                ${isDragOver ? 'bg-blue-100' : 'bg-gray-100'}
              `}>
                <Upload className={`w-8 h-8 ${isDragOver ? 'text-blue-600' : 'text-gray-400'}`} />
              </div>
            </div>
            
            <div>
              <p className="text-lg font-medium text-gray-900 mb-2">
                Drop your image here
              </p>
              <p className="text-gray-500">
                or click to browse files
              </p>
            </div>
            
            <div className="text-sm text-gray-400">
              Supports: JPG, PNG, GIF, WebP
            </div>
          </div>
        </div>
        
        <div className="mt-6 flex items-center justify-center space-x-4 text-sm text-gray-500">
          <div className="flex items-center space-x-2">
            <Camera className="w-4 h-4" />
            <span>Max 10MB</span>
          </div>
          <div className="w-1 h-1 bg-gray-300 rounded-full"></div>
          <div className="flex items-center space-x-2">
            <Upload className="w-4 h-4" />
            <span>Instant results</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageUploader;