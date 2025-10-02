// Advanced Computer Vision Image Analysis API
// Updated to work with the new OpenCV + Haar Cascade backend

export interface ClassificationResult {
  label: string;
  confidence: number;
  description: string;
  method: string;
  bbox?: number[];
  type?: string;
}

export interface AnalysisData {
  processing_time_ms: number;
  model_used: string;
  total_predictions: number;
  top_confidence: number;
  tensorflow_available: boolean;
  image_processed: boolean;
  methods_used?: string[];
  image_dimensions?: string;
  total_detections?: number;
  face_count?: number;
  eye_count?: number;
  has_smile?: boolean;
}

export interface ApiResponse {
  success: boolean;
  results: ClassificationResult[];
  analysis: AnalysisData;
  timestamp: string;
  processing_method: string;
  error?: string;
}

const API_BASE_URL = 'http://localhost:5000/api';

export const analyzeImage = async (imageFile: File): Promise<ApiResponse> => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);

    console.log('🔍 Sending image for computer vision analysis...');
    
    const response = await fetch(`${API_BASE_URL}/classify`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Network error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result: ApiResponse = await response.json();
    
    console.log('✅ Computer vision analysis completed:', {
      processingTime: result.analysis?.processing_time_ms + 'ms',
      detections: result.analysis?.total_detections,
      methods: result.analysis?.methods_used
    });

    return result;
  } catch (error) {
    console.error('❌ Error analyzing image:', error);
    throw error;
  }
};

export const analyzeBatchImages = async (imageFiles: File[]): Promise<any> => {
  try {
    const formData = new FormData();
    imageFiles.forEach(file => {
      formData.append('images', file);
    });

    console.log(`🔍 Sending ${imageFiles.length} images for batch analysis...`);

    const response = await fetch(`${API_BASE_URL}/batch_classify`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Network error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    
    console.log('✅ Batch analysis completed:', {
      totalImages: result.total_images,
      successful: result.successful_classifications
    });

    return result;
  } catch (error) {
    console.error('❌ Error analyzing batch images:', error);
    throw error;
  }
};

export const getSystemHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('❌ Error checking system health:', error);
    throw error;
  }
};

export const getSystemCapabilities = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/model_info`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('❌ Error getting system capabilities:', error);
    throw error;
  }
};

// Helper function to format confidence as percentage
export const formatConfidence = (confidence: number): string => {
  return `${(confidence * 100).toFixed(1)}%`;
};

// Helper function to get method color for UI
export const getMethodColor = (method: string): string => {
  switch (method) {
    case 'Haar Cascade':
      return 'bg-blue-100 text-blue-800';
    case 'Corner Detection':
      return 'bg-green-100 text-green-800';
    case 'Edge Analysis':
      return 'bg-purple-100 text-purple-800';
    case 'Color Analysis':
      return 'bg-yellow-100 text-yellow-800';
    case 'Texture Analysis':
      return 'bg-pink-100 text-pink-800';
    case 'Geometric Analysis':
      return 'bg-indigo-100 text-indigo-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

// Helper function to get detection type icon
export const getDetectionIcon = (type: string): string => {
  switch (type) {
    case 'human_face':
      return '👤';
    case 'eye':
      return '👁️';
    case 'smile':
      return '😊';
    case 'structured_object':
      return '🏗️';
    case 'complex_object':
      return '🔧';
    case 'textured_surface':
      return '🎨';
    case 'colorful_image':
      return '🌈';
    case 'landscape_view':
      return '🏞️';
    case 'portrait_view':
      return '🖼️';
    case 'high_resolution':
      return '📸';
    default:
      return '🔍';
  }
};
