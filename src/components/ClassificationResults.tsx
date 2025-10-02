import React from 'react';
import { Brain, TrendingUp, Award, Loader2, Camera } from 'lucide-react';
import { ClassificationResult, AnalysisData } from '../utils/imageAnalysis';
import { formatConfidence, getDetectionIcon } from '../utils/imageAnalysis';

interface ClassificationResultsProps {
  results: ClassificationResult[] | null;
  analysis: AnalysisData | null;
  isProcessing: boolean;
  uploadedImage: string | null;
}

const ClassificationResults: React.FC<ClassificationResultsProps> = ({
  results,
  analysis,
  isProcessing,
  uploadedImage
}) => {
  if (!uploadedImage) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Results</h3>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="bg-gray-100 p-4 rounded-full mb-4">
              <Brain className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500 mb-2">No image uploaded yet</p>
            <p className="text-sm text-gray-400">Upload an image to see classification results</p>
          </div>
        </div>
      </div>
    );
  }

  if (isProcessing) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Results</h3>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="bg-blue-100 p-4 rounded-full mb-4">
              <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
            </div>
            <p className="text-gray-900 font-medium mb-2">Analyzing image...</p>
            <p className="text-sm text-gray-500">Computer vision algorithms are processing your image</p>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200">
        <div className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Classification Results</h3>
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="bg-gray-100 p-4 rounded-full mb-4">
              <Brain className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500">Waiting for classification...</p>
          </div>
        </div>
      </div>
    );
  }

  const topResult = results[0];
  const otherResults = results.slice(1);

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-200">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Classification Results</h3>
          <div className="flex items-center space-x-2 text-sm text-green-600 bg-green-50 px-3 py-1 rounded-full">
            <Award className="w-4 h-4" />
            <span>Analysis Complete</span>
          </div>
        </div>

        {/* Top Result */}
        <div className="bg-gradient-to-r from-blue-50 to-emerald-50 rounded-xl p-6 mb-6 border border-blue-100">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">{getDetectionIcon(topResult.type || 'default')}</span>
              <h4 className="text-xl font-bold text-gray-900">{topResult.label}</h4>
            </div>
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-green-600" />
              <span className="text-lg font-bold text-green-600">
                {formatConfidence(topResult.confidence)}
              </span>
            </div>
          </div>
          
          <p className="text-gray-600 mb-3">{topResult.description}</p>
          
          <div className="flex items-center space-x-2 mb-4">
            <span className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              AI Classification
            </span>
          </div>
          
          <div className="bg-white rounded-lg p-3">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-gray-600">Confidence Level</span>
              <span className="font-medium">{formatConfidence(topResult.confidence)}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-1000"
                style={{ width: `${topResult.confidence * 100}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* Analysis Summary */}
        {analysis && (
          <div className="bg-gray-50 rounded-xl p-4 mb-6">
            <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
              <Camera className="w-4 h-4 mr-2" />
              Analysis Summary
            </h5>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="font-semibold text-blue-600">{analysis.processing_time_ms}ms</div>
                <div className="text-gray-500">Processing Time</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-green-600">{analysis.total_predictions || 0}</div>
                <div className="text-gray-500">Results</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-purple-600">
                  {(analysis.top_confidence * 100).toFixed(1)}%
                </div>
                <div className="text-gray-500">Confidence</div>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-gray-200">
              <div className="text-xs text-gray-600">
                <span className="font-medium">Classification: </span>
                CIFAR-10 Neural Network
              </div>
            </div>
          </div>
        )}

        {/* Simplified Results Summary */}
        {otherResults.length > 0 && (
          <div className="bg-gray-50 rounded-xl p-4">
            <h5 className="text-sm font-semibold text-gray-700 mb-3">
              Other Possibilities
            </h5>
            <div className="grid grid-cols-2 gap-2 text-sm">
              {otherResults.slice(0, 4).map((result, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-gray-600">{result.label}</span>
                  <span className="text-gray-500">{formatConfidence(result.confidence)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClassificationResults;