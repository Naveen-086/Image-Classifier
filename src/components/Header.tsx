import React from 'react';
import { Brain } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">AI Vision</h1>
              <p className="text-sm text-gray-500">AI-Powered Image Classification</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
                        <div className="inline-flex items-center px-3 py-2 text-sm font-medium text-green-600 bg-green-50 border border-green-200 rounded-lg">
              <Brain className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">AI Ready</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;