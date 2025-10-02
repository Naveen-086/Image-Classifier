import React from 'react';
import { Zap, Brain, Shield, TrendingUp } from 'lucide-react';

const Features: React.FC = () => {
  const features = [
    {
      icon: <Brain className="w-6 h-6" />,
      title: 'Advanced AI Model',
      description: 'State-of-the-art computer vision and neural network algorithms'
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'Lightning Fast',
      description: 'Get classification results in seconds with optimized inference'
    },
    {
      icon: <TrendingUp className="w-6 h-6" />,
      title: 'High Accuracy',
      description: 'Over 95% accuracy on thousands of object categories'
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: 'Secure Processing',
      description: 'Your images are processed securely and not stored on our servers'
    }
  ];

  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
      {features.map((feature, index) => (
        <div
          key={index}
          className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
        >
          <div className="bg-blue-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4 text-blue-600">
            {feature.icon}
          </div>
          <h4 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h4>
          <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
        </div>
      ))}
    </div>
  );
};

export default Features;