
import React from 'react';
import { NeuralNetwork } from './nn';

interface NetworkVisualizerProps {
  nn: NeuralNetwork;
  labels: string[];
}

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ nn, labels }) => {
  const width = 800;
  const height = 450;
  const paddingX = 50;
  const paddingY = 40;
  const availableWidth = width - paddingX * 2 - 120; // Room for labels on right
  const layerGap = availableWidth / (nn.layers.length - 1);
  
  const getNodePos = (layerIdx: number, nodeIdx: number) => {
    const x = paddingX + layerIdx * layerGap;
    const layerSize = nn.layers[layerIdx];
    const nodeGap = (height - paddingY * 2) / (layerSize + 1);
    const y = paddingY + (nodeIdx + 1) * nodeGap;
    return { x, y };
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-xl border border-gray-200" dir="ltr">
      <h3 className="text-lg font-bold mb-4 text-center text-gray-800" dir="rtl">מבנה רשת הנוירונים</h3>
      <div className="w-full">
        <svg 
          viewBox={`0 0 ${width} ${height}`} 
          width="100%" 
          height="auto"
          preserveAspectRatio="xMidYMid meet"
          className="overflow-visible"
        >
          {/* Weights (connections) */}
          {nn.weights.map((layerWeights, lIdx) => 
            layerWeights.map((nodeWeights, nIdx) => 
              nodeWeights.map((weight, wIdx) => {
                const start = getNodePos(lIdx, wIdx);
                const end = getNodePos(lIdx + 1, nIdx);
                const opacity = Math.min(Math.abs(weight), 1);
                const color = weight > 0 ? `rgba(59, 130, 246, ${opacity})` : `rgba(239, 68, 68, ${opacity})`;
                return (
                  <line 
                    key={`w-${lIdx}-${nIdx}-${wIdx}`}
                    x1={start.x} y1={start.y} 
                    x2={end.x} y2={end.y} 
                    stroke={color}
                    strokeWidth={Math.abs(weight) * 2}
                  />
                );
              })
            )
          )}

          {/* Nodes (activations) */}
          {nn.layers.map((layerSize, lIdx) => 
            Array.from({ length: layerSize }).map((_, nIdx) => {
              const pos = getNodePos(lIdx, nIdx);
              const activation = nn.activations[lIdx]?.[nIdx] || 0;
              const size = lIdx === 0 ? 5 : 12;
              return (
                <g key={`n-${lIdx}-${nIdx}`}>
                  <circle 
                    cx={pos.x} cy={pos.y} r={size}
                    fill={`rgba(16, 185, 129, ${activation})`}
                    stroke="#10b981"
                    strokeWidth="2"
                  />
                  {lIdx === nn.layers.length - 1 && (
                    <text 
                      x={pos.x + 18} y={pos.y + 5} 
                      className="text-sm font-bold fill-slate-700"
                      style={{ direction: 'ltr', unicodeBidi: 'bidi-override' }}
                    >
                      {`${labels[nIdx]}: ${(activation * 100).toFixed(1)}%`}
                    </text>
                  )}
                </g>
              );
            })
          )}
        </svg>
      </div>
    </div>
  );
};

export default NetworkVisualizer;
