
import React from 'react';
import { NeuralNetwork } from './nn';

interface NetworkVisualizerProps {
  nn: NeuralNetwork;
  labels: string[];
  onOpenDetails?: () => void;
}

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ nn, labels, onOpenDetails }) => {
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
    <div className="bg-white p-6 rounded-2xl shadow-xl border border-gray-200 relative" dir="ltr">
      <div className="flex justify-between items-center mb-4">
        {onOpenDetails && (
          <button 
            onClick={onOpenDetails}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors text-slate-500 hover:text-slate-800"
            title="תצוגה מלאה"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 3 21 3 21 9"></polyline>
              <polyline points="9 21 3 21 3 15"></polyline>
              <line x1="21" y1="3" x2="14" y2="10"></line>
              <line x1="3" y1="21" x2="10" y2="14"></line>
            </svg>
          </button>
        )}
        <h3 className="text-lg font-bold text-center text-gray-800 flex-1" dir="rtl">מבנה רשת הנוירונים</h3>
        {onOpenDetails && <div className="w-9" />} {/* Spacer for centering header */}
      </div>
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
              const size = lIdx === 0 ? 2 : 12;
              return (
                <g key={`n-${lIdx}-${nIdx}`}>
                  <circle 
                    cx={pos.x} cy={pos.y} r={size}
                    fill={`rgba(16, 185, 129, ${activation})`}
                    stroke="#10b981"
                    strokeWidth={lIdx === 0 ? "0.5" : "2"}
                    className="transition-all duration-300 ease-out"
                  />
                  {lIdx === 0 && (nIdx === 0 || nIdx === 50) && (
                    <text
                      x={pos.x - 10} y={pos.y}
                      textAnchor="end"
                      className="text-[10px] fill-slate-400 font-bold"
                      dominantBaseline="middle"
                    >
                      {nIdx === 0 ? 'סכומי שורות' : 'סכומי עמודות'}
                    </text>
                  )}
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
