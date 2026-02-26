
import React from 'react';
import { NeuralNetwork } from './nn';

interface DetailedVisualizerProps {
  nn: NeuralNetwork;
  labels: string[];
}

const DetailedVisualizer: React.FC<DetailedVisualizerProps> = ({ nn, labels }) => {
  const width = 1200;
  const height = 800;
  const paddingX = 100;
  const paddingY = 60;
  const availableWidth = width - paddingX * 2 - 150;
  const layerGap = availableWidth / (nn.layers.length - 1);
  
  const getNodePos = (layerIdx: number, nodeIdx: number) => {
    const x = paddingX + layerIdx * layerGap;
    const layerSize = nn.layers[layerIdx];
    const nodeGap = (height - paddingY * 2) / (layerSize + 1);
    const y = paddingY + (nodeIdx + 1) * nodeGap;
    return { x, y };
  };

  return (
    <div className="overflow-auto max-h-[80vh] bg-slate-900 p-8 rounded-xl" dir="ltr">
      <svg width={width} height={height} className="mx-auto overflow-visible">
        {/* Weights with numbers */}
        {nn.weights.map((layerWeights, lIdx) => 
          layerWeights.map((nodeWeights, nIdx) => 
            nodeWeights.map((weight, wIdx) => {
              const start = getNodePos(lIdx, wIdx);
              const end = getNodePos(lIdx + 1, nIdx);
              const opacity = Math.abs(weight) * 0.5;
              const color = weight > 0 ? `rgba(59, 130, 246, ${opacity})` : `rgba(239, 68, 68, ${opacity})`;
              
              // Only show numbers for significant weights to avoid clutter, 
              // or skip for the huge input layer if needed.
              // Here we show them for connections to the hidden and output layers.
              return (
                <g key={`dw-${lIdx}-${nIdx}-${wIdx}`}>
                  <line 
                    x1={start.x} y1={start.y} 
                    x2={end.x} y2={end.y} 
                    stroke={color}
                    strokeWidth={Math.abs(weight) * 2}
                  />
                  {/* Show weight text near the middle of the line for non-input layers */}
                  {lIdx > 0 && Math.abs(weight) > 0.1 && (
                    <text 
                      x={(start.x + end.x) / 2} 
                      y={(start.y + end.y) / 2}
                      className="text-[8px] fill-slate-400 font-mono"
                      textAnchor="middle"
                    >
                      {weight.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })
          )
        )}

        {/* Nodes */}
        {nn.layers.map((layerSize, lIdx) => 
          Array.from({ length: layerSize }).map((_, nIdx) => {
            const pos = getNodePos(lIdx, nIdx);
            const activation = nn.activations[lIdx]?.[nIdx] || 0;
            const size = lIdx === 0 ? 4 : 12;
            return (
              <g key={`dn-${lIdx}-${nIdx}`}>
                <circle 
                  cx={pos.x} cy={pos.y} r={size}
                  fill={`rgba(16, 185, 129, ${activation})`}
                  stroke="#10b981"
                  strokeWidth="2"
                />
                {/* Activation values */}
                {lIdx > 0 && (
                  <text 
                    x={pos.x} y={pos.y + size + 15} 
                    className="text-[10px] fill-white font-bold"
                    textAnchor="middle"
                    style={{ direction: 'ltr', unicodeBidi: 'bidi-override' }}
                  >
                    {lIdx === nn.layers.length - 1 
                      ? `${labels[nIdx]}: ${(activation * 100).toFixed(1)}%` 
                      : activation.toFixed(3)}
                  </text>
                )}
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
};

export default DetailedVisualizer;
