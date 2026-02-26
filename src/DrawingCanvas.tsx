
import React, { useRef, useEffect, useState } from 'react';

interface DrawingCanvasProps {
  onDraw: (data: number[]) => void;
  size?: number;
  gridSize?: number;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ onDraw, size = 200, gridSize = 10 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, size, size);
      }
    }
  }, [size]);

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    if ('touches' in e) {
      e.preventDefault();
    }
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    processCanvas();
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    if ('touches' in e) {
      e.preventDefault();
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = ('touches' in e) ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = ('touches' in e) ? e.touches[0].clientY - rect.top : e.clientY - rect.top;

    ctx.fillStyle = 'black';
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
  };

  const processCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Create a temporary small canvas to downsample
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = gridSize;
    tempCanvas.height = gridSize;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    tempCtx.drawImage(canvas, 0, 0, size, size, 0, 0, gridSize, gridSize);
    const imageData = tempCtx.getImageData(0, 0, gridSize, gridSize);
    const data = imageData.data;
    
    const grayscale: number[] = [];
    for (let i = 0; i < data.length; i += 4) {
      // Convert to 0 (white) or 1 (black)
      const avg = (data[i] + data[i+1] + data[i+2]) / 3;
      grayscale.push(avg < 128 ? 1 : 0);
    }
    onDraw(grayscale);
  };

  const clear = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, size, size);
        processCanvas();
      }
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="border-2 border-gray-400 cursor-crosshair rounded shadow-inner touch-none"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
      />
      <button 
        onClick={clear}
        className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
      >
        נקה לוח
      </button>
    </div>
  );
};

export default DrawingCanvas;
