
import React, { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import { translations } from './translations';
import type { Language } from './translations';

export interface DrawingCanvasHandle {
  clear: () => void;
}

interface DrawingCanvasProps {
  onDraw: (grid: number[], metadata: { width: number, height: number }) => void;
  size?: number;
  gridSize?: number;
  lang?: Language;
}

const DrawingCanvas = forwardRef<DrawingCanvasHandle, DrawingCanvasProps>(({ onDraw, size = 200, gridSize = 10, lang = 'he' }, ref) => {
  const t = translations[lang];
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);

  const processCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get the full image data to find the bounding box
    const fullImageData = ctx.getImageData(0, 0, size, size);
    const pixels = fullImageData.data;

    let minX = size, minY = size, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = (y * size + x) * 4;
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];
        // If it's not white (drawing is black)
        if (r < 250 || g < 250 || b < 250) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    // Create a temporary small canvas to downsample
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = gridSize;
    tempCanvas.height = gridSize;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    // Fill with white background
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, gridSize, gridSize);

    if (found) {
      const contentWidth = maxX - minX + 1;
      const contentHeight = maxY - minY + 1;
      
      // We want to fit the content into the gridSize x gridSize area, 
      // leaving some padding (e.g., 10%)
      const padding = Math.floor(gridSize * 0.1);
      const targetAreaSize = gridSize - 2 * padding;
      
      const scale = Math.min(targetAreaSize / contentWidth, targetAreaSize / contentHeight);
      const finalWidth = contentWidth * scale;
      const finalHeight = contentHeight * scale;
      
      const offsetX = padding + (targetAreaSize - finalWidth) / 2;
      const offsetY = padding + (targetAreaSize - finalHeight) / 2;

      tempCtx.drawImage(
        canvas,
        minX, minY, contentWidth, contentHeight,
        offsetX, offsetY, finalWidth, finalHeight
      );
    } else {
      // If nothing is drawn, just draw the blank canvas (already filled with white)
    }

    const imageData = tempCtx.getImageData(0, 0, gridSize, gridSize);
    const data = imageData.data;
    
    const grayscale: number[] = [];
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i+1] + data[i+2]) / 3;
      grayscale.push(avg < 128 ? 1 : 0);
    }
    
    // Send both the grid and the original bounding box dimensions
    onDraw(grayscale, { 
      width: found ? (maxX - minX + 1) : 0, 
      height: found ? (maxY - minY + 1) : 0 
    });
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

  useImperativeHandle(ref, () => ({
    clear
  }));

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
    isDrawingRef.current = true;
    draw(e);
  };

  const stopDrawing = () => {
    isDrawingRef.current = false;
    processCanvas();
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawingRef.current) return;
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
    ctx.arc(x, y, 12, 0, Math.PI * 2);
    ctx.fill();
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
        {t.clearCanvas}
      </button>
    </div>
  );
});

export default DrawingCanvas;
