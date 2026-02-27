import React, { useState, useEffect, useCallback, useRef } from 'react';
import { NeuralNetwork } from './nn';
import DrawingCanvas from './DrawingCanvas';
import NetworkVisualizer from './NetworkVisualizer';
import DetailedVisualizer from './DetailedVisualizer';
import Modal from './Modal';

const GRID_SIZE = 50;
const INPUT_SIZE = GRID_SIZE * 2; // 50 row sums + 50 col sums

const getFeatures = (grid: number[]): number[] => {
  const rowSums = new Array(GRID_SIZE).fill(0);
  const colSums = new Array(GRID_SIZE).fill(0);
  
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const val = grid[r * GRID_SIZE + c];
      rowSums[r] += val;
      colSums[c] += val;
    }
  }
  
  // Normalize by GRID_SIZE (max sum is GRID_SIZE)
  return [...rowSums.map(s => s / GRID_SIZE), ...colSums.map(s => s / GRID_SIZE)];
};

const SamplePreview: React.FC<{ input: number[], size: number }> = ({ input, size }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, size, size);
        ctx.fillStyle = '#1e293b'; // slate-800
        
        const pixelSize = size / GRID_SIZE;
        for (let i = 0; i < input.length; i++) {
          if (input[i] > 0) {
            const r = Math.floor(i / GRID_SIZE);
            const c = i % GRID_SIZE;
            ctx.fillRect(c * pixelSize, r * pixelSize, pixelSize + 0.5, pixelSize + 0.5);
          }
        }
      }
    }
  }, [input, size]);

  return <canvas ref={canvasRef} width={size} height={size} className="rounded border border-slate-200" />;
};

const LossHistory: React.FC<{ history: number[] }> = ({ history }) => {
  if (history.length < 2) return null;
  const max = Math.max(...history);
  const min = Math.min(...history);
  const range = max - min || 1;
  const width = 200;
  const height = 40;
  const points = history.map((h, i) => ({
    x: (i / (history.length - 1)) * width,
    y: height - ((h - min) / range) * height
  }));
  const path = `M ${points.map(p => `${p.x},${p.y}`).join(' L ')}`;

  return (
    <div className="mt-2">
      <svg width={width} height={height} className="overflow-visible">
        <path d={path} fill="none" stroke="#2563eb" strokeWidth="2" />
      </svg>
    </div>
  );
};

const App: React.FC = () => {
  const [labels, setLabels] = useState<string[]>(() => {
    const saved = localStorage.getItem('ann_labels');
    return saved ? JSON.parse(saved) : ['א', 'ב', 'ט'];
  });
  const [newLabelName, setNewLabelName] = useState('');
  const [nn, setNn] = useState(() => {
    const network = new NeuralNetwork([INPUT_SIZE, 16, labels.length]);
    const savedWeights = localStorage.getItem('ann_weights');
    if (savedWeights) {
      network.importState(JSON.parse(savedWeights));
    }
    return network;
  });
  const [currentInput, setCurrentInput] = useState<number[]>(new Array(GRID_SIZE * GRID_SIZE).fill(0));
  const [loss, setLoss] = useState<number>(() => {
    const saved = localStorage.getItem('ann_loss');
    return saved ? parseFloat(saved) : 0;
  });
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [trainingData, setTrainingData] = useState<{input: number[], label: number}[]>(() => {
    const saved = localStorage.getItem('ann_data');
    if (!saved) return [];
    try {
      const parsed = JSON.parse(saved);
      // Migration: if data is from old 10x10 grid, clear it
      if (parsed.length > 0 && parsed[0].input.length !== GRID_SIZE * GRID_SIZE) {
        localStorage.removeItem('ann_weights');
        localStorage.removeItem('ann_iterations');
        localStorage.removeItem('ann_loss');
        return [];
      }
      return parsed;
    } catch {
      return [];
    }
  });
  const [iterations, setIterations] = useState<number>(() => {
    const saved = localStorage.getItem('ann_iterations');
    return saved ? parseInt(saved, 10) : 0;
  });
  const [isTraining, setIsTraining] = useState(false);
  const [isDetailedViewOpen, setIsDetailedViewOpen] = useState(false);

  // Persistence: Save on change
  useEffect(() => {
    localStorage.setItem('ann_labels', JSON.stringify(labels));
  }, [labels]);

  useEffect(() => {
    localStorage.setItem('ann_data', JSON.stringify(trainingData));
  }, [trainingData]);

  useEffect(() => {
    localStorage.setItem('ann_iterations', iterations.toString());
  }, [iterations]);

  useEffect(() => {
    localStorage.setItem('ann_loss', loss.toString());
  }, [loss]);

  // Save weights when training is paused or active
  useEffect(() => {
    if (isTraining || iterations > 0) {
      localStorage.setItem('ann_weights', JSON.stringify(nn.exportState()));
    }
  }, [isTraining, iterations, nn]);

  const handleDraw = (data: number[]) => {
    setCurrentInput(data);
    nn.feedForward(getFeatures(data));
  };

  const resetWeights = () => {
    nn.resetWeights();
    localStorage.removeItem('ann_weights');
    localStorage.removeItem('ann_iterations');
    localStorage.removeItem('ann_loss');
    setIterations(0);
    setLoss(0);
    setLossHistory([]);
    // Force re-render for visualizer
    nn.feedForward(getFeatures(currentInput));
  };

  const downloadState = () => {
    const state = {
      weights: nn.weights,
      biases: nn.biases,
      labels,
      iterations,
      loss
    };
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ann-brain-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const uploadState = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const state = JSON.parse(event.target?.result as string);
        if (state.weights && state.biases) {
          nn.importState(state);
          if (state.labels) setLabels(state.labels);
          if (state.iterations) setIterations(state.iterations);
          if (state.loss) setLoss(state.loss);
          // Save to local storage
          localStorage.setItem('ann_weights', JSON.stringify(nn.exportState()));
          localStorage.setItem('ann_labels', JSON.stringify(state.labels));
          localStorage.setItem('ann_iterations', state.iterations.toString());
          localStorage.setItem('ann_loss', state.loss.toString());
          nn.feedForward(getFeatures(currentInput));
          alert('המוח נטען בהצלחה!');
        }
      } catch (err) {
        console.error(err);
        alert('שגיאה בטעינת הקובץ.');
      }
    };
    reader.readAsText(file);
  };

  const removeSample = (idx: number) => {
    const newData = trainingData.filter((_, i) => i !== idx);
    setTrainingData(newData);
  };

  const addLabel = () => {
    if (newLabelName && labels.length < 10 && !labels.includes(newLabelName)) {
      const updatedLabels = [...labels, newLabelName];
      setLabels(updatedLabels);
      setNewLabelName('');
      setNn(new NeuralNetwork([INPUT_SIZE, 16, updatedLabels.length]));
      // Note: trainingData is kept so the user doesn't lose their drawings
      localStorage.removeItem('ann_weights');
      localStorage.removeItem('ann_iterations');
      localStorage.removeItem('ann_loss');
      setIsTraining(false);
      setIterations(0);
      setLoss(0);
    }
  };

  const removeLabel = (index: number) => {
    if (labels.length > 1) {
      const updatedLabels = labels.filter((_, i) => i !== index);
      
      // Remove samples of the deleted label and update indices for the rest
      const updatedData = trainingData
        .filter(sample => sample.label !== index)
        .map(sample => ({
          ...sample,
          label: sample.label > index ? sample.label - 1 : sample.label
        }));
      
      setLabels(updatedLabels);
      setTrainingData(updatedData);
      setNn(new NeuralNetwork([INPUT_SIZE, 16, updatedLabels.length]));
      localStorage.removeItem('ann_weights');
      localStorage.removeItem('ann_iterations');
      localStorage.removeItem('ann_loss');
      setIsTraining(false);
      setIterations(0);
      setLoss(0);
    }
  };

  const addTrainingSample = (labelIdx: number) => {
    setTrainingData([...trainingData, { input: currentInput, label: labelIdx }]);
  };

  const trainOnce = useCallback(() => {
    if (trainingData.length === 0) return;
    
    let totalLoss = 0;
    
    // Add hidden empty samples to improve confidence on empty canvas
    // Count is approximately (total samples / number of labels)
    const emptySamplesCount = Math.max(1, Math.floor(trainingData.length / labels.length));
    const emptyFeatures = getFeatures(new Array(GRID_SIZE * GRID_SIZE).fill(0));
    const zeroTarget = new Array(labels.length).fill(0);

    // Create a training batch with real samples and hidden empty samples
    const batch = [
      ...trainingData.map(sample => ({
        features: getFeatures(sample.input),
        target: labels.map((_, i) => i === sample.label ? 1 : 0)
      })),
      ...Array.from({ length: emptySamplesCount }, () => ({
        features: emptyFeatures,
        target: zeroTarget
      }))
    ];

    // Shuffle batch for better SGD
    batch.sort(() => Math.random() - 0.5);
    
    batch.forEach(sample => {
      totalLoss += nn.train(sample.features, sample.target);
    });
    
    const avgLoss = totalLoss / batch.length;
    setLoss(avgLoss);
    setLossHistory(prev => [...prev.slice(-99), avgLoss]);
    setIterations(prev => prev + 1);
    nn.feedForward(getFeatures(currentInput));
  }, [trainingData, nn, currentInput, labels]);

  useEffect(() => {
    let interval: number;
    if (isTraining && trainingData.length > 0) {
      interval = setInterval(trainOnce, 50) as unknown as number;
    }
    return () => clearInterval(interval);
  }, [isTraining, trainingData, trainOnce]);

  return (
    <div className="min-h-screen bg-slate-100 p-8 font-sans text-slate-900" dir="rtl">
      <header className="max-w-6xl mx-auto mb-12 text-center">
        <h1 className="text-4xl font-extrabold mb-4 text-slate-800">המחשת רשתות נוירונים</h1>
        <p className="text-xl text-slate-600">אמן רשת נוירונים אמיתית לזיהוי אותיות וצורות.</p>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Interactive Drawing */}
        <section className="space-y-6">
          <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold flex items-center gap-2">
                <span className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm">1</span>
                צייר ותייג
              </h2>
              <div className="flex flex-col items-end gap-1">
                <div className="flex gap-2">
                  <input
                    type="text"
                    maxLength={1}
                    value={newLabelName}
                    disabled={labels.length >= 10}
                    onChange={(e) => setNewLabelName(e.target.value)}
                    placeholder={labels.length >= 10 ? "מלא" : "תו חדש"}
                    className="w-24 px-3 py-1 border rounded bg-slate-50 text-slate-800 disabled:bg-slate-200 disabled:cursor-not-allowed"
                  />
                  <button
                    onClick={addLabel}
                    disabled={labels.length >= 10 || !newLabelName}
                    className="px-3 py-1 bg-blue-600 text-white rounded disabled:opacity-50 font-bold"
                  >
                    + הוסף
                  </button>
                </div>
                {labels.length >= 10 && (
                  <span className="text-[10px] text-amber-600 font-bold">הגעת למקסימום! יותר מ-10 תוים זה "כבד" מדי למודל הקטן הזה.</span>
                )}
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-8 items-center">
              <DrawingCanvas onDraw={handleDraw} gridSize={GRID_SIZE} />
              
              <div className="flex flex-col gap-3 w-full max-w-[200px]">
                <div className="mb-2 p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="text-xs font-bold text-slate-500 mb-2">תוצאות זיהוי:</p>
                  <div className="space-y-1">
                    {labels.map((label, idx) => {
                      const activation = nn.activations[nn.layers.length - 1]?.[idx] || 0;
                      return (
                        <div key={`result-${idx}`} className="flex items-center justify-between text-sm">
                          <span className="font-bold">{label}:</span>
                          <div className="flex items-center gap-2 flex-1 mx-2">
                            <div className="h-1.5 bg-slate-200 rounded-full flex-1 overflow-hidden">
                              <div 
                                className="h-full bg-blue-500 transition-all duration-300" 
                                style={{ width: `${activation * 100}%` }}
                              />
                            </div>
                          </div>
                          <span className="text-[10px] font-mono w-8 text-left">{(activation * 100).toFixed(0)}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <p className="text-sm font-medium text-slate-500">איסוף דוגמאות:</p>
                {labels.map((label, idx) => (
                  <div key={`${label}-${idx}`} className="flex gap-1">
                    <button
                      onClick={() => addTrainingSample(idx)}
                      className="flex-1 py-2 px-4 bg-slate-800 text-white rounded-lg hover:bg-slate-700 transition-colors font-semibold"
                    >
                      הוסף כ-"{label}"
                    </button>
                    <button 
                      onClick={() => removeLabel(idx)}
                      className="px-2 text-slate-400 hover:text-red-500 transition-colors"
                      title="הסר תגית"
                    >
                      ×
                    </button>
                  </div>
                ))}
                <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-100">
                  <p className="text-xs text-blue-700">סה"כ דגימות: {trainingData.length}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <span className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center text-sm">2</span>
              בקרת אימון
            </h2>
            <div className="space-y-4">
              <div className="flex gap-4">
                <button
                  disabled={trainingData.length === 0}
                  onClick={() => setIsTraining(!isTraining)}
                  className={`flex-1 py-3 rounded-xl font-bold text-lg transition-all ${
                    isTraining 
                    ? 'bg-amber-500 hover:bg-amber-600 text-white' 
                    : 'bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 disabled:cursor-not-allowed'
                  }`}
                >
                  {isTraining ? '⏸ השהה אימון' : iterations > 0 ? '▶ המשך אימון' : '▶ התחל אימון'}
                </button>
                <button
                  onClick={() => {
                    setTrainingData([]);
                    setIsTraining(false);
                    setIterations(0);
                    setLoss(0);
                    localStorage.removeItem('ann_weights');
                    localStorage.removeItem('ann_iterations');
                    localStorage.removeItem('ann_loss');
                    setNn(new NeuralNetwork([INPUT_SIZE, 16, labels.length]));
                  }}
                  className="px-6 py-3 bg-slate-200 text-slate-700 rounded-xl hover:bg-slate-300 font-bold"
                >
                  נקה הכל
                </button>
              </div>
              <div className="flex gap-4">
                <button
                  onClick={resetWeights}
                  className="flex-1 py-2 bg-amber-100 text-amber-700 rounded-lg hover:bg-amber-200 font-bold border border-amber-200 transition-colors"
                >
                  אפס משקולות
                </button>
              </div>
              <div className="flex gap-4 pt-2 border-t border-slate-100">
                <button
                  onClick={downloadState}
                  className="flex-1 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 text-xs font-bold border border-slate-200"
                >
                  💾 הורד מוח
                </button>
                <div className="flex-1 relative">
                  <input
                    type="file"
                    accept=".json"
                    onChange={uploadState}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <button
                    className="w-full py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 text-xs font-bold border border-slate-200 pointer-events-none"
                  >
                    📂 טען מוח
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 uppercase tracking-wider font-bold">איטרציות</p>
                  <p className="text-2xl font-mono font-bold text-slate-800">{iterations}</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 uppercase tracking-wider font-bold">הפסד (MSE)</p>
                  <p className="text-2xl font-mono font-bold text-slate-800">{loss.toFixed(6)}</p>
                  <LossHistory history={lossHistory} />
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200">
            <h2 className="text-xl font-bold mb-4">איך זה עובד?</h2>
            <div className="space-y-3 text-sm text-slate-600 leading-relaxed text-right">
              <p>
                <strong>1. מעבר קדימה (Feedforward):</strong> הציור שלך נדגם ברזולוציה של 50x50 פיקסלים. 
                מתוכם מופקים 100 מאפיינים: סכום הפיקסלים בכל שורה (50) וסכום הפיקסלים בכל עמודה (50). 
                אלו מפעילים את שכבת הקלט. האותות עוברים דרך המשקולות, מסוכמים ועוברים טרנספורמציה בכל נוירון.
              </p>
              <p>
                <strong>2. פעפוע לאחור (Backpropagation):</strong> כשהרשת מקבלת תיוג, היא בודקת כמה היא טעתה. 
                היא חוזרת אחורה ומבינה אילו משקולות גרמו לטעות.
              </p>
              <p>
                <strong>3. עדכון משקולות:</strong> הרשת משנה מעט כל משקולת כדי להקטין את הטעות בפעם הבאה. 
                אחרי כמה מאות איטרציות, היא כבר יודעת לזהות את האותיות שלך!
              </p>
              <div className="pt-3 border-t border-slate-100">
                <p className="font-bold text-slate-800 mb-1">מה זה מדד ההפסד (MSE)?</p>
                <p>
                  <strong>MSE (Mean Squared Error)</strong> הוא "מדד הטעות" של הרשת. ככל שהמספר נמוך יותר, 
                  כך הרשת מדויקת יותר. המדד מחשב את ממוצע ריבועי ההפרשים בין הניבוי של הרשת לבין התיוג האמיתי שלך. 
                  בזמן האימון, תראה את המספר הזה יורד (בשאיפה לאפס) - זה הסימן שהרשת באמת לומדת!
                </p>
              </div>
              <div className="pt-3 border-t border-slate-100">
                <p className="font-bold text-slate-800 mb-1">למה הרשת "שקטה" כשהקנבס ריק?</p>
                <p>
                  כדי להגביר את הדיוק, אנחנו מוסיפים באופן אוטומטי "דגימות ריקות" לתהליך האימון. 
                  זה מלמד את הרשת שאם הקלט הוא אפס (קנבס ריק), עליה להחזיר 0 עבור כל האותיות. 
                  זה משפר משמעותית את הביטחון של הרשת ומפחית זיהויים שגויים.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Visualization */}
        <section className="space-y-6">
          <NetworkVisualizer 
            nn={nn} 
            labels={labels} 
            onOpenDetails={() => setIsDetailedViewOpen(true)}
          />
          
          <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-6 h-6 bg-slate-200 text-slate-600 rounded-full flex items-center justify-center text-xs">3</span>
              גלריית דגימות (עד 10 סוגי תוים)
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4" dir="ltr">
              {trainingData.map((sample, idx) => (
                <div key={`sample-${idx}`} className="relative group bg-slate-50 p-2 rounded-lg border border-slate-100">
                  <div className="aspect-square flex items-center justify-center">
                    <SamplePreview input={sample.input} size={150} />
                  </div>
                  <div className="flex justify-between items-center mt-2">
                    <span className="text-xs font-bold bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                      {labels[sample.label]}
                    </span>
                    <button 
                      onClick={() => removeSample(idx)}
                      className="text-slate-400 hover:text-red-500 text-xs font-bold"
                    >
                      מחק
                    </button>
                  </div>
                </div>
              ))}
              {trainingData.length === 0 && (
                <p className="col-span-full text-center py-8 text-slate-400 text-sm italic">
                  אין דגימות עדיין. הגיע הזמן לצייר!
                </p>
              )}
            </div>
          </div>
        </section>
      </main>

      <Modal 
        isOpen={isDetailedViewOpen} 
        onClose={() => setIsDetailedViewOpen(false)} 
        title="מבט מעמיק על רשת הנוירונים"
      >
        <div className="mb-4 text-slate-300 text-sm text-right">
          <p>כאן ניתן לראות את הערכים המספריים של המשקולות (על הקווים) וערכי ההפעלה (מתחת לעיגולים).</p>
        </div>
        <DetailedVisualizer nn={nn} labels={labels} />
      </Modal>
    </div>
  );
};

export default App;
