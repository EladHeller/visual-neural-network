import React, { useState, useEffect, useCallback } from 'react';
import { NeuralNetwork } from './nn';
import DrawingCanvas from './DrawingCanvas';
import NetworkVisualizer from './NetworkVisualizer';
import DetailedVisualizer from './DetailedVisualizer';
import Modal from './Modal';

const GRID_SIZE = 10;
const INPUT_SIZE = GRID_SIZE * GRID_SIZE;

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
  const [currentInput, setCurrentInput] = useState<number[]>(new Array(INPUT_SIZE).fill(0));
  const [loss, setLoss] = useState<number>(() => {
    const saved = localStorage.getItem('ann_loss');
    return saved ? parseFloat(saved) : 0;
  });
  const [trainingData, setTrainingData] = useState<{input: number[], label: number}[]>(() => {
    const saved = localStorage.getItem('ann_data');
    return saved ? JSON.parse(saved) : [];
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
    nn.feedForward(data);
  };

  const resetWeights = () => {
    nn.resetWeights();
    localStorage.removeItem('ann_weights');
    localStorage.removeItem('ann_iterations');
    localStorage.removeItem('ann_loss');
    setIterations(0);
    setLoss(0);
    // Force re-render for visualizer
    nn.feedForward(currentInput);
  };

  const removeSample = (idx: number) => {
    const newData = trainingData.filter((_, i) => i !== idx);
    setTrainingData(newData);
  };

  const addLabel = () => {
    if (newLabelName && labels.length < 5 && !labels.includes(newLabelName)) {
      const updatedLabels = [...labels, newLabelName];
      setLabels(updatedLabels);
      setNewLabelName('');
      setNn(new NeuralNetwork([INPUT_SIZE, 16, updatedLabels.length]));
      setTrainingData([]);
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
      setLabels(updatedLabels);
      setNn(new NeuralNetwork([INPUT_SIZE, 16, updatedLabels.length]));
      setTrainingData([]);
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
    const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
    
    shuffled.forEach(sample => {
      const target = new Array(labels.length).fill(0);
      target[sample.label] = 1;
      totalLoss += nn.train(sample.input, target);
    });
    
    setLoss(totalLoss / trainingData.length);
    setIterations(prev => prev + 1);
    nn.feedForward(currentInput);
  }, [trainingData, nn, currentInput, labels.length]);

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
              <div className="flex gap-2">
                <input
                  type="text"
                  maxLength={1}
                  value={newLabelName}
                  onChange={(e) => setNewLabelName(e.target.value)}
                  placeholder="תו חדש"
                  className="w-24 px-3 py-1 border rounded bg-slate-50 text-slate-800"
                />
                <button
                  onClick={addLabel}
                  disabled={labels.length >= 5 || !newLabelName}
                  className="px-3 py-1 bg-blue-600 text-white rounded disabled:opacity-50 font-bold"
                >
                  + הוסף
                </button>
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-8 items-center">
              <DrawingCanvas onDraw={handleDraw} gridSize={GRID_SIZE} />
              
              <div className="flex flex-col gap-3 w-full max-w-[200px]">
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
                  {isTraining ? '⏸ השהה אימון' : '▶ התחל אימון'}
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
                <button
                  onClick={() => setIsDetailedViewOpen(true)}
                  className="flex-1 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-700 font-bold transition-colors"
                >
                  ארכיטקטורה מפורטת
                </button>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 uppercase tracking-wider font-bold">איטרציות</p>
                  <p className="text-2xl font-mono font-bold text-slate-800">{iterations}</p>
                </div>
                <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 uppercase tracking-wider font-bold">הפסד (MSE)</p>
                  <p className="text-2xl font-mono font-bold text-slate-800">{loss.toFixed(6)}</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200">
            <h2 className="text-xl font-bold mb-4">איך זה עובד?</h2>
            <div className="space-y-3 text-sm text-slate-600 leading-relaxed text-right">
              <p>
                <strong>1. מעבר קדימה (Feedforward):</strong> הציור שלך הופך ל-{INPUT_SIZE} פיקסלים. 
                אלו מפעילים את שכבת הקלט (Input Layer). האותות עוברים דרך המשקולות (Weights), 
                מסוכמים ועוברים טרנספורמציה בכל נוירון (Neuron).
              </p>
              <p>
                <strong>2. פעפוע לאחור (Backpropagation):</strong> כשהרשת מקבלת תיוג, היא בודקת כמה היא טעתה. 
                היא חוזרת אחורה ומבינה אילו משקולות גרמו לטעות.
              </p>
              <p>
                <strong>3. עדכון משקולות:</strong> הרשת משנה מעט כל משקולת כדי להקטין את הטעות בפעם הבאה. 
                אחרי {iterations} איטרציות, היא כבר יודעת לזהות את האותיות שלך!
              </p>
              <div className="pt-3 border-t border-slate-100">
                <p className="font-bold text-slate-800 mb-1">למה הרשת "חיה" גם כשהקנבס ריק?</p>
                <p>
                  גם כשלא ציירת כלום (קלט 0), הרשת עדיין מציגה אחוזים. זה קורה בגלל ה-<strong>Bias (הטיה)</strong>: 
                  לכל נוירון יש "דעה מוקדמת" אקראית שנוספת לחישוב. בנוסף, פונקציית ה-<strong>Sigmoid</strong> תמיד 
                  מחזירה ערך בין 0 ל-1 (למשל, 0.5 עבור קלט של 0), כך שהרשת תמיד "מנחשת" משהו, גם אם זה ניחוש אקראי לחלוטין.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Visualization */}
        <section className="space-y-6">
          <NetworkVisualizer nn={nn} labels={labels} />
          
          <div className="bg-white p-6 rounded-2xl shadow-xl border border-slate-200">
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span className="w-6 h-6 bg-slate-200 text-slate-600 rounded-full flex items-center justify-center text-xs">3</span>
              גלריית דגימות (עד 3 לכל אות)
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4" dir="ltr">
              {trainingData.map((sample, idx) => (
                <div key={`sample-${idx}`} className="relative group bg-slate-50 p-2 rounded-lg border border-slate-100">
                  <div className="grid grid-cols-10 gap-[1px] bg-slate-200 p-[1px] rounded overflow-hidden aspect-square" dir="ltr">
                    {sample.input.map((pixel, pIdx) => (
                      <div 
                        key={pIdx} 
                        className={`w-full h-full ${pixel > 0 ? 'bg-slate-800' : 'bg-white'}`}
                      />
                    ))}
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
