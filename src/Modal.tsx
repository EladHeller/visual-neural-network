
import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 backdrop-blur-sm p-4 animate-in fade-in duration-300">
      <div className="bg-slate-900 text-white w-full max-w-6xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh] animate-pop-in">
        <div className="flex justify-between items-center p-6 border-b border-slate-700">
          <h2 className="text-2xl font-bold">{title}</h2>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-lg transition-all text-2xl active-pop"
          >
            ×
          </button>
        </div>
        <div className="p-6 overflow-auto bg-slate-800 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;
