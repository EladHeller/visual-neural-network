
import { describe, it, expect, beforeEach } from 'vitest';
import { NeuralNetwork } from './nn';

describe('NeuralNetwork', () => {
  let nn: NeuralNetwork;

  beforeEach(() => {
    // 2 inputs, 2 hidden neurons, 1 output
    nn = new NeuralNetwork([2, 2, 1], 0.1);
  });

  it('should initialize with correct weights and biases shapes', () => {
    expect(nn.weights.length).toBe(2);
    expect(nn.biases.length).toBe(2);
    
    // First layer weights: [rows (next layer), cols (current layer)]
    expect(nn.weights[0].length).toBe(2); // hidden layer neurons
    expect(nn.weights[0][0].length).toBe(2); // input neurons
    
    // Second layer weights: [rows (next layer), cols (current layer)]
    expect(nn.weights[1].length).toBe(1); // output layer neuron
    expect(nn.weights[1][0].length).toBe(2); // hidden layer neurons
  });

  it('should feed forward and produce an output', () => {
    const input = [0.5, 0.8];
    const output = nn.feedForward(input);
    
    expect(output.length).toBe(1);
    expect(output[0]).toBeGreaterThan(0);
    expect(output[0]).toBeLessThan(1);
  });

  it('should calculate loss during training', () => {
    const input = [1, 0];
    const target = [1];
    const loss = nn.train(input, target);
    
    expect(typeof loss).toBe('number');
    expect(loss).toBeGreaterThanOrEqual(0);
  });

  it('should decrease loss over multiple training steps (XOR-ish)', () => {
    // A simple problem: if first input is 1, output should be 1.
    const trainingData = [
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [1] },
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [0] },
    ];
    
    const initialLoss = nn.train(trainingData[0].input, trainingData[0].target);
    
    // Train for 500 iterations
    for (let i = 0; i < 500; i++) {
      trainingData.forEach(sample => {
        nn.train(sample.input, sample.target);
      });
    }
    
    const finalLoss = nn.train(trainingData[0].input, trainingData[0].target);
    expect(finalLoss).toBeLessThan(initialLoss);
  });

  it('should correctly export and import state', () => {
    const originalOutput = nn.feedForward([0.5, 0.5]);
    const state = nn.exportState();
    
    const newNn = new NeuralNetwork([2, 2, 1]);
    newNn.importState(state);
    const newOutput = newNn.feedForward([0.5, 0.5]);
    
    expect(newOutput[0]).toBeCloseTo(originalOutput[0], 10);
  });
});
