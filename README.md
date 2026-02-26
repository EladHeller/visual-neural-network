# Visual Neural Network (ANN Visualizer)

An interactive, browser-based neural network visualizer that allows users to draw, label, and train a real artificial neural network (ANN) in real-time.

[**Live Demo**](https://eladheller.github.io/visual-neural-network/)

## Features

- **Interactive Drawing Canvas**: Draw characters or shapes on a 10x10 grid.
- **Real-time Training**: Train the network using Backpropagation and Gradient Descent directly in the browser.
- **Live Visualization**: See the neural network's architecture, weights (line thickness/color), and activations (node brightness) update as it learns.
- **Customizable Labels**: Add up to 5 custom labels to teach the network different patterns.
- **Persistence**: Your training data, labels, and learned weights are automatically saved to `localStorage`.
- **Detailed Architecture View**: Dive deep into the specific numerical values of every weight and bias in the network.
- **Hebrew UI**: Designed with a right-to-left (RTL) interface for Hebrew speakers.

## Tech Stack

- **Framework**: [React 19](https://react.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Bundler**: [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS 4](https://tailwindcss.com/)
- **Deployment**: GitHub Pages

## How it Works

1. **Feedforward**: Your drawing is converted into a 100-pixel input vector. These signals pass through the hidden layer, are multiplied by weights, and transformed by activation functions (Sigmoid) to produce a prediction at the output layer.
2. **Backpropagation**: When you label a drawing, the network calculates the error (Mean Squared Error) between its prediction and the target. It then propagates this error backward to determine how much each weight contributed to the mistake.
3. **Weight Update**: The network slightly adjusts every weight to minimize the error for the next iteration. Over hundreds of iterations, it "learns" to recognize your specific drawing style.

## Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/EladHeller/visual-neural-network.git
   cd visual-neural-network
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## License

MIT
