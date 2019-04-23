const brain = require('brain.js')
const fs = require('fs')

const trainingData = [
  [0, 1],
  [1, 1],
  [1, 2],
  [2, 3],
  [3, 5],
  [5, 8],
  [8, 13],
  [13, 21],
  [21, 34],
  
];

const net = new brain.recurrent.LSTMTimeStep({ hiddenLayers: [12] });

//load existing ann
if (fs.existsSync('neural_network.json')) {
  net.fromJSON(JSON.parse(fs.readFileSync('neural_network.json', 'utf8')));
}


net.train(trainingData, { log: true, errorThresh: 0.0539, learningRate: 0.0003, iterations: 10000 });

let outputs = []
outputs.push(net.run([3, 5]));

for (let i = 0; i < outputs.length; i++) {
  console.log("OUTPUT FOR INPUT #" + (i + 1));
  console.log(" Raw Output: " + (parseFloat(outputs[i])));
  console.log(" Rounded Output: " + Math.round(parseFloat(outputs[i])));
}

//save trained ann
fs.writeFileSync('neural_network.json', JSON.stringify(net.toJSON(), null, ' '));
