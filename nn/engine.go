package nn

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
)

type Neuron struct {
	W []*Value
	B *Value
}

type Layer struct {
	Neurons []*Neuron
}

type MLP struct {
	Layers []*Layer
}

type MLPState struct {
	NumInputs   int   `json:"num_inputs"`
	LayerSizes  []int `json:"layer_sizes"`
	LayerStates [][]struct {
		Weights []float64 `json:"weights"`
		Bias    float64   `json:"bias"`
	} `json:"layer_states"`
}

func NewNeuron(numInputs int) *Neuron {
	w := make([]*Value, numInputs)
	for i := range w {
		w[i] = &Value{
			Data:     rand.Float64()*2 - 1,
			Grad:     0.0,
			Children: nil,
			Op:       "",
			Label:    fmt.Sprintf("w%d", i),
		}
	}
	b := &Value{
		Data:     rand.Float64()*2 - 1,
		Grad:     0.0,
		Children: nil,
		Op:       "",
		Label:    "b",
	}
	return &Neuron{W: w, B: b}
}

func (n *Neuron) Print() {
	fmt.Printf("Neuron(W: %v, B: %v)\n", n.W, n.B)
}

func (n *Neuron) Call(x []*Value) *Value {
	sum := n.B
	for i, w := range n.W {
		sum = Add(sum, Mul(w, x[i]))
	}
	out := Tanh(sum)
	return out
}

func (n *Neuron) Parameters() []*Value {
	return append(n.W, n.B)
}

func NewLayer(numInputs, numOutputs int) *Layer {
	neurons := make([]*Neuron, numOutputs)
	for i := range neurons {
		neurons[i] = NewNeuron(numInputs)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Print() {
	for _, neuron := range l.Neurons {
		neuron.Print()
	}
}

func (l *Layer) Call(x []*Value) []*Value {
	outputs := make([]*Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Call(x)
	}
	return outputs
}

func (l *Layer) Parameters() []*Value {
	params := []*Value{}
	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

func NewMLP(numInputs int, layerSizes []int) *MLP {
	layers := make([]*Layer, len(layerSizes))

	// current size represents the number of inputs for the next layer
	currentSize := numInputs

	for i, outputSize := range layerSizes {
		layers[i] = NewLayer(currentSize, outputSize)
		currentSize = outputSize
	}

	return &MLP{Layers: layers}
}

func (m *MLP) Print() {
	for _, layer := range m.Layers {
		layer.Print()
	}
}

func (m *MLP) Call(x []*Value) []*Value {
	for _, layer := range m.Layers {
		x = layer.Call(x)
	}
	return x
}

func (m *MLP) Parameters() []*Value {
	params := []*Value{}
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func (m *MLP) Save(filename string) error {
	// Create state object
	state := MLPState{
		LayerSizes: make([]int, len(m.Layers)),
		LayerStates: make([][]struct {
			Weights []float64 `json:"weights"`
			Bias    float64   `json:"bias"`
		}, len(m.Layers)),
	}

	// Get number of inputs from first layer's first neuron
	if len(m.Layers) > 0 && len(m.Layers[0].Neurons) > 0 {
		state.NumInputs = len(m.Layers[0].Neurons[0].W)
	}

	// Save each layer's state
	for i, layer := range m.Layers {
		state.LayerSizes[i] = len(layer.Neurons)
		state.LayerStates[i] = make([]struct {
			Weights []float64 `json:"weights"`
			Bias    float64   `json:"bias"`
		}, len(layer.Neurons))

		for j, neuron := range layer.Neurons {
			weights := make([]float64, len(neuron.W))
			for k, w := range neuron.W {
				weights[k] = w.Data
			}
			state.LayerStates[i][j] = struct {
				Weights []float64 `json:"weights"`
				Bias    float64   `json:"bias"`
			}{
				Weights: weights,
				Bias:    neuron.B.Data,
			}
		}
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling model state: %v", err)
	}

	// Write to file
	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("error writing model state: %v", err)
	}

	return nil
}

func LoadMLP(filename string) (*MLP, error) {
	// Read file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading model state: %v", err)
	}

	// Unmarshal JSON
	var state MLPState
	err = json.Unmarshal(data, &state)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling model state: %v", err)
	}

	// Create new MLP
	mlp := NewMLP(state.NumInputs, state.LayerSizes)

	// Load weights and biases
	for i, layer := range mlp.Layers {
		for j, neuron := range layer.Neurons {
			for k, w := range neuron.W {
				w.Data = state.LayerStates[i][j].Weights[k]
			}
			neuron.B.Data = state.LayerStates[i][j].Bias
		}
	}

	return mlp, nil
}
